"""
TinyLM Training Script with Hydra + TensorBoard + Checkpointing.

Usage:
    python train.py                                    # Default config
    python train.py model=medium training=long        # Override configs
    python train.py training.steps=5000 model.dim=512 # Override params
    python train.py resume.enabled=true resume.checkpoint_path=path/to/ckpt.pt
"""

import os
import glob
import logging
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf

from tinylm import TinyLM, build_sincos
from tinylm.quant import QuantConfig

log = logging.getLogger(__name__)


class CharDataset(torch.utils.data.Dataset):
    """Character-level dataset for language modeling."""

    def __init__(self, text: str, seq_len: int, tokenizer: Tokenizer):
        self.seq_len = seq_len
        self.tok = tokenizer
        self.ids = self.tok.encode(text).ids

    def __len__(self):
        return max(0, len(self.ids) - self.seq_len)

    def __getitem__(self, i):
        x = self.ids[i:i + self.seq_len]
        y = self.ids[i + 1:i + self.seq_len + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def build_tokenizer(corpus_paths: list, out_path: str) -> Tokenizer:
    """Build BPE tokenizer from corpus files."""
    tok = Tokenizer(BPE(unk_token="<unk>"))
    tok.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(vocab_size=4096, min_frequency=2, special_tokens=["<unk>"])

    def line_iter():
        for p in corpus_paths:
            with open(p, 'r', encoding='utf-8') as f:
                for line in f:
                    yield line.strip()

    tok.train_from_iterator(line_iter(), trainer=trainer)
    tok.save(out_path)
    return tok


class CheckpointManager:
    """Manages checkpoint saving and loading with rotation."""

    def __init__(self, checkpoint_dir: str, keep_last: int = 3):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last = keep_last
        self.best_loss = float('inf')

    def save(self, state: dict, step: int, is_best: bool = False):
        """Save checkpoint and optionally update best model."""
        # Save regular checkpoint
        ckpt_path = self.checkpoint_dir / f"step_{step:08d}.pt"
        torch.save(state, ckpt_path)
        log.info(f"Saved checkpoint: {ckpt_path}")

        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best.pt"
            torch.save(state, best_path)
            log.info(f"New best model saved: {best_path}")

        # Rotate old checkpoints
        self._rotate_checkpoints()

    def _rotate_checkpoints(self):
        """Keep only the last N checkpoints."""
        ckpts = sorted(glob.glob(str(self.checkpoint_dir / "step_*.pt")))
        while len(ckpts) > self.keep_last:
            oldest = ckpts.pop(0)
            os.remove(oldest)
            log.debug(f"Removed old checkpoint: {oldest}")

    def load(self, checkpoint_path: str) -> dict:
        """Load checkpoint from path."""
        log.info(f"Loading checkpoint: {checkpoint_path}")
        return torch.load(checkpoint_path, map_location='cpu')

    def get_latest(self) -> str | None:
        """Get path to the latest checkpoint."""
        ckpts = sorted(glob.glob(str(self.checkpoint_dir / "step_*.pt")))
        return ckpts[-1] if ckpts else None


class MetricsLogger:
    """Handles TensorBoard and console logging."""

    def __init__(self, log_dir: str, enabled: bool = True):
        self.enabled = enabled
        if enabled:
            self.writer = SummaryWriter(log_dir)
            log.info(f"TensorBoard logging to: {log_dir}")
        else:
            self.writer = None

    def log_scalar(self, tag: str, value: float, step: int):
        if self.writer:
            self.writer.add_scalar(tag, value, step)

    def log_scalars(self, main_tag: str, tag_scalar_dict: dict, step: int):
        if self.writer:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)

    def log_histogram(self, tag: str, values, step: int):
        if self.writer:
            self.writer.add_histogram(tag, values, step)

    def log_hparams(self, hparams: dict, metrics: dict):
        if self.writer:
            self.writer.add_hparams(hparams, metrics)

    def close(self):
        if self.writer:
            self.writer.close()


@torch.no_grad()
def evaluate(model, dataloader, sin, cos, device, use_amp=False):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    n_batches = 0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        if use_amp and device == 'cuda':
            with torch.cuda.amp.autocast():
                logits = model(x, sin, cos)
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), y.view(-1)
                )
        else:
            logits = model(x, sin, cos)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), y.view(-1)
            )

        total_loss += loss.item()
        n_batches += 1

    model.train()
    avg_loss = total_loss / max(1, n_batches)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return avg_loss, perplexity


def get_lr_scheduler(optimizer, cfg):
    """Create learning rate scheduler with optional warmup."""
    warmup_steps = cfg.training.warmup_steps
    total_steps = cfg.training.steps
    min_lr = cfg.training.lr * cfg.training.min_lr_ratio

    if cfg.training.lr_schedule == 'cosine':
        if warmup_steps > 0:
            # Warmup + Cosine decay
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1e-8 / cfg.training.lr,
                end_factor=1.0,
                total_iters=warmup_steps
            )
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_steps - warmup_steps,
                eta_min=min_lr
            )
            return torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup, cosine],
                milestones=[warmup_steps]
            )
        else:
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_steps,
                eta_min=min_lr
            )
    elif cfg.training.lr_schedule == 'linear':
        return torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps if warmup_steps > 0 else total_steps
        )
    elif cfg.training.lr_schedule == 'constant':
        return None
    else:
        return None


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""

    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """Main training function."""
    # Print config
    log.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Set seed for reproducibility
    torch.manual_seed(cfg.experiment.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.experiment.seed)

    # Device setup
    device = cfg.device if torch.cuda.is_available() else 'cpu'
    log.info(f"Using device: {device}")

    # Get Hydra output directory
    hydra_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    checkpoint_dir = hydra_dir / "checkpoints"
    tensorboard_dir = hydra_dir / "tensorboard"

    # Initialize checkpoint manager and metrics logger
    ckpt_manager = CheckpointManager(checkpoint_dir, keep_last=cfg.checkpoint.keep_last)
    metrics_logger = MetricsLogger(str(tensorboard_dir), enabled=cfg.logging.tensorboard)

    # Check data files exist
    if not os.path.exists(cfg.data.train_path):
        raise FileNotFoundError(
            f"Training data not found at {cfg.data.train_path}. "
            f"Run 'python {cfg.data.prepare_script}' first."
        )

    # Build or load tokenizer
    tokenizer_path = "tokenizer.json"
    if not os.path.exists(tokenizer_path):
        log.info("Building tokenizer...")
        build_tokenizer([cfg.data.train_path, cfg.data.val_path], tokenizer_path)
    tokenizer = Tokenizer.from_file(tokenizer_path)

    # Load data
    log.info("Loading data...")
    with open(cfg.data.train_path, 'r', encoding='utf-8') as f:
        train_text = f.read()
    with open(cfg.data.val_path, 'r', encoding='utf-8') as f:
        val_text = f.read()

    train_ds = CharDataset(train_text, cfg.training.seq_len, tokenizer)
    val_ds = CharDataset(val_text, cfg.training.seq_len, tokenizer)

    train_dl = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=cfg.training.batch_size, shuffle=False, drop_last=True)

    # Build quantization config
    quant_config = None
    if cfg.quant.enabled:
        quant_config = QuantConfig(
            enabled=cfg.quant.enabled,
            method=cfg.quant.method,
            threshold_factor=cfg.quant.threshold_factor,
            per_channel=cfg.quant.per_channel,
            backend=cfg.quant.backend,
            quantize_attention=cfg.quant.quantize_attention,
            quantize_mlp=cfg.quant.quantize_mlp,
            quantize_head=cfg.quant.quantize_head,
        )
        log.info(f"Ternary quantization enabled: {quant_config}")

    # Create model
    model = TinyLM(
        vocab_size=tokenizer.get_vocab_size(),
        dim=cfg.model.dim,
        n_layers=cfg.model.n_layers,
        n_heads=cfg.model.n_heads,
        dropout=cfg.model.dropout,
        quant_config=quant_config
    ).to(device)

    if cfg.compile and hasattr(torch, 'compile'):
        log.info("Compiling model with torch.compile...")
        model = torch.compile(model)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"Model parameters: {n_params:,} ({n_params/1e6:.2f}M)")

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
        betas=tuple(cfg.training.betas)
    )

    # Learning rate scheduler
    scheduler = get_lr_scheduler(optimizer, cfg)

    # Early stopping
    early_stopping = None
    if cfg.training.early_stopping_patience > 0:
        early_stopping = EarlyStopping(
            patience=cfg.training.early_stopping_patience,
            min_delta=cfg.training.early_stopping_min_delta
        )
        log.info(f"Early stopping enabled: patience={cfg.training.early_stopping_patience}")

    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.training.mixed_precision) if device == 'cuda' else None

    # RoPE embeddings
    sin, cos = build_sincos(cfg.model.max_seq_len, cfg.model.dim // cfg.model.n_heads, device)

    # Resume from checkpoint if specified
    start_step = 0
    best_val_loss = float('inf')

    if cfg.resume.enabled:
        ckpt_path = cfg.resume.checkpoint_path or ckpt_manager.get_latest()
        if ckpt_path and os.path.exists(ckpt_path):
            checkpoint = ckpt_manager.load(ckpt_path)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if scheduler and 'scheduler' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler'])
            if scaler and 'scaler' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler'])
            start_step = checkpoint.get('step', 0) + 1
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            log.info(f"Resumed from step {start_step}, best_val_loss={best_val_loss:.4f}")
        else:
            log.warning("Resume enabled but no checkpoint found. Starting from scratch.")

    # Training loop
    log.info(f"Starting training from step {start_step} to {cfg.training.steps}")

    step = start_step
    train_iter = iter(train_dl)
    pbar = tqdm(total=cfg.training.steps - start_step, initial=0)
    accum_loss = 0.0

    while step < cfg.training.steps:
        # Get batch
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dl)
            x, y = next(train_iter)

        x, y = x.to(device), y.to(device)

        # Zero gradients at start of accumulation
        if step % cfg.training.grad_accum_steps == 0:
            optimizer.zero_grad(set_to_none=True)

        # Forward pass
        try:
            if cfg.training.mixed_precision and scaler is not None:
                with torch.cuda.amp.autocast():
                    logits = model(x, sin, cos)
                    loss = nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)), y.view(-1)
                    )
                    loss = loss / cfg.training.grad_accum_steps
                scaler.scale(loss).backward()
            else:
                logits = model(x, sin, cos)
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), y.view(-1)
                )
                loss = loss / cfg.training.grad_accum_steps
                loss.backward()

            accum_loss += loss.item() * cfg.training.grad_accum_steps

        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                log.warning(f"OOM at step {step}. Clearing cache.")
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                continue
            raise

        # Update weights after accumulation
        if (step + 1) % cfg.training.grad_accum_steps == 0:
            if cfg.training.mixed_precision and scaler is not None:
                scaler.unscale_(optimizer)
                if cfg.training.grad_clip > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), cfg.training.grad_clip
                    )
                scaler.step(optimizer)
                scaler.update()
            else:
                if cfg.training.grad_clip > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), cfg.training.grad_clip
                    )
                optimizer.step()

            if scheduler:
                scheduler.step()

            train_loss = accum_loss / cfg.training.grad_accum_steps
            accum_loss = 0.0

            # Get current LR
            current_lr = optimizer.param_groups[0]['lr']

            # Log training metrics
            if step % cfg.logging.log_every == 0:
                train_ppl = torch.exp(torch.tensor(train_loss)).item()
                metrics_logger.log_scalar('train/loss', train_loss, step)
                metrics_logger.log_scalar('train/perplexity', train_ppl, step)
                metrics_logger.log_scalar('train/lr', current_lr, step)
                if 'grad_norm' in dir():
                    metrics_logger.log_scalar('train/grad_norm', grad_norm.item(), step)

                pbar.set_description(
                    f"Loss: {train_loss:.3f} | PPL: {train_ppl:.1f} | LR: {current_lr:.2e}"
                )

            # Validation
            if step % cfg.logging.eval_every == 0 and step > 0:
                val_loss, val_ppl = evaluate(
                    model, val_dl, sin, cos, device,
                    use_amp=cfg.training.mixed_precision
                )
                metrics_logger.log_scalar('val/loss', val_loss, step)
                metrics_logger.log_scalar('val/perplexity', val_ppl, step)

                is_best = val_loss < best_val_loss
                if is_best:
                    best_val_loss = val_loss
                    log.info(f"[Step {step}] New best val_loss: {val_loss:.4f} (PPL: {val_ppl:.1f})")

                # Save checkpoint
                if cfg.checkpoint.save_best and is_best:
                    state = {
                        'step': step,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict() if scheduler else None,
                        'scaler': scaler.state_dict() if scaler else None,
                        'best_val_loss': best_val_loss,
                        'config': OmegaConf.to_container(cfg, resolve=True),
                        'tokenizer': tokenizer.to_str(),
                        'quant_config': quant_config.to_dict() if quant_config else None,
                    }
                    ckpt_manager.save(state, step, is_best=True)

                # Early stopping check
                if early_stopping is not None and early_stopping(val_loss):
                    log.info(f"Early stopping triggered at step {step} (patience={cfg.training.early_stopping_patience})")
                    break

            # Periodic checkpoint
            if step % cfg.checkpoint.save_every == 0 and step > 0:
                state = {
                    'step': step,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict() if scheduler else None,
                    'scaler': scaler.state_dict() if scaler else None,
                    'best_val_loss': best_val_loss,
                    'config': OmegaConf.to_container(cfg, resolve=True),
                    'tokenizer': tokenizer.to_str(),
                    'quant_config': quant_config.to_dict() if quant_config else None,
                }
                ckpt_manager.save(state, step, is_best=False)

        step += 1
        pbar.update(1)

    pbar.close()

    # Final evaluation and checkpoint
    final_val_loss, final_val_ppl = evaluate(
        model, val_dl, sin, cos, device,
        use_amp=cfg.training.mixed_precision
    )
    log.info(f"Final val_loss: {final_val_loss:.4f} (PPL: {final_val_ppl:.1f})")
    log.info(f"Best val_loss: {best_val_loss:.4f}")

    # Log hyperparameters
    hparams = {
        'model/dim': cfg.model.dim,
        'model/n_layers': cfg.model.n_layers,
        'model/n_heads': cfg.model.n_heads,
        'training/lr': cfg.training.lr,
        'training/batch_size': cfg.training.batch_size,
        'training/steps': cfg.training.steps,
    }
    metrics = {
        'hparam/best_val_loss': best_val_loss,
        'hparam/final_val_loss': final_val_loss,
    }
    metrics_logger.log_hparams(hparams, metrics)
    metrics_logger.close()

    log.info(f"Training complete! Output dir: {hydra_dir}")


if __name__ == '__main__':
    main()
