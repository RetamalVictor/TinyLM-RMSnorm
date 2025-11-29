"""
TinyLM Training Script with Hydra + TensorBoard + Checkpointing.

Usage:
    python train.py                                    # Default config
    python train.py model=medium training=long        # Override configs
    python train.py training.steps=5000 model.dim=512 # Override params
    python train.py resume.enabled=true resume.checkpoint_path=path/to/ckpt.pt
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from tokenizers import Tokenizer
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf

from tinylm import TinyLM, build_sincos
from tinylm.quant import QuantConfig
from tinylm.training import (
    build_tokenizer,
    create_dataloaders,
    CheckpointManager,
    MetricsLogger,
    get_lr_scheduler,
    EarlyStopping,
    setup_signal_handlers,
    is_shutdown_requested,
    evaluate,
    count_parameters,
)

log = logging.getLogger(__name__)

# Setup graceful shutdown
setup_signal_handlers()


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

    # Build or load tokenizer (stored in Hydra output dir)
    tokenizer_path = hydra_dir / "tokenizer.json"
    if not tokenizer_path.exists():
        log.info("Building tokenizer...")
        build_tokenizer([cfg.data.train_path, cfg.data.val_path], str(tokenizer_path))
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    # Create dataloaders
    train_dl, val_dl = create_dataloaders(
        train_path=cfg.data.train_path,
        val_path=cfg.data.val_path,
        tokenizer=tokenizer,
        seq_len=cfg.training.seq_len,
        batch_size=cfg.training.batch_size,
        log_fn=log.info,
    )

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
    n_params = count_parameters(model)
    log.info(f"Model parameters: {n_params:,} ({n_params/1e6:.2f}M)")

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
        betas=tuple(cfg.training.betas)
    )

    # Learning rate scheduler
    scheduler = get_lr_scheduler(
        optimizer,
        total_steps=cfg.training.steps,
        warmup_steps=cfg.training.warmup_steps,
        schedule=cfg.training.lr_schedule,
        min_lr_ratio=cfg.training.min_lr_ratio,
        base_lr=cfg.training.lr,
    )

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
    grad_norm = None

    # Initial validation (baseline)
    max_eval_batches = cfg.logging.get('max_eval_batches', None)
    if start_step == 0:
        log.info(f"Running initial validation (max_batches={max_eval_batches})...")
        val_loss, val_ppl = evaluate(model, val_dl, sin, cos, device, use_amp=cfg.training.mixed_precision, max_batches=max_eval_batches)
        metrics_logger.log_scalar('val/loss', val_loss, 0)
        metrics_logger.log_scalar('val/perplexity', val_ppl, 0)
        metrics_logger.flush()
        best_val_loss = val_loss
        log.info(f"[Step 0] Initial val_loss: {val_loss:.4f} (PPL: {val_ppl:.1f})")

    while step < cfg.training.steps:
        # Check for graceful shutdown
        if is_shutdown_requested():
            log.info(f"Graceful shutdown at step {step}")
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
            break

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

            # Compute optimizer step (actual update count)
            optimizer_step = (step + 1) // cfg.training.grad_accum_steps

            # Log training metrics
            if optimizer_step % cfg.logging.log_every == 0:
                train_ppl = torch.exp(torch.tensor(train_loss)).item()
                metrics_logger.log_scalar('train/loss', train_loss, optimizer_step)
                metrics_logger.log_scalar('train/perplexity', train_ppl, optimizer_step)
                metrics_logger.log_scalar('train/lr', current_lr, optimizer_step)
                if grad_norm is not None:
                    metrics_logger.log_scalar('train/grad_norm', grad_norm.item(), optimizer_step)

                pbar.set_description(
                    f"Loss: {train_loss:.3f} | PPL: {train_ppl:.1f} | LR: {current_lr:.2e}"
                )

            # Validation
            if optimizer_step % cfg.logging.eval_every == 0 and optimizer_step > 0:
                val_loss, val_ppl = evaluate(
                    model, val_dl, sin, cos, device,
                    use_amp=cfg.training.mixed_precision,
                    max_batches=max_eval_batches
                )
                metrics_logger.log_scalar('val/loss', val_loss, optimizer_step)
                metrics_logger.log_scalar('val/perplexity', val_ppl, optimizer_step)
                metrics_logger.flush()

                is_best = val_loss < best_val_loss
                if is_best:
                    best_val_loss = val_loss
                    log.info(f"[Step {optimizer_step}] New best val_loss: {val_loss:.4f} (PPL: {val_ppl:.1f})")

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
                    log.info(f"Early stopping triggered at step {optimizer_step} (patience={cfg.training.early_stopping_patience})")
                    break

            # Periodic checkpoint
            if optimizer_step % cfg.checkpoint.save_every == 0 and optimizer_step > 0:
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

    # Final evaluation and checkpoint (use more batches for accurate final score)
    final_eval_batches = max_eval_batches * 5 if max_eval_batches else 500  # 5x normal or 500
    log.info(f"Running final evaluation (max_batches={final_eval_batches})...")
    final_val_loss, final_val_ppl = evaluate(
        model, val_dl, sin, cos, device,
        use_amp=cfg.training.mixed_precision,
        max_batches=final_eval_batches,
        log_progress=True
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
