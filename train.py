"""
TinyLM Training Script with Hydra + TensorBoard + Checkpointing.

Usage:
    # Train from scratch
    python train.py                                    # Default config
    python train.py model=medium training=long        # Override configs
    python train.py data=tinystories tokenizer=bytelevel  # Data + tokenizer

    # Resume training (same data, continue from step N)
    python train.py resume.enabled=true resume.checkpoint_path=path/to/ckpt.pt

    # Fine-tune (new data, load weights only, fresh optimizer)
    python train.py finetune=full finetune.checkpoint_path=path/to/ckpt.pt data=tinystories
    python train.py finetune=freeze_early finetune.checkpoint_path=path/to/ckpt.pt
    python train.py finetune=freeze_embeddings finetune.checkpoint_path=path/to/ckpt.pt
"""

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from tokenizers import Tokenizer
from torch.optim import AdamW
from tqdm import tqdm

from tinylm import TinyLM
from tinylm.architectures import ArchitectureConfig, get_architecture
from tinylm.kernels import available_backends, set_backend
from tinylm.quant import QuantConfig
from tinylm.training import (
    CheckpointManager,
    EarlyStopping,
    MetricsLogger,
    Trainer,
    TrainerConfig,
    build_tokenizer,
    count_parameters,
    create_dataloaders,
    get_lr_scheduler,
    setup_signal_handlers,
)

log = logging.getLogger(__name__)

# Setup graceful shutdown
setup_signal_handlers()


def create_trainer_hooks(
    cfg: DictConfig,
    metrics_logger: MetricsLogger,
    ckpt_manager: CheckpointManager,
    early_stopping: EarlyStopping | None,
    tokenizer: Tokenizer,
    arch_config: ArchitectureConfig,
    quant_config: QuantConfig | None,
    pbar: tqdm,
):
    """Create hook functions for the Trainer.

    Returns dict of hook callbacks that integrate with TinyLM's
    logging, checkpointing, and early stopping infrastructure.
    """

    def on_step_end(trainer: Trainer, metrics: dict):
        """Called after each optimizer step."""
        optimizer_step = metrics.get("optimizer_step", 0)

        # Log metrics
        if optimizer_step % cfg.logging.log_every == 0:
            train_loss = metrics.get("train_loss", 0)
            train_ppl = torch.exp(torch.tensor(train_loss)).item()
            current_lr = metrics.get("lr", 0)

            metrics_logger.log_scalar("train/loss", train_loss, optimizer_step)
            metrics_logger.log_scalar("train/perplexity", train_ppl, optimizer_step)
            metrics_logger.log_scalar("train/lr", current_lr, optimizer_step)

            grad_norm = metrics.get("grad_norm")
            if grad_norm is not None:
                metrics_logger.log_scalar("train/grad_norm", grad_norm, optimizer_step)

            pbar.set_description(
                f"Loss: {train_loss:.3f} | PPL: {train_ppl:.1f} | LR: {current_lr:.2e}"
            )

        pbar.update(cfg.training.grad_accum_steps)

    def on_eval(trainer: Trainer, metrics: dict):
        """Called after evaluation."""
        optimizer_step = metrics.get("optimizer_step", 0)
        val_loss = metrics["val_loss"]
        val_ppl = metrics["val_perplexity"]
        is_best = metrics.get("is_best", False)

        metrics_logger.log_scalar("val/loss", val_loss, optimizer_step)
        metrics_logger.log_scalar("val/perplexity", val_ppl, optimizer_step)
        metrics_logger.flush()

        if is_best:
            log.info(f"[Step {optimizer_step}] New best val_loss: {val_loss:.4f} (PPL: {val_ppl:.1f})")

            # Save best checkpoint
            if cfg.checkpoint.save_best:
                state = _build_checkpoint_state(trainer, cfg, tokenizer, arch_config, quant_config)
                ckpt_manager.save(state, trainer.state.step, is_best=True)

        # Early stopping check
        if early_stopping is not None and early_stopping(val_loss):
            log.info(
                f"Early stopping triggered at step {optimizer_step} "
                f"(patience={cfg.training.early_stopping_patience})"
            )
            trainer.stop()

        # Periodic checkpoint
        if optimizer_step % cfg.checkpoint.save_every == 0 and optimizer_step > 0:
            state = _build_checkpoint_state(trainer, cfg, tokenizer, arch_config, quant_config)
            ckpt_manager.save(state, trainer.state.step, is_best=False)

    def on_train_end(trainer: Trainer, metrics: dict):
        """Called when training ends."""
        pbar.close()

    return {
        "on_step_end": on_step_end,
        "on_eval": on_eval,
        "on_train_end": on_train_end,
    }


def _build_checkpoint_state(
    trainer: Trainer,
    cfg: DictConfig,
    tokenizer: Tokenizer,
    arch_config: ArchitectureConfig,
    quant_config: QuantConfig | None,
) -> dict:
    """Build checkpoint state dict."""
    state = trainer.state_dict()
    state["config"] = OmegaConf.to_container(cfg, resolve=True)
    state["tokenizer"] = tokenizer.to_str()
    state["quant_config"] = quant_config.to_dict() if quant_config else None
    state["arch_config"] = arch_config.to_dict()
    return state


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
    device = cfg.device if torch.cuda.is_available() else "cpu"
    log.info(f"Using device: {device}")

    # Setup kernel backend
    kernel_backend = cfg.get("kernels", {}).get("backend", "auto")
    try:
        set_backend(kernel_backend)
        log.info(f"Kernel backend: {kernel_backend} (available: {available_backends()})")
    except ValueError as e:
        log.warning(f"Could not set kernel backend '{kernel_backend}': {e}. Using auto.")
        set_backend("auto")

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

    # Check if fine-tuning - load checkpoint once for tokenizer and arch_config
    finetune_cfg = getattr(cfg, "finetune", None)
    finetune_ckpt = None
    if finetune_cfg and finetune_cfg.checkpoint_path:
        log.info(f"Loading checkpoint for fine-tuning: {finetune_cfg.checkpoint_path}")
        finetune_ckpt = torch.load(finetune_cfg.checkpoint_path, map_location="cpu")
        # Load tokenizer from checkpoint (must match!)
        tokenizer = Tokenizer.from_str(finetune_ckpt["tokenizer"])
        tokenizer.save(str(tokenizer_path))
    elif not tokenizer_path.exists():
        log.info(f"Building tokenizer (type={cfg.tokenizer.type}, vocab_size={cfg.tokenizer.vocab_size})...")
        build_tokenizer(
            [cfg.data.train_path, cfg.data.val_path],
            str(tokenizer_path),
            vocab_size=cfg.tokenizer.vocab_size,
            tokenizer_type=cfg.tokenizer.type,
        )
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    else:
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

    # Get architecture config
    arch_name = cfg.model.get("architecture", "llama")
    if finetune_ckpt is not None:
        if "arch_config" in finetune_ckpt:
            arch_config = ArchitectureConfig.from_dict(finetune_ckpt["arch_config"])
            arch_name = arch_config.name
        else:
            # Legacy checkpoint without arch_config - assume llama
            arch_config = get_architecture("llama")
            arch_name = "llama"
            log.warning("Checkpoint missing arch_config, assuming llama architecture")
    else:
        arch_config = get_architecture(arch_name)
    log.info(f"Architecture: {arch_config}")

    # Create model
    model = TinyLM(
        vocab_size=tokenizer.get_vocab_size(),
        dim=cfg.model.dim,
        n_layers=cfg.model.n_layers,
        n_heads=cfg.model.n_heads,
        max_seq_len=cfg.model.max_seq_len,
        dropout=cfg.model.dropout,
        arch_config=arch_config,
        quant_config=quant_config,
    ).to(device)

    # Fine-tuning: load weights and freeze layers
    if finetune_ckpt is not None:
        log.info("Loading model weights for fine-tuning...")

        # Load model weights
        state_dict = finetune_ckpt["model"]
        # Handle compiled model prefix
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        log.info("Loaded model weights from checkpoint")

        # Freeze layers as specified
        frozen_params = 0
        if finetune_cfg.freeze_embeddings:
            for param in model.tok.parameters():
                param.requires_grad = False
                frozen_params += param.numel()
            log.info("Frozen: embeddings")

        if finetune_cfg.freeze_head:
            for param in model.head.parameters():
                param.requires_grad = False
                frozen_params += param.numel()
            log.info("Frozen: output head")

        if finetune_cfg.freeze_layers:
            for layer_idx in finetune_cfg.freeze_layers:
                if layer_idx < len(model.blocks):
                    for param in model.blocks[layer_idx].parameters():
                        param.requires_grad = False
                        frozen_params += param.numel()
            log.info(f"Frozen: layers {list(finetune_cfg.freeze_layers)}")

        log.info(f"Frozen parameters: {frozen_params:,}")

    if cfg.compile and hasattr(torch, "compile"):
        log.info("Compiling model with torch.compile...")
        model = torch.compile(model)

    # Count parameters (trainable only when fine-tuning)
    n_params = count_parameters(model)
    log.info(f"Model parameters: {n_params:,} ({n_params/1e6:.2f}M)")

    # Optimizer (only trainable params)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    log.info(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    optimizer = AdamW(
        trainable_params,
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
        betas=tuple(cfg.training.betas),
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
            min_delta=cfg.training.early_stopping_min_delta,
        )
        log.info(f"Early stopping enabled: patience={cfg.training.early_stopping_patience}")

    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.training.mixed_precision) if device == "cuda" else None

    # Create Trainer config
    trainer_config = TrainerConfig(
        total_steps=cfg.training.steps,
        grad_accum_steps=cfg.training.grad_accum_steps,
        grad_clip=cfg.training.grad_clip,
        mixed_precision=cfg.training.mixed_precision,
        log_every=cfg.logging.log_every,
        eval_every=cfg.logging.eval_every,
        max_eval_batches=cfg.logging.get("max_eval_batches", None),
        save_every=cfg.checkpoint.save_every,
        save_best=cfg.checkpoint.save_best,
        device=device,
    )

    # Create Trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        config=trainer_config,
        scheduler=scheduler,
        scaler=scaler,
    )

    # Resume from checkpoint if specified
    start_step = 0
    if cfg.resume.enabled:
        ckpt_path = cfg.resume.checkpoint_path or ckpt_manager.get_latest()
        if ckpt_path and os.path.exists(ckpt_path):
            checkpoint = ckpt_manager.load(ckpt_path)
            trainer.load_state_dict(checkpoint)
            start_step = trainer.state.step + 1
            log.info(f"Resumed from step {start_step}, best_val_loss={trainer.state.best_val_loss:.4f}")
        else:
            log.warning("Resume enabled but no checkpoint found. Starting from scratch.")

    # Create progress bar
    pbar = tqdm(total=cfg.training.steps - start_step, initial=0)

    # Create and register hooks
    hooks = create_trainer_hooks(
        cfg=cfg,
        metrics_logger=metrics_logger,
        ckpt_manager=ckpt_manager,
        early_stopping=early_stopping,
        tokenizer=tokenizer,
        arch_config=arch_config,
        quant_config=quant_config,
        pbar=pbar,
    )
    for event, callback in hooks.items():
        trainer.add_hook(event, callback)

    # Initial validation (baseline)
    max_eval_batches = cfg.logging.get("max_eval_batches", None)
    if start_step == 0:
        log.info(f"Running initial validation (max_batches={max_eval_batches})...")
        eval_metrics = trainer.evaluate(val_dl, max_batches=max_eval_batches)
        val_loss, val_ppl = eval_metrics["val_loss"], eval_metrics["val_perplexity"]
        metrics_logger.log_scalar("val/loss", val_loss, 0)
        metrics_logger.log_scalar("val/perplexity", val_ppl, 0)
        metrics_logger.flush()
        trainer.state.best_val_loss = val_loss
        log.info(f"[Step 0] Initial val_loss: {val_loss:.4f} (PPL: {val_ppl:.1f})")

    # Run training
    log.info(f"Starting training from step {start_step} to {cfg.training.steps}")
    final_metrics = trainer.train(train_dl, val_dl, start_step=start_step)

    # Final evaluation (use more batches for accurate final score)
    final_eval_batches = max_eval_batches * 5 if max_eval_batches else 500
    log.info(f"Running final evaluation (max_batches={final_eval_batches})...")
    final_eval = trainer.evaluate(val_dl, max_batches=final_eval_batches)
    final_val_loss, final_val_ppl = final_eval["val_loss"], final_eval["val_perplexity"]
    log.info(f"Final val_loss: {final_val_loss:.4f} (PPL: {final_val_ppl:.1f})")
    log.info(f"Best val_loss: {trainer.state.best_val_loss:.4f}")

    # Log hyperparameters
    hparams = {
        "model/dim": cfg.model.dim,
        "model/n_layers": cfg.model.n_layers,
        "model/n_heads": cfg.model.n_heads,
        "model/architecture": arch_name,
        "training/lr": cfg.training.lr,
        "training/batch_size": cfg.training.batch_size,
        "training/steps": cfg.training.steps,
    }
    metrics = {
        "hparam/best_val_loss": trainer.state.best_val_loss,
        "hparam/final_val_loss": final_val_loss,
    }
    metrics_logger.log_hparams(hparams, metrics)
    metrics_logger.close()

    log.info(f"Training complete! Output dir: {hydra_dir}")


if __name__ == "__main__":
    main()
