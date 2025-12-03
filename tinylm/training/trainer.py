"""Trainer class for TinyLM with hooks for distributed training."""

import logging
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, Any, List

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from tinylm.training.utils import is_shutdown_requested

log = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    """Configuration for the Trainer."""

    # Training parameters
    total_steps: int = 10000
    grad_accum_steps: int = 1
    grad_clip: float = 1.0
    mixed_precision: bool = False

    # Logging
    log_every: int = 10
    eval_every: int = 100
    max_eval_batches: Optional[int] = None

    # Checkpointing
    save_every: int = 500
    save_best: bool = True

    # Device
    device: str = "cuda"


@dataclass
class TrainerState:
    """Mutable state for the Trainer."""

    step: int = 0
    best_val_loss: float = float("inf")
    accum_loss: float = 0.0
    grad_norm: Optional[float] = None
    should_stop: bool = False


class Trainer:
    """Trainer class with hooks for distributed training.

    This class encapsulates the training loop and provides hooks
    for extending functionality (logging, checkpointing, distributed).

    Hooks:
        on_train_start: Called before training starts
        on_step_start: Called before each training step
        on_step_end: Called after each training step (with metrics)
        on_eval: Called after evaluation (with metrics)
        on_checkpoint: Called when saving a checkpoint
        on_train_end: Called when training ends

    Example:
        >>> trainer = Trainer(model, optimizer, config)
        >>> trainer.add_hook("on_step_end", lambda t, m: print(f"Loss: {m['loss']}"))
        >>> trainer.train(train_dl, val_dl)
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        config: TrainerConfig,
        scheduler: Optional[LRScheduler] = None,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
    ):
        """Initialize the Trainer.

        Args:
            model: The model to train
            optimizer: The optimizer
            config: Trainer configuration
            scheduler: Optional learning rate scheduler
            scaler: Optional gradient scaler for mixed precision
        """
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.scheduler = scheduler
        self.scaler = scaler

        # State
        self.state = TrainerState()

        # Hooks
        self._hooks: Dict[str, List[Callable]] = {
            "on_train_start": [],
            "on_step_start": [],
            "on_step_end": [],
            "on_eval": [],
            "on_checkpoint": [],
            "on_train_end": [],
        }

        # Distributed training attributes
        self._is_distributed = False
        self._world_size = 1
        self._rank = 0
        self._local_rank = 0

    # -------------------------------------------------------------------------
    # Distributed Training Support
    # -------------------------------------------------------------------------

    def setup_distributed(
        self,
        backend: str = "nccl",
        init_method: Optional[str] = None,
    ) -> None:
        """Setup distributed training.

        This is a no-op by default. Subclasses or extensions can override
        to setup DDP/FSDP.

        Args:
            backend: Distributed backend ("nccl", "gloo", "mpi")
            init_method: URL for process group initialization
        """
        # No-op by default - DDP/FSDP will be implemented in Issue 10/11
        log.debug(f"setup_distributed called (backend={backend}) - no-op in base Trainer")

    def wrap_model(
        self,
        wrapper: Optional[str] = None,
        **kwargs,
    ) -> nn.Module:
        """Wrap the model for distributed training.

        Args:
            wrapper: Wrapper type ("ddp", "fsdp", or None)
            **kwargs: Additional arguments for the wrapper

        Returns:
            The wrapped (or original) model
        """
        if wrapper is None:
            return self.model

        if wrapper == "ddp":
            # Will be implemented in Issue 10
            log.warning("DDP wrapper not yet implemented (Issue 10)")
            return self.model
        elif wrapper == "fsdp":
            # Will be implemented in Issue 11
            log.warning("FSDP wrapper not yet implemented (Issue 11)")
            return self.model
        else:
            raise ValueError(f"Unknown wrapper: {wrapper}")

    @property
    def is_main_process(self) -> bool:
        """Check if this is the main process (rank 0)."""
        return self._rank == 0

    # -------------------------------------------------------------------------
    # Hooks
    # -------------------------------------------------------------------------

    def add_hook(self, event: str, callback: Callable) -> None:
        """Add a callback hook for an event.

        Args:
            event: Event name (on_train_start, on_step_end, etc.)
            callback: Callable that takes (trainer, metrics_dict)
        """
        if event not in self._hooks:
            raise ValueError(f"Unknown hook event: {event}")
        self._hooks[event].append(callback)

    def _call_hooks(self, event: str, metrics: Optional[Dict[str, Any]] = None) -> None:
        """Call all registered hooks for an event."""
        metrics = metrics or {}
        for callback in self._hooks[event]:
            try:
                callback(self, metrics)
            except Exception as e:
                log.warning(f"Hook {event} raised exception: {e}")

    # -------------------------------------------------------------------------
    # Training Loop
    # -------------------------------------------------------------------------

    def train_step(
        self,
        batch: tuple,
    ) -> Dict[str, float]:
        """Execute a single training step.

        Args:
            batch: Tuple of (input, target) tensors

        Returns:
            Dict with metrics (loss, etc.)
        """
        x, y = batch
        x = x.to(self.config.device)
        y = y.to(self.config.device)

        # Zero gradients at start of accumulation
        if self.state.step % self.config.grad_accum_steps == 0:
            self.optimizer.zero_grad(set_to_none=True)

        # Forward pass
        if self.config.mixed_precision and self.scaler is not None:
            with torch.cuda.amp.autocast():
                logits = self.model(x)
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), y.view(-1)
                )
                loss = loss / self.config.grad_accum_steps
            self.scaler.scale(loss).backward()
        else:
            logits = self.model(x)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), y.view(-1)
            )
            loss = loss / self.config.grad_accum_steps
            loss.backward()

        self.state.accum_loss += loss.item() * self.config.grad_accum_steps

        metrics = {"loss": loss.item() * self.config.grad_accum_steps}

        # Update weights after accumulation
        if (self.state.step + 1) % self.config.grad_accum_steps == 0:
            metrics.update(self._optimizer_step())

        return metrics

    def _optimizer_step(self) -> Dict[str, float]:
        """Execute optimizer step with gradient clipping.

        Returns:
            Dict with metrics (train_loss, lr, grad_norm)
        """
        metrics = {}

        if self.config.mixed_precision and self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
            if self.config.grad_clip > 0:
                self.state.grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip
                ).item()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            if self.config.grad_clip > 0:
                self.state.grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip
                ).item()
            self.optimizer.step()

        if self.scheduler:
            self.scheduler.step()

        # Compute metrics
        train_loss = self.state.accum_loss / self.config.grad_accum_steps
        self.state.accum_loss = 0.0

        metrics["train_loss"] = train_loss
        metrics["lr"] = self.optimizer.param_groups[0]["lr"]
        if self.state.grad_norm is not None:
            metrics["grad_norm"] = self.state.grad_norm

        return metrics

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        max_batches: Optional[int] = None,
    ) -> Dict[str, float]:
        """Evaluate the model on a validation set.

        Args:
            dataloader: Validation data loader
            max_batches: Maximum batches to evaluate (None for all)

        Returns:
            Dict with metrics (val_loss, val_perplexity)
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        max_batches = max_batches or self.config.max_eval_batches

        for x, y in dataloader:
            if max_batches and n_batches >= max_batches:
                break
            if is_shutdown_requested():
                break

            x = x.to(self.config.device)
            y = y.to(self.config.device)

            if self.config.mixed_precision and self.config.device == "cuda":
                with torch.cuda.amp.autocast():
                    logits = self.model(x)
                    loss = nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)), y.view(-1)
                    )
            else:
                logits = self.model(x)
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), y.view(-1)
                )

            total_loss += loss.item()
            n_batches += 1

        self.model.train()

        avg_loss = total_loss / max(1, n_batches)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        return {
            "val_loss": avg_loss,
            "val_perplexity": perplexity,
        }

    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        start_step: int = 0,
    ) -> Dict[str, float]:
        """Run the full training loop.

        Args:
            train_dataloader: Training data loader
            val_dataloader: Optional validation data loader
            start_step: Step to start from (for resuming)

        Returns:
            Dict with final metrics
        """
        self.state.step = start_step
        self.model.train()

        # Call on_train_start hooks
        self._call_hooks("on_train_start", {"start_step": start_step})

        train_iter = iter(train_dataloader)

        while self.state.step < self.config.total_steps:
            # Check for shutdown
            if is_shutdown_requested() or self.state.should_stop:
                log.info(f"Training stopped at step {self.state.step}")
                break

            # Get batch
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_dataloader)
                batch = next(train_iter)

            # Call on_step_start hooks
            self._call_hooks("on_step_start", {"step": self.state.step})

            # Training step
            try:
                metrics = self.train_step(batch)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    log.warning(f"OOM at step {self.state.step}. Clearing cache.")
                    self.optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    self.state.step += 1
                    continue
                raise

            metrics["step"] = self.state.step

            # Call on_step_end hooks (only on optimizer steps)
            if (self.state.step + 1) % self.config.grad_accum_steps == 0:
                optimizer_step = (self.state.step + 1) // self.config.grad_accum_steps
                metrics["optimizer_step"] = optimizer_step
                self._call_hooks("on_step_end", metrics)

                # Evaluation
                if (
                    val_dataloader is not None
                    and optimizer_step % self.config.eval_every == 0
                    and optimizer_step > 0
                ):
                    eval_metrics = self.evaluate(val_dataloader)
                    eval_metrics["optimizer_step"] = optimizer_step

                    # Update best val loss
                    is_best = eval_metrics["val_loss"] < self.state.best_val_loss
                    if is_best:
                        self.state.best_val_loss = eval_metrics["val_loss"]
                    eval_metrics["is_best"] = is_best

                    self._call_hooks("on_eval", eval_metrics)

            self.state.step += 1

        # Call on_train_end hooks
        final_metrics = {
            "final_step": self.state.step,
            "best_val_loss": self.state.best_val_loss,
        }
        self._call_hooks("on_train_end", final_metrics)

        return final_metrics

    # -------------------------------------------------------------------------
    # Checkpointing
    # -------------------------------------------------------------------------

    def state_dict(self) -> Dict[str, Any]:
        """Get trainer state for checkpointing.

        Returns:
            Dict containing model, optimizer, scheduler, scaler states
        """
        state = {
            "step": self.state.step,
            "best_val_loss": self.state.best_val_loss,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        if self.scheduler:
            state["scheduler"] = self.scheduler.state_dict()
        if self.scaler:
            state["scaler"] = self.scaler.state_dict()
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load trainer state from checkpoint.

        Args:
            state: State dict from state_dict()
        """
        self.state.step = state.get("step", 0)
        self.state.best_val_loss = state.get("best_val_loss", float("inf"))
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        if self.scheduler and "scheduler" in state:
            self.scheduler.load_state_dict(state["scheduler"])
        if self.scaler and "scaler" in state:
            self.scaler.load_state_dict(state["scaler"])

    def stop(self) -> None:
        """Request training to stop after current step."""
        self.state.should_stop = True
