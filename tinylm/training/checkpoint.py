"""Checkpoint management for training."""

import glob
import logging
import os
from pathlib import Path

import torch

log = logging.getLogger(__name__)


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
