"""Learning rate scheduling and early stopping."""

import torch


def get_lr_scheduler(optimizer, total_steps: int, warmup_steps: int = 0,
                     schedule: str = 'cosine', min_lr_ratio: float = 0.1,
                     base_lr: float = 3e-4):
    """Create learning rate scheduler with optional warmup.

    Args:
        optimizer: PyTorch optimizer
        total_steps: Total training steps
        warmup_steps: Number of warmup steps
        schedule: 'cosine', 'linear', or 'constant'
        min_lr_ratio: Minimum LR as ratio of base LR
        base_lr: Base learning rate

    Returns:
        LR scheduler or None for constant schedule
    """
    min_lr = base_lr * min_lr_ratio

    if schedule == 'cosine':
        if warmup_steps > 0:
            # Warmup + Cosine decay
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1e-8 / base_lr,
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
    elif schedule == 'linear':
        return torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps if warmup_steps > 0 else total_steps
        )
    elif schedule == 'constant':
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
        """Check if training should stop.

        Args:
            val_loss: Current validation loss

        Returns:
            True if training should stop
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop

    def reset(self):
        """Reset early stopping state."""
        self.counter = 0
        self.best_loss = float('inf')
        self.should_stop = False
