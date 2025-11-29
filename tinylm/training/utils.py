"""Training utilities."""

import os
import signal
import logging

import torch
import torch.nn as nn

log = logging.getLogger(__name__)

# Graceful shutdown flag
_shutdown_requested = False


def _signal_handler(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True
    log.warning("Shutdown requested, will save checkpoint after current step...")


def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown."""
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)


def is_shutdown_requested() -> bool:
    """Check if shutdown was requested."""
    return _shutdown_requested


def reset_shutdown_flag():
    """Reset the shutdown flag."""
    global _shutdown_requested
    _shutdown_requested = False


@torch.no_grad()
def evaluate(model, dataloader, sin, cos, device, use_amp=False, max_batches=None, log_progress=False):
    """Evaluate model on validation set.

    Args:
        model: The model to evaluate
        dataloader: Validation data loader
        sin, cos: RoPE embeddings
        device: Device to run on
        use_amp: Whether to use automatic mixed precision
        max_batches: Maximum batches to evaluate (None for all)
        log_progress: Log progress every 100 batches

    Returns:
        Tuple of (average_loss, perplexity)
    """
    model.eval()
    total_loss = 0
    n_batches = 0

    for x, y in dataloader:
        if max_batches and n_batches >= max_batches:
            break
        if is_shutdown_requested():
            break

        if log_progress and n_batches > 0 and n_batches % 100 == 0:
            log.info(f"  Eval progress: {n_batches} batches...")

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


def count_parameters(model) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
