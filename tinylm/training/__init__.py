"""Training utilities for TinyLM."""

from tinylm.training.data import (
    CharDataset,
    StreamingDataset,
    build_tokenizer,
    create_dataloaders,
    get_file_size_mb,
)
from tinylm.training.checkpoint import CheckpointManager
from tinylm.training.metrics import MetricsLogger
from tinylm.training.scheduler import get_lr_scheduler, EarlyStopping
from tinylm.training.utils import (
    setup_signal_handlers,
    is_shutdown_requested,
    reset_shutdown_flag,
    evaluate,
    count_parameters,
)

__all__ = [
    # Data
    'CharDataset',
    'StreamingDataset',
    'build_tokenizer',
    'create_dataloaders',
    'get_file_size_mb',
    # Checkpoint
    'CheckpointManager',
    # Metrics
    'MetricsLogger',
    # Scheduler
    'get_lr_scheduler',
    'EarlyStopping',
    # Utils
    'setup_signal_handlers',
    'is_shutdown_requested',
    'reset_shutdown_flag',
    'evaluate',
    'count_parameters',
]
