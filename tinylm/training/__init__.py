"""Training utilities for TinyLM."""

from tinylm.training.checkpoint import CheckpointManager
from tinylm.training.data import (
    CharDataset,
    StreamingDataset,
    build_tokenizer,
    create_dataloaders,
    get_file_size_mb,
)
from tinylm.training.metrics import MetricsLogger
from tinylm.training.optimizers import Muon, build_optimizer
from tinylm.training.scheduler import EarlyStopping, get_lr_scheduler
from tinylm.training.trainer import Trainer, TrainerConfig, TrainerState
from tinylm.training.utils import (
    count_parameters,
    evaluate,
    is_shutdown_requested,
    reset_shutdown_flag,
    setup_signal_handlers,
)

__all__ = [
    # Trainer
    'Trainer',
    'TrainerConfig',
    'TrainerState',
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
    # Optimizers
    'Muon',
    'build_optimizer',
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
