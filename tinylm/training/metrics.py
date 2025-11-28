"""Metrics logging for training."""

import logging
from torch.utils.tensorboard import SummaryWriter

log = logging.getLogger(__name__)


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

    def flush(self):
        if self.writer:
            self.writer.flush()

    def close(self):
        if self.writer:
            self.writer.close()
