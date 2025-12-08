"""TinyLM CLI tools.

This module contains command-line interface tools:
- train: Training script with Hydra configuration
- infer: Inference/generation script
- evaluate: Model evaluation and registry management
"""

from tinylm.cli.train import main as train_main
from tinylm.cli.infer import main as infer_main
from tinylm.cli.evaluate import main as evaluate_main

__all__ = ["train_main", "infer_main", "evaluate_main"]
