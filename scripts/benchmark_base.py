"""
Base utilities for benchmark scripts to eliminate code duplication.

Provides common functionality for:
- Model loading from checkpoints
- Configuration handling
- CSV writing with proper formatting
- Statistical measurements
"""

import os
import sys
import torch
import csv
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

# Add parent directory to path for imports
ROOT = Path(__file__).parent.parent
if ROOT not in sys.path:
    sys.path.insert(0, str(ROOT))

from tokenizers import Tokenizer
from model import TinyLM, build_sincos, prealloc_kvcache


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarks."""
    checkpoint: str
    device: str = 'cuda'
    dtype: str = 'fp16'
    label: Optional[str] = None
    output_dir: str = 'out'
    seed: int = 42

    def __post_init__(self):
        """Set default label from GPU name if not provided."""
        if self.label is None and self.device == 'cuda':
            try:
                import subprocess
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                    capture_output=True, text=True
                )
                if result.returncode == 0:
                    self.label = result.stdout.strip().replace(' ', '_')
            except:
                self.label = 'gpu'
        elif self.label is None:
            self.label = 'cpu'


class BenchmarkBase:
    """Base class for benchmarks with common functionality."""

    def __init__(self, config: BenchmarkConfig):
        """Initialize benchmark with configuration.

        Args:
            config: Benchmark configuration
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.model_config = None
        self._setup()

    def _setup(self):
        """Setup model, tokenizer, and configuration."""
        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)

        # Set random seeds
        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)
            np.random.seed(self.config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.config.seed)

    def load_checkpoint(self) -> Dict[str, Any]:
        """Load checkpoint and extract components.

        Returns:
            Dictionary containing checkpoint data

        Raises:
            FileNotFoundError: If checkpoint doesn't exist
            RuntimeError: If checkpoint is invalid
        """
        if not os.path.exists(self.config.checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {self.config.checkpoint}")

        try:
            checkpoint = torch.load(self.config.checkpoint, map_location='cpu')
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint: {e}")

        # Extract tokenizer
        if 'tok' not in checkpoint:
            raise ValueError("Checkpoint missing tokenizer")
        self.tokenizer = Tokenizer.from_str(checkpoint['tok'])

        # Extract model configuration
        self.model_config = checkpoint.get('config')
        if self.model_config is None:
            # Use default configuration if not present
            self.model_config = {
                'dim': 384,
                'n_layers': 6,
                'n_heads': 6,
                'vocab_size': self.tokenizer.get_vocab_size()
            }

        return checkpoint

    def create_model(self, dropout: float = 0.0) -> TinyLM:
        """Create and initialize model from checkpoint.

        Args:
            dropout: Dropout probability (default 0.0 for inference)

        Returns:
            Initialized model
        """
        checkpoint = self.load_checkpoint()

        # Create model
        self.model = TinyLM(
            vocab_size=self.model_config['vocab_size'],
            dim=self.model_config['dim'],
            n_layers=self.model_config['n_layers'],
            n_heads=self.model_config['n_heads'],
            dropout=dropout
        )

        # Move to device
        device = torch.device(self.config.device)
        self.model = self.model.to(device).eval()

        # Load state dict
        state_dict = checkpoint['model']
        # Handle compiled model state dicts
        if any(k.startswith('_orig_mod.') for k in state_dict):
            state_dict = {
                k.replace('_orig_mod.', '', 1): v
                for k, v in state_dict.items()
            }
        self.model.load_state_dict(state_dict, strict=False)

        # Convert to specified dtype
        if self.config.dtype == 'fp16':
            self.model = self.model.half()
        elif self.config.dtype == 'bf16':
            self.model = self.model.bfloat16()

        return self.model

    def write_csv(self, filepath: str, rows: List[Tuple], headers: Optional[Tuple] = None):
        """Write benchmark results to CSV.

        Args:
            filepath: Path to output CSV
            rows: Data rows to write
            headers: Optional header row
        """
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            if headers:
                writer.writerow(headers)
            writer.writerows(rows)
        print(f"Wrote results to {filepath}")

    def append_csv(self, filepath: str, rows: List[Tuple], headers: Optional[Tuple] = None):
        """Append benchmark results to existing CSV.

        Args:
            filepath: Path to output CSV
            rows: Data rows to append
            headers: Header row (written only if file doesn't exist)
        """
        file_exists = os.path.exists(filepath)
        mode = 'a' if file_exists else 'w'

        with open(filepath, mode, newline='') as f:
            writer = csv.writer(f)
            if not file_exists and headers:
                writer.writerow(headers)
            writer.writerows(rows)

        action = "Appended to" if file_exists else "Created"
        print(f"{action} {filepath}")

    @staticmethod
    def measure_with_stats(
        func,
        n_runs: int = 5,
        warmup: int = 2
    ) -> Dict[str, float]:
        """Measure function execution time with statistics.

        Args:
            func: Function to benchmark
            n_runs: Number of measurement runs
            warmup: Number of warmup runs

        Returns:
            Dictionary with mean, std, min, max timings
        """
        import time

        # Warmup runs
        for _ in range(warmup):
            func()
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        # Measurement runs
        timings = []
        for _ in range(n_runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            start = time.perf_counter()
            func()

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            end = time.perf_counter()
            timings.append(end - start)

        timings = np.array(timings)
        return {
            'mean': timings.mean(),
            'std': timings.std(),
            'min': timings.min(),
            'max': timings.max(),
            'median': np.median(timings)
        }

    def get_device_dtype(self) -> Tuple[torch.device, torch.dtype]:
        """Get device and dtype for tensors.

        Returns:
            Tuple of (device, dtype)
        """
        device = torch.device(self.config.device)

        if self.config.dtype == 'fp16':
            dtype = torch.float16
        elif self.config.dtype == 'bf16':
            dtype = torch.bfloat16
        else:
            dtype = torch.float32

        return device, dtype


class KVCacheBenchmark(BenchmarkBase):
    """Specialized benchmark for KV-cache measurements."""

    def prepare_rope_tables(self, max_seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare RoPE sin/cos tables.

        Args:
            max_seq_len: Maximum sequence length

        Returns:
            Tuple of (sin, cos) tensors
        """
        if self.model is None:
            self.create_model()

        device, dtype = self.get_device_dtype()
        head_dim = self.model_config['dim'] // self.model_config['n_heads']

        sin, cos = build_sincos(max_seq_len, head_dim, device)
        return sin.to(dtype), cos.to(dtype)

    def create_kv_cache(
        self,
        batch_size: int,
        max_seq_len: int
    ) -> Dict[str, torch.Tensor]:
        """Create pre-allocated KV cache.

        Args:
            batch_size: Batch size
            max_seq_len: Maximum sequence length

        Returns:
            Dictionary with 'k' and 'v' cache tensors
        """
        device, dtype = self.get_device_dtype()
        head_dim = self.model_config['dim'] // self.model_config['n_heads']

        return prealloc_kvcache(
            batch_size,
            max_seq_len,
            self.model_config['n_heads'],
            head_dim,
            device.type,
            dtype
        )