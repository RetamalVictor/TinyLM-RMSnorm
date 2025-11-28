"""
Benchmark RMSNorm CUDA kernel performance against PyTorch reference.

This refactored version uses the benchmark base class to eliminate duplication.
"""

import argparse
import time
import torch
import torch.nn as nn
from typing import List, Tuple, Dict

from benchmark_base import BenchmarkConfig, BenchmarkBase
from model import RMSNormCUDA


class RMSNormRef(nn.Module):
    """Reference RMSNorm implementation using PyTorch ops."""

    def __init__(self, dim: int, eps: float = 1e-6):
        """Initialize RMSNorm layer.

        Args:
            dim: Dimension to normalize
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMSNorm to input.

        Args:
            x: Input tensor of shape [..., dim]

        Returns:
            Normalized tensor of same shape
        """
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class RMSNormBenchmarkRunner(BenchmarkBase):
    """Runner for RMSNorm kernel benchmarks."""

    def __init__(self, config: BenchmarkConfig, args):
        """Initialize with config and additional arguments."""
        super().__init__(config)
        self.args = args

    def benchmark_module(
        self,
        module: nn.Module,
        shape: Tuple[int, int, int],
        iters: int = 100
    ) -> Dict[str, float]:
        """Benchmark a normalization module with statistics.

        Args:
            module: Module to benchmark
            shape: Input shape (batch, seq_len, hidden_dim)
            iters: Iterations per measurement run

        Returns:
            Dictionary with timing statistics in milliseconds
        """
        device, dtype = self.get_device_dtype()
        B, T, C = shape
        x = torch.randn(B, T, C, device=device, dtype=dtype, requires_grad=False)

        # Define the benchmark function
        def run_forward():
            for _ in range(iters):
                _ = module(x)

        # Measure with statistics
        stats = self.measure_with_stats(
            run_forward,
            n_runs=self.args.n_runs,
            warmup=3
        )

        # Convert to ms per iteration
        ms_stats = {
            'mean': stats['mean'] * 1000.0 / iters,
            'std': stats['std'] * 1000.0 / iters,
            'min': stats['min'] * 1000.0 / iters,
            'max': stats['max'] * 1000.0 / iters
        }

        return ms_stats

    def run(self) -> Tuple[Tuple, List[Tuple]]:
        """Run the RMSNorm benchmark.

        Returns:
            Tuple of (headers, results)
        """
        device, dtype = self.get_device_dtype()

        # Test shapes
        shapes = [
            (16, 256, 512),
            (16, 256, 1024),
            (16, 256, 2048),
            (8, 512, 1024)
        ]

        results = []
        headers = ('B', 'T', 'C', 'dtype', 'op', 'ms_per_iter', 'std_ms', 'speedup')

        print(f"\nRMSNorm Kernel Benchmark (dtype={self.config.dtype}):")
        print("-" * 60)

        for B, T, C in shapes:
            # Create modules
            ref_module = RMSNormRef(C).to(device).to(dtype)
            fused_module = RMSNormCUDA(C).to(device).to(dtype)

            # Benchmark both implementations
            ref_stats = self.benchmark_module(
                ref_module, (B, T, C), self.args.iters
            )
            fused_stats = self.benchmark_module(
                fused_module, (B, T, C), self.args.iters
            )

            # Calculate speedup
            speedup = ref_stats['mean'] / max(fused_stats['mean'], 1e-9)

            # Print results
            print(f"Shape ({B:2}, {T:3}, {C:4}):")
            print(f"  Reference: {ref_stats['mean']:6.3f} ± {ref_stats['std']:.3f} ms")
            print(f"  Fused:     {fused_stats['mean']:6.3f} ± {fused_stats['std']:.3f} ms")
            print(f"  Speedup:   {speedup:6.2f}x")

            # Store results
            results.append((
                B, T, C, self.config.dtype, 'ref',
                f'{ref_stats["mean"]:.4f}',
                f'{ref_stats["std"]:.4f}',
                '1.00'
            ))
            results.append((
                B, T, C, self.config.dtype, 'fused',
                f'{fused_stats["mean"]:.4f}',
                f'{fused_stats["std"]:.4f}',
                f'{speedup:.2f}'
            ))

        return headers, results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='RMSNorm kernel benchmark')

    # Model configuration
    parser.add_argument('--dtype', default='fp16', choices=['fp16', 'fp32', 'bf16'],
                       help='Data type for benchmarking')
    parser.add_argument('--device', default='cuda', help='Device to use')

    # Benchmark parameters
    parser.add_argument('--iters', type=int, default=200,
                       help='Iterations per measurement')
    parser.add_argument('--n_runs', type=int, default=10,
                       help='Number of runs for statistics')

    # Output
    parser.add_argument('--label', type=str, help='Device label')
    parser.add_argument('--out', default='out/rmsnorm_bench.csv',
                       help='Output CSV path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # Create configuration
    config = BenchmarkConfig(
        checkpoint='',  # Not needed for this benchmark
        device=args.device,
        dtype=args.dtype,
        label=args.label,
        output_dir='out',
        seed=args.seed
    )

    # Run benchmark
    runner = RMSNormBenchmarkRunner(config, args)
    headers, results = runner.run()

    # Write results
    runner.write_csv(args.out, results, headers)

    print(f"\nBenchmark complete! Results saved to {args.out}")


if __name__ == "__main__":
    main()