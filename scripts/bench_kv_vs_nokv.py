"""
Benchmark KV-cache vs no-KV-cache performance comparison.

This refactored version uses the benchmark base class to eliminate duplication
and provides proper statistical measurements.
"""

import argparse
from typing import List, Tuple

import torch
from benchmark_base import BenchmarkConfig, KVCacheBenchmark


class KVComparisonRunner(KVCacheBenchmark):
    """Runner for KV-cache comparison benchmarks."""

    def __init__(self, config: BenchmarkConfig, args):
        """Initialize with config and additional arguments."""
        super().__init__(config)
        self.args = args
        self.warmup = 20

    def benchmark_with_kv(self) -> Tuple[float, float]:
        """Benchmark with KV-cache enabled.

        Returns:
            Tuple of (mean_tps, std_tps)
        """
        device, _ = self.get_device_dtype()

        # Load checkpoint for tokenizer if needed
        if self.tokenizer is None:
            self.load_checkpoint()

        # Encode prompt
        ids = torch.tensor(
            self.tokenizer.encode(self.args.prompt).ids,
            device=device
        ).unsqueeze(0)

        # Prepare RoPE tables
        max_len = ids.size(1) + self.warmup + self.args.steps
        sin, cos = self.prepare_rope_tables(max_len)

        # Pre-allocate cache
        cache = self.create_kv_cache(1, max_len)

        # Prefill cache
        _ = self.model(ids, sin, cos, cache, start_pos=0)

        # Warmup incremental decoding
        for _ in range(self.warmup):
            logits = self.model(
                ids[:, -1:], sin, cos, cache,
                start_pos=ids.size(1) - 1
            )[:, -1, :]
            ids = torch.cat([
                ids,
                torch.argmax(logits, dim=-1, keepdim=True)
            ], dim=1)

        # Measure with multiple runs
        def run_with_kv():
            nonlocal ids
            temp_ids = ids.clone()
            for _ in range(self.args.steps):
                logits = self.model(
                    temp_ids[:, -1:], sin, cos, cache,
                    start_pos=temp_ids.size(1) - 1
                )[:, -1, :]
                temp_ids = torch.cat([
                    temp_ids,
                    torch.argmax(logits, dim=-1, keepdim=True)
                ], dim=1)

        stats = self.measure_with_stats(
            run_with_kv,
            n_runs=self.args.n_runs,
            warmup=2
        )

        mean_tps = self.args.steps / stats['mean']
        std_tps = self.args.steps * stats['std'] / (stats['mean'] ** 2)

        return mean_tps, std_tps

    def benchmark_no_kv(self) -> Tuple[float, float]:
        """Benchmark without KV-cache (full recomputation).

        Returns:
            Tuple of (mean_tps, std_tps)
        """
        device, _ = self.get_device_dtype()

        # Load checkpoint for tokenizer if needed
        if self.tokenizer is None:
            self.load_checkpoint()

        # Encode prompt
        ids = torch.tensor(
            self.tokenizer.encode(self.args.prompt).ids,
            device=device
        ).unsqueeze(0)

        # Prepare RoPE tables
        max_len = 8192
        sin, cos = self.prepare_rope_tables(max_len)

        # Warmup
        tmp = ids.clone()
        for _ in range(5):
            logits = self.model(
                tmp, sin, cos, cache=None, start_pos=0
            )[:, -1, :]
            tmp = torch.cat([
                tmp,
                torch.argmax(logits, dim=-1, keepdim=True)
            ], dim=1)

        # Measure with multiple runs
        def run_no_kv():
            temp_ids = ids.clone()
            for _ in range(self.args.steps):
                logits = self.model(
                    temp_ids, sin, cos, cache=None, start_pos=0
                )[:, -1, :]
                temp_ids = torch.cat([
                    temp_ids,
                    torch.argmax(logits, dim=-1, keepdim=True)
                ], dim=1)

        stats = self.measure_with_stats(
            run_no_kv,
            n_runs=self.args.n_runs,
            warmup=2
        )

        mean_tps = self.args.steps / stats['mean']
        std_tps = self.args.steps * stats['std'] / (stats['mean'] ** 2)

        return mean_tps, std_tps

    def run(self) -> Tuple[Tuple, List[Tuple]]:
        """Run the KV-cache comparison benchmark.

        Returns:
            Tuple of (headers, results)
        """
        # Create model
        self.create_model(dropout=0.0)

        print("\nKV-Cache Comparison Benchmark")
        print(f"  Prompt: '{self.args.prompt}'")
        print(f"  Steps: {self.args.steps}")
        print(f"  Data type: {self.config.dtype}")
        print()

        # Benchmark with KV-cache
        kv_mean, kv_std = self.benchmark_with_kv()
        print(f"With KV-cache:    {kv_mean:7.2f} ± {kv_std:5.2f} tokens/sec")

        # Benchmark without KV-cache
        nokv_mean, nokv_std = self.benchmark_no_kv()
        print(f"Without KV-cache: {nokv_mean:7.2f} ± {nokv_std:5.2f} tokens/sec")

        # Calculate speedup
        speedup = kv_mean / max(nokv_mean, 1e-9)
        print(f"Speedup:          {speedup:7.2f}x")

        # Prepare results
        headers = ('label', 'mode', 'steps', 'dtype', 'tokens_per_sec', 'std_dev')
        results = [
            (
                self.config.label,
                'with_kv',
                self.args.steps,
                self.config.dtype,
                f'{kv_mean:.2f}',
                f'{kv_std:.2f}'
            ),
            (
                self.config.label,
                'no_kv',
                self.args.steps,
                self.config.dtype,
                f'{nokv_mean:.2f}',
                f'{nokv_std:.2f}'
            )
        ]

        return headers, results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='KV-cache vs no-cache comparison')

    # Checkpoint and model
    parser.add_argument('--ckpt', required=True, help='Path to checkpoint')
    parser.add_argument('--dtype', default='fp16', choices=['fp16', 'fp32', 'bf16'])
    parser.add_argument('--device', default='cuda', help='Device to use')

    # Benchmark parameters
    parser.add_argument('--steps', type=int, default=256,
                       help='Number of generation steps')
    parser.add_argument('--n_runs', type=int, default=10,
                       help='Number of runs for statistics')

    # Other options
    parser.add_argument('--prompt', default='Once upon a time',
                       help='Prompt to use')
    parser.add_argument('--label', type=str, help='Device label')
    parser.add_argument('--out', default='out/kv_vs_nokv.csv',
                       help='Output CSV path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # Create configuration
    config = BenchmarkConfig(
        checkpoint=args.ckpt,
        device=args.device,
        dtype=args.dtype,
        label=args.label,
        output_dir='out',
        seed=args.seed
    )

    # Run benchmark
    runner = KVComparisonRunner(config, args)
    headers, results = runner.run()

    # Append to CSV (preserving original behavior)
    runner.append_csv(args.out, results, headers)

    print(f"\nBenchmark complete! Results appended to {args.out}")


if __name__ == "__main__":
    main()
