"""
Benchmark decoding throughput (tokens per second).

This refactored version uses the benchmark base class to eliminate duplication.
"""

import argparse
import time
import torch
from typing import List, Tuple

from benchmark_base import BenchmarkConfig, KVCacheBenchmark


class DecodeThroughputRunner(KVCacheBenchmark):
    """Runner for decode throughput benchmarks."""

    def __init__(self, config: BenchmarkConfig, args):
        """Initialize with config and additional arguments."""
        super().__init__(config)
        self.args = args
        self.warmup_steps = 20

    def run(self) -> Tuple[Tuple, List[Tuple]]:
        """Run the decode throughput benchmark.

        Returns:
            Tuple of (headers, results)
        """
        # Create model
        self.create_model(dropout=0.0)

        # Prepare RoPE tables
        max_len = 8192
        sin, cos = self.prepare_rope_tables(max_len)

        device, _ = self.get_device_dtype()

        # Load checkpoint for tokenizer
        if self.tokenizer is None:
            self.load_checkpoint()

        # Encode prompt
        ids = torch.tensor(
            self.tokenizer.encode(self.args.prompt).ids,
            device=device
        ).unsqueeze(0)

        # Pre-allocate KV cache
        cache = self.create_kv_cache(
            1,
            ids.size(1) + self.args.steps + self.warmup_steps
        )

        # Warmup
        for _ in range(self.warmup_steps):
            logits = self.model(
                ids[:, -1:], sin, cos, cache,
                start_pos=ids.size(1) - 1
            )[:, -1, :]
            ids = torch.cat([
                ids,
                torch.argmax(logits, dim=-1, keepdim=True)
            ], dim=1)

        # Measure with multiple runs for statistics
        def run_decode():
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

        # Get timing statistics
        stats = self.measure_with_stats(
            run_decode,
            n_runs=self.args.n_runs,
            warmup=2
        )

        # Calculate tokens per second
        mean_tps = self.args.steps / stats['mean']
        std_tps = self.args.steps * stats['std'] / (stats['mean'] ** 2)

        print(f"\nDecode Throughput Benchmark:")
        print(f"  Steps: {self.args.steps}")
        print(f"  Tokens/sec: {mean_tps:.2f} ± {std_tps:.2f}")
        print(f"  Latency: {stats['mean']*1000:.2f} ± {stats['std']*1000:.2f} ms")

        # Prepare results
        headers = ('label', 'steps', 'tokens_per_sec', 'std_dev', 'latency_ms')
        results = [(
            self.config.label,
            self.args.steps,
            f'{mean_tps:.2f}',
            f'{std_tps:.2f}',
            f'{stats["mean"]*1000:.2f}'
        )]

        return headers, results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Decode throughput benchmark')

    # Checkpoint and model
    parser.add_argument('--ckpt', required=True, help='Path to checkpoint')
    parser.add_argument('--dtype', default='fp16', choices=['fp16', 'fp32', 'bf16'])
    parser.add_argument('--device', default='cuda', help='Device to use')

    # Benchmark parameters
    parser.add_argument('--steps', type=int, default=256,
                       help='Number of decoding steps')
    parser.add_argument('--n_runs', type=int, default=10,
                       help='Number of runs for statistics')

    # Other options
    parser.add_argument('--prompt', default='Once upon a time',
                       help='Prompt to use')
    parser.add_argument('--label', type=str, help='Device label')
    parser.add_argument('--out', default='out/decode_bench.csv',
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
    runner = DecodeThroughputRunner(config, args)
    headers, results = runner.run()

    # Append to CSV (preserving original behavior)
    runner.append_csv(args.out, results, headers)

    print(f"\nBenchmark complete! Results appended to {args.out}")


if __name__ == "__main__":
    main()