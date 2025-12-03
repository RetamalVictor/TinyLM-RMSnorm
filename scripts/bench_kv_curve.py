"""
Benchmark KV-cache performance across different context lengths.

This refactored version uses the benchmark base class to eliminate duplication.
"""

import argparse
from typing import List, Tuple

import torch
from benchmark_base import BenchmarkConfig, KVCacheBenchmark


class KVCurveRunner(KVCacheBenchmark):
    """Runner for KV-cache curve benchmarks."""

    def __init__(self, config: BenchmarkConfig, args):
        """Initialize with config and additional arguments."""
        super().__init__(config)
        self.args = args
        self.warmup = 10

    def make_ids(self, length: int) -> torch.Tensor:
        """Create input token IDs.

        Args:
            length: Sequence length

        Returns:
            Token ID tensor of shape [1, length]
        """
        if self.tokenizer is None:
            self.load_checkpoint()

        device, _ = self.get_device_dtype()

        # Encode prompt
        base_ids = self.tokenizer.encode(self.args.prompt).ids

        if len(base_ids) >= length:
            ids = base_ids[:length]
        else:
            # Pad with random tokens
            vocab_size = self.tokenizer.get_vocab_size()
            extra = torch.randint(0, vocab_size, (length - len(base_ids),)).tolist()
            ids = base_ids + extra

        return torch.tensor(ids, device=device).unsqueeze(0)

    def measure_with_kv(
        self,
        ids: torch.Tensor,
        steps: int,
        sin: torch.Tensor,
        cos: torch.Tensor
    ) -> Tuple[float, float]:
        """Measure throughput with KV-cache.

        Returns:
            Tuple of (mean tokens/sec, std deviation)
        """
        # Pre-allocate cache
        cache = self.create_kv_cache(1, ids.size(1) + self.warmup + steps)

        # Prefill cache
        _ = self.model(ids, sin, cos, cache, start_pos=0)

        # Warmup incremental decoding
        for _ in range(self.warmup):
            logits = self.model(ids[:, -1:], sin, cos, cache, start_pos=ids.size(1)-1)[:, -1, :]
            ids = torch.cat([ids, torch.argmax(logits, dim=-1, keepdim=True)], dim=1)

        # Measure with multiple runs for statistics
        def run_inference():
            nonlocal ids
            temp_ids = ids.clone()
            for _ in range(steps):
                logits = self.model(
                    temp_ids[:, -1:], sin, cos, cache,
                    start_pos=temp_ids.size(1)-1
                )[:, -1, :]
                temp_ids = torch.cat([temp_ids, torch.argmax(logits, dim=-1, keepdim=True)], dim=1)

        stats = self.measure_with_stats(run_inference, n_runs=self.args.n_runs, warmup=2)

        # Calculate tokens per second
        mean_tps = steps / stats['mean']
        std_tps = steps * stats['std'] / (stats['mean'] ** 2)  # Error propagation

        return mean_tps, std_tps

    def measure_no_kv(
        self,
        ids: torch.Tensor,
        steps: int,
        sin: torch.Tensor,
        cos: torch.Tensor
    ) -> Tuple[float, float]:
        """Measure throughput without KV-cache.

        Returns:
            Tuple of (mean tokens/sec, std deviation)
        """
        # Warmup
        tmp = ids.clone()
        for _ in range(3):
            logits = self.model(tmp, sin, cos, cache=None, start_pos=0)[:, -1, :]
            tmp = torch.cat([tmp, torch.argmax(logits, dim=-1, keepdim=True)], dim=1)

        # Measure with multiple runs
        def run_inference():
            temp_ids = ids.clone()
            for _ in range(steps):
                logits = self.model(temp_ids, sin, cos, cache=None, start_pos=0)[:, -1, :]
                temp_ids = torch.cat([temp_ids, torch.argmax(logits, dim=-1, keepdim=True)], dim=1)

        stats = self.measure_with_stats(run_inference, n_runs=self.args.n_runs, warmup=2)

        # Calculate tokens per second
        mean_tps = steps / stats['mean']
        std_tps = steps * stats['std'] / (stats['mean'] ** 2)

        return mean_tps, std_tps

    def run(self) -> List[Tuple]:
        """Run the benchmark across all context lengths.

        Returns:
            List of result tuples
        """
        # Create model
        self.create_model(dropout=0.0)

        # Prepare RoPE tables
        max_len = max(self.args.lengths) + self.args.steps + self.warmup + 8
        sin, cos = self.prepare_rope_tables(max_len)

        # Results storage
        results = []
        headers = ('label', 'dtype', 'context_len', 'mode', 'tokens_per_sec', 'std_dev')

        for length in self.args.lengths:
            try:
                print(f"\nContext length: {length}")

                # Create input
                ids = self.make_ids(length)

                # Measure with KV-cache
                kv_mean, kv_std = self.measure_with_kv(
                    ids.clone(), self.args.steps, sin, cos
                )

                # Measure without KV-cache
                nokv_mean, nokv_std = self.measure_no_kv(
                    ids.clone(), self.args.steps, sin, cos
                )

                # Calculate speedup
                speedup = kv_mean / max(nokv_mean, 1e-9)

                print(f"  With KV:    {kv_mean:7.1f} ± {kv_std:5.1f} tok/s")
                print(f"  Without KV: {nokv_mean:7.1f} ± {nokv_std:5.1f} tok/s")
                print(f"  Speedup:    {speedup:7.2f}x")

                # Store results
                results.append((
                    self.config.label,
                    self.config.dtype,
                    length,
                    'with_kv',
                    f'{kv_mean:.3f}',
                    f'{kv_std:.3f}'
                ))
                results.append((
                    self.config.label,
                    self.config.dtype,
                    length,
                    'no_kv',
                    f'{nokv_mean:.3f}',
                    f'{nokv_std:.3f}'
                ))

            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    print("  OOM - skipping")
                    torch.cuda.empty_cache()
                else:
                    raise

        return headers, results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='KV-cache performance benchmark')

    # Checkpoint and model
    parser.add_argument('--ckpt', required=True, help='Path to checkpoint')
    parser.add_argument('--dtype', default='fp16', choices=['fp16', 'fp32', 'bf16'])
    parser.add_argument('--device', default='cuda', help='Device to use')

    # Benchmark parameters
    parser.add_argument('--lengths', type=int, nargs='+',
                       default=[32, 64, 128, 192, 256],
                       help='Context lengths to test')
    parser.add_argument('--steps', type=int, default=128,
                       help='Number of generation steps')
    parser.add_argument('--n_runs', type=int, default=5,
                       help='Number of runs for statistics')

    # Other options
    parser.add_argument('--prompt', default='Once upon a time',
                       help='Prompt to use')
    parser.add_argument('--label', type=str, help='Device label')
    parser.add_argument('--out', default='out/kv_curve_stats.csv',
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
    runner = KVCurveRunner(config, args)
    headers, results = runner.run()

    # Write results
    runner.write_csv(args.out, results, headers)

    print(f"\nBenchmark complete! Results saved to {args.out}")


if __name__ == "__main__":
    main()
