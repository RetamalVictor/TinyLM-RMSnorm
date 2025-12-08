#!/usr/bin/env python3
"""Benchmark MHA vs GQA vs MQA with different attention backends.

Measures:
- Peak memory usage
- Inference latency (prefill and generation)
- Throughput (tokens/second)

Usage:
    uv run python benchmarks/bench_attention.py
    uv run python benchmarks/bench_attention.py --device cuda
    uv run python benchmarks/bench_attention.py --seq-len 512 --n-heads 16
"""

import argparse
import gc
import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from tinylm import TinyLM
from tinylm.architectures import ArchitectureConfig


@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark run."""
    name: str
    attention_type: str
    n_kv_heads: Optional[int]
    attention_op: str

    def __str__(self):
        kv_str = f"kv={self.n_kv_heads}" if self.n_kv_heads else "kv=full"
        return f"{self.name} ({kv_str}, {self.attention_op})"


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    config: BenchmarkConfig
    # Memory
    param_memory_mb: float
    peak_memory_mb: float
    cache_memory_mb: float
    # Latency
    prefill_latency_ms: float
    generation_latency_ms: float
    # Throughput
    prefill_tokens_per_sec: float
    generation_tokens_per_sec: float
    # Model info
    total_params: int


def get_memory_mb(device: torch.device) -> float:
    """Get current memory usage in MB."""
    if device.type == "cuda":
        torch.cuda.synchronize()
        return torch.cuda.memory_allocated(device) / 1024 / 1024
    else:
        # For CPU, we can't easily measure memory
        return 0.0


def get_peak_memory_mb(device: torch.device) -> float:
    """Get peak memory usage in MB."""
    if device.type == "cuda":
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated(device) / 1024 / 1024
    else:
        return 0.0


def reset_memory_stats(device: torch.device):
    """Reset memory statistics."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()
    gc.collect()


def create_model(
    config: BenchmarkConfig,
    vocab_size: int,
    dim: int,
    n_layers: int,
    n_heads: int,
    max_seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> TinyLM:
    """Create model with specified configuration."""
    arch_config = ArchitectureConfig(
        name=config.name,
        attention_type=config.attention_type,
        n_kv_heads=config.n_kv_heads,
        attention_op=config.attention_op,
    )

    model = TinyLM(
        vocab_size=vocab_size,
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        max_seq_len=max_seq_len,
        arch_config=arch_config,
    )

    return model.to(device=device, dtype=dtype).eval()


def benchmark_prefill(
    model: TinyLM,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    warmup_runs: int = 3,
    benchmark_runs: int = 10,
) -> tuple[float, float]:
    """Benchmark prefill (processing prompt).

    Returns:
        Tuple of (latency_ms, tokens_per_second)
    """
    x = torch.randint(0, model.vocab_size, (batch_size, seq_len), device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(x)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    latencies = []
    with torch.no_grad():
        for _ in range(benchmark_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()

            _ = model(x)

            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # ms

    avg_latency = sum(latencies) / len(latencies)
    total_tokens = batch_size * seq_len
    tokens_per_sec = total_tokens / (avg_latency / 1000)

    return avg_latency, tokens_per_sec


def benchmark_generation(
    model: TinyLM,
    batch_size: int,
    prompt_len: int,
    gen_len: int,
    device: torch.device,
    warmup_runs: int = 2,
    benchmark_runs: int = 5,
) -> tuple[float, float]:
    """Benchmark autoregressive generation.

    Returns:
        Tuple of (latency_ms_per_token, tokens_per_second)
    """
    prompt = torch.randint(0, model.vocab_size, (batch_size, prompt_len), device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            cache = model.create_kv_cache(batch_size, prompt_len + gen_len)
            _ = model(prompt, cache=cache, start_pos=0)
            for i in range(min(5, gen_len)):
                next_token = torch.randint(0, model.vocab_size, (batch_size, 1), device=device)
                _ = model(next_token, cache=cache, start_pos=prompt_len + i)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Benchmark generation phase only
    latencies = []
    with torch.no_grad():
        for _ in range(benchmark_runs):
            cache = model.create_kv_cache(batch_size, prompt_len + gen_len)

            # Prefill
            _ = model(prompt, cache=cache, start_pos=0)

            if device.type == "cuda":
                torch.cuda.synchronize()

            # Generation
            start = time.perf_counter()
            for i in range(gen_len):
                next_token = torch.randint(0, model.vocab_size, (batch_size, 1), device=device)
                _ = model(next_token, cache=cache, start_pos=prompt_len + i)

            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()

            latencies.append((end - start) * 1000)  # ms

    avg_total_latency = sum(latencies) / len(latencies)
    avg_per_token = avg_total_latency / gen_len
    total_tokens = batch_size * gen_len
    tokens_per_sec = total_tokens / (avg_total_latency / 1000)

    return avg_per_token, tokens_per_sec


def run_benchmark(
    config: BenchmarkConfig,
    vocab_size: int,
    dim: int,
    n_layers: int,
    n_heads: int,
    max_seq_len: int,
    batch_size: int,
    prompt_len: int,
    gen_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> BenchmarkResult:
    """Run full benchmark for a configuration."""
    reset_memory_stats(device)

    # Create model and measure parameter memory
    model = create_model(
        config, vocab_size, dim, n_layers, n_heads, max_seq_len, device, dtype
    )
    param_memory = get_memory_mb(device)
    total_params = model.get_num_params()

    # Create cache and measure cache memory
    cache = model.create_kv_cache(batch_size, max_seq_len)
    cache_memory = get_memory_mb(device) - param_memory

    # Run prefill benchmark
    reset_memory_stats(device)
    model = create_model(
        config, vocab_size, dim, n_layers, n_heads, max_seq_len, device, dtype
    )

    prefill_latency, prefill_tps = benchmark_prefill(
        model, batch_size, prompt_len, device
    )
    prefill_peak = get_peak_memory_mb(device)

    # Run generation benchmark
    reset_memory_stats(device)
    model = create_model(
        config, vocab_size, dim, n_layers, n_heads, max_seq_len, device, dtype
    )

    gen_latency, gen_tps = benchmark_generation(
        model, batch_size, prompt_len, gen_len, device
    )
    gen_peak = get_peak_memory_mb(device)

    peak_memory = max(prefill_peak, gen_peak)

    # Cleanup
    del model
    del cache
    reset_memory_stats(device)

    return BenchmarkResult(
        config=config,
        param_memory_mb=param_memory,
        peak_memory_mb=peak_memory,
        cache_memory_mb=cache_memory,
        prefill_latency_ms=prefill_latency,
        generation_latency_ms=gen_latency,
        prefill_tokens_per_sec=prefill_tps,
        generation_tokens_per_sec=gen_tps,
        total_params=total_params,
    )


def print_results(results: list[BenchmarkResult], device: torch.device):
    """Print benchmark results in a formatted table."""
    print("\n" + "=" * 100)
    print("ATTENTION BENCHMARK RESULTS")
    print("=" * 100)

    # Header
    print(f"\n{'Configuration':<35} {'Params':>10} {'Peak Mem':>10} {'Cache':>10} "
          f"{'Prefill':>12} {'Gen/tok':>12} {'Gen TPS':>12}")
    print(f"{'':35} {'(M)':>10} {'(MB)':>10} {'(MB)':>10} "
          f"{'(ms)':>12} {'(ms)':>12} {'(tok/s)':>12}")
    print("-" * 100)

    for r in results:
        params_m = r.total_params / 1e6
        print(f"{str(r.config):<35} {params_m:>10.2f} {r.peak_memory_mb:>10.1f} "
              f"{r.cache_memory_mb:>10.1f} {r.prefill_latency_ms:>12.2f} "
              f"{r.generation_latency_ms:>12.2f} {r.generation_tokens_per_sec:>12.1f}")

    print("-" * 100)

    # Analysis
    print("\n" + "=" * 100)
    print("ANALYSIS")
    print("=" * 100)

    # Group by attention type
    mha_results = [r for r in results if r.config.attention_type == "mha"]
    gqa_results = [r for r in results if r.config.attention_type == "gqa"]
    mqa_results = [r for r in results if r.config.attention_type == "mqa"]

    if mha_results:
        mha_baseline = mha_results[0]

        print(f"\n{'Relative to MHA (standard):':<40}")
        print("-" * 60)

        for r in results:
            if r == mha_baseline:
                continue

            mem_ratio = r.peak_memory_mb / mha_baseline.peak_memory_mb if mha_baseline.peak_memory_mb > 0 else 1.0
            cache_ratio = r.cache_memory_mb / mha_baseline.cache_memory_mb if mha_baseline.cache_memory_mb > 0 else 1.0
            speed_ratio = r.generation_tokens_per_sec / mha_baseline.generation_tokens_per_sec if mha_baseline.generation_tokens_per_sec > 0 else 1.0
            param_ratio = r.total_params / mha_baseline.total_params

            param_diff = (1 - param_ratio) * 100
            mem_diff = (1 - mem_ratio) * 100
            cache_diff = (1 - cache_ratio) * 100
            speed_diff = (speed_ratio - 1) * 100

            print(f"  {str(r.config):<35}")
            print(f"    Params:     {param_diff:+.1f}% {'(fewer)' if param_diff > 0 else '(more)'}")
            if device.type == "cuda":
                print(f"    Peak mem:   {mem_diff:+.1f}% {'(less)' if mem_diff > 0 else '(more)'}")
                print(f"    Cache mem:  {cache_diff:+.1f}% {'(less)' if cache_diff > 0 else '(more)'}")
            print(f"    Gen speed:  {speed_diff:+.1f}% {'(faster)' if speed_diff > 0 else '(slower)'}")
            print()


def main():
    parser = argparse.ArgumentParser(description="Benchmark MHA/GQA/MQA attention")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run on (cuda/cpu)")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32", "bfloat16"],
                        help="Data type")
    parser.add_argument("--vocab-size", type=int, default=32000, help="Vocabulary size")
    parser.add_argument("--dim", type=int, default=512, help="Model dimension")
    parser.add_argument("--n-layers", type=int, default=6, help="Number of layers")
    parser.add_argument("--n-heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--max-seq-len", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--prompt-len", type=int, default=128, help="Prompt length for prefill")
    parser.add_argument("--gen-len", type=int, default=64, help="Generation length")
    parser.add_argument("--gqa-kv-heads", type=int, default=None, help="KV heads for GQA (default: n_heads // 4)")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype_map = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}
    dtype = dtype_map[args.dtype]

    # Adjust dtype for CPU
    if device.type == "cpu" and dtype == torch.float16:
        print("Note: Using float32 on CPU (float16 not well supported)")
        dtype = torch.float32

    gqa_kv_heads = args.gqa_kv_heads or max(1, args.n_heads // 4)

    print("=" * 100)
    print("BENCHMARK CONFIGURATION")
    print("=" * 100)
    print(f"Device:        {device}")
    print(f"Dtype:         {dtype}")
    print(f"Model:         dim={args.dim}, layers={args.n_layers}, heads={args.n_heads}")
    print(f"Vocab:         {args.vocab_size}")
    print(f"Sequence:      max={args.max_seq_len}, prompt={args.prompt_len}, gen={args.gen_len}")
    print(f"Batch size:    {args.batch_size}")
    print(f"GQA KV heads:  {gqa_kv_heads}")

    if device.type == "cuda":
        print(f"GPU:           {torch.cuda.get_device_name(device)}")
        print(f"GPU Memory:    {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB")

    # Define configurations to benchmark
    configs = [
        # MHA
        BenchmarkConfig("MHA", "mha", None, "standard"),
        BenchmarkConfig("MHA+Flash", "mha", None, "flash"),
        # GQA
        BenchmarkConfig("GQA", "gqa", gqa_kv_heads, "standard"),
        BenchmarkConfig("GQA+Flash", "gqa", gqa_kv_heads, "flash"),
        # MQA
        BenchmarkConfig("MQA", "mqa", 1, "standard"),
        BenchmarkConfig("MQA+Flash", "mqa", 1, "flash"),
    ]

    # Run benchmarks
    results = []
    for i, config in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] Benchmarking {config}...")
        try:
            result = run_benchmark(
                config=config,
                vocab_size=args.vocab_size,
                dim=args.dim,
                n_layers=args.n_layers,
                n_heads=args.n_heads,
                max_seq_len=args.max_seq_len,
                batch_size=args.batch_size,
                prompt_len=args.prompt_len,
                gen_len=args.gen_len,
                device=device,
                dtype=dtype,
            )
            results.append(result)
            print(f"  ✓ Done: {result.generation_tokens_per_sec:.1f} tok/s")
        except Exception as e:
            print(f"  ✗ Failed: {e}")

    # Print results
    print_results(results, device)


if __name__ == "__main__":
    main()
