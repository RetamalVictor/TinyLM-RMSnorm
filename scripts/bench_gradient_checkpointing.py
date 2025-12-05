#!/usr/bin/env python
"""
Benchmark gradient checkpointing: memory, speed, and accuracy comparison.

This script measures:
1. Peak memory usage during training forward+backward
2. Training step time (forward + backward + optimizer)
3. Loss convergence over N steps

Usage:
    uv run python scripts/bench_gradient_checkpointing.py [--device cuda|cpu]
    uv run python scripts/bench_gradient_checkpointing.py --steps 100 --batch-size 8
"""

import argparse
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW

from tinylm import TinyLM


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    checkpointing: bool
    peak_memory_mb: float
    avg_step_time_ms: float
    final_loss: float
    losses: List[float]


def measure_memory_and_time(
    model: TinyLM,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    steps: int,
    device: torch.device,
    warmup_steps: int = 3,
) -> Tuple[float, float, float, List[float]]:
    """Measure peak memory and average step time.

    Args:
        model: TinyLM model instance
        batch_size: Batch size for training
        seq_len: Sequence length
        vocab_size: Vocabulary size
        steps: Number of training steps
        device: Device to run on
        warmup_steps: Number of warmup steps

    Returns:
        Tuple of (peak_memory_mb, avg_step_time_ms, final_loss, losses)
    """
    model.train()
    optimizer = AdamW(model.parameters(), lr=1e-4)

    # Track memory
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    losses = []
    step_times = []

    for step in range(steps + warmup_steps):
        # Generate random data
        x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        y = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

        if device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        # Forward pass
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = nn.functional.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

        # Backward pass
        loss.backward()

        # Optimizer step
        optimizer.step()

        if device.type == "cuda":
            torch.cuda.synchronize()
        end_time = time.perf_counter()

        # Record after warmup
        if step >= warmup_steps:
            losses.append(loss.item())
            step_times.append((end_time - start_time) * 1000)

    # Get peak memory
    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        peak_memory_mb = 0.0  # Can't easily measure CPU memory

    avg_step_time_ms = sum(step_times) / len(step_times)
    final_loss = losses[-1]

    return peak_memory_mb, avg_step_time_ms, final_loss, losses


def run_benchmark(
    config: Dict,
    batch_size: int,
    seq_len: int,
    steps: int,
    device: torch.device,
    gradient_checkpointing: bool,
) -> BenchmarkResult:
    """Run benchmark for a specific configuration.

    Args:
        config: Model configuration
        batch_size: Batch size
        seq_len: Sequence length
        steps: Number of training steps
        device: Device to run on
        gradient_checkpointing: Whether to enable checkpointing

    Returns:
        BenchmarkResult with measurements
    """
    torch.manual_seed(42)

    model = TinyLM(
        **config,
        gradient_checkpointing=gradient_checkpointing,
    ).to(device)

    peak_mem, avg_time, final_loss, losses = measure_memory_and_time(
        model=model,
        batch_size=batch_size,
        seq_len=seq_len,
        vocab_size=config["vocab_size"],
        steps=steps,
        device=device,
    )

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return BenchmarkResult(
        checkpointing=gradient_checkpointing,
        peak_memory_mb=peak_mem,
        avg_step_time_ms=avg_time,
        final_loss=final_loss,
        losses=losses,
    )


def print_comparison(result_no_ckpt: BenchmarkResult, result_ckpt: BenchmarkResult):
    """Print comparison of benchmark results."""
    print("\n" + "=" * 70)
    print("GRADIENT CHECKPOINTING BENCHMARK RESULTS")
    print("=" * 70)

    print(f"\n{'Metric':<25} {'No Checkpointing':>20} {'With Checkpointing':>20}")
    print("-" * 70)

    # Memory comparison
    mem_diff = result_no_ckpt.peak_memory_mb - result_ckpt.peak_memory_mb
    mem_pct = (mem_diff / result_no_ckpt.peak_memory_mb * 100) if result_no_ckpt.peak_memory_mb > 0 else 0
    print(f"{'Peak Memory (MB)':<25} {result_no_ckpt.peak_memory_mb:>20.1f} {result_ckpt.peak_memory_mb:>20.1f}")
    if result_no_ckpt.peak_memory_mb > 0:
        print(f"{'Memory Savings':<25} {'':<20} {f'-{mem_pct:.1f}%':>20}")

    # Time comparison
    time_diff = result_ckpt.avg_step_time_ms - result_no_ckpt.avg_step_time_ms
    time_pct = (time_diff / result_no_ckpt.avg_step_time_ms * 100)
    print(f"{'Avg Step Time (ms)':<25} {result_no_ckpt.avg_step_time_ms:>20.2f} {result_ckpt.avg_step_time_ms:>20.2f}")
    print(f"{'Time Overhead':<25} {'':<20} {f'+{time_pct:.1f}%':>20}")

    # Loss comparison
    print(f"{'Final Loss':<25} {result_no_ckpt.final_loss:>20.4f} {result_ckpt.final_loss:>20.4f}")

    # Check if losses diverged significantly
    loss_diff = abs(result_no_ckpt.final_loss - result_ckpt.final_loss)
    loss_ratio = loss_diff / result_no_ckpt.final_loss * 100
    if loss_ratio < 1:
        print(f"{'Loss Difference':<25} {'':>20} {f'{loss_ratio:.2f}% (OK)':>20}")
    else:
        print(f"{'Loss Difference':<25} {'':>20} {f'{loss_ratio:.2f}% (WARNING)':>20}")

    print("=" * 70)

    # Summary
    print("\nSUMMARY:")
    if result_ckpt.peak_memory_mb > 0:
        print(f"  - Memory reduction: {mem_pct:.1f}%")
    print(f"  - Time overhead: {time_pct:.1f}%")
    print(f"  - Loss difference: {loss_ratio:.2f}%")

    if mem_pct > 10 and loss_ratio < 1:
        print("\n  Gradient checkpointing is RECOMMENDED for this configuration.")
    elif mem_pct > 5:
        print("\n  Gradient checkpointing provides modest memory savings.")
    else:
        print("\n  Gradient checkpointing may not be beneficial for this configuration.")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark gradient checkpointing memory and performance"
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (cuda or cpu)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Batch size for training"
    )
    parser.add_argument(
        "--seq-len", type=int, default=128,
        help="Sequence length"
    )
    parser.add_argument(
        "--steps", type=int, default=20,
        help="Number of training steps"
    )
    parser.add_argument(
        "--dim", type=int, default=256,
        help="Model dimension"
    )
    parser.add_argument(
        "--n-layers", type=int, default=8,
        help="Number of transformer layers"
    )
    parser.add_argument(
        "--n-heads", type=int, default=8,
        help="Number of attention heads"
    )
    args = parser.parse_args()

    device = torch.device(args.device)

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    config = {
        "vocab_size": 10000,
        "dim": args.dim,
        "n_layers": args.n_layers,
        "n_heads": args.n_heads,
        "max_seq_len": 512,
        "dropout": 0.0,
    }

    print(f"\nModel config: dim={config['dim']}, layers={config['n_layers']}, heads={config['n_heads']}")
    print(f"Training: batch_size={args.batch_size}, seq_len={args.seq_len}, steps={args.steps}")

    # Run benchmark without checkpointing
    print("\nRunning benchmark WITHOUT gradient checkpointing...")
    result_no_ckpt = run_benchmark(
        config=config,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        steps=args.steps,
        device=device,
        gradient_checkpointing=False,
    )

    # Run benchmark with checkpointing
    print("Running benchmark WITH gradient checkpointing...")
    result_ckpt = run_benchmark(
        config=config,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        steps=args.steps,
        device=device,
        gradient_checkpointing=True,
    )

    # Print comparison
    print_comparison(result_no_ckpt, result_ckpt)


if __name__ == "__main__":
    main()
