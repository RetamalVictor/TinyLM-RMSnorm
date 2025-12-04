#!/usr/bin/env python3
"""Evaluate a trained model and optionally register it.

Usage:
    # Evaluate and show results
    uv run python evaluate.py --ckpt outputs/.../best.pt

    # Evaluate and register if passes
    uv run python evaluate.py --ckpt outputs/.../best.pt --register --name llama-13M-v1

    # List registered models
    uv run python evaluate.py --list

    # Custom thresholds
    uv run python evaluate.py --ckpt outputs/.../best.pt --coherence-threshold 0.6
"""

import argparse
from datetime import datetime
from pathlib import Path

from tinylm.evaluation import FunctionalEvaluator
from tinylm.inference import generate, load_checkpoint
from tinylm.registry import ModelEntry, ModelRegistry


def main():
    parser = argparse.ArgumentParser(description="Evaluate and register models")

    # Actions
    parser.add_argument("--list", action="store_true", help="List registered models")
    parser.add_argument("--remove", type=str, help="Remove model from registry")

    # Evaluation
    parser.add_argument("--ckpt", type=str, help="Path to checkpoint")
    parser.add_argument("--device", default="cuda", help="Device to use")

    # Thresholds
    parser.add_argument("--coherence-threshold", type=float, default=0.5)
    parser.add_argument("--repetition-threshold", type=float, default=0.6)
    parser.add_argument("--completion-threshold", type=float, default=0.5)
    parser.add_argument("--gen-length", type=int, default=100)

    # Registration
    parser.add_argument("--register", action="store_true", help="Register if passes")
    parser.add_argument("--name", type=str, help="Model name for registration")
    parser.add_argument("--tags", type=str, nargs="+", default=[], help="Tags")
    parser.add_argument("--notes", type=str, help="Notes about the model")
    parser.add_argument("--force", action="store_true", help="Register even if fails")

    args = parser.parse_args()
    registry = ModelRegistry()

    # List models
    if args.list:
        models = registry.list()
        if not models:
            print("No models registered.")
        else:
            print(f"\nRegistered models ({len(models)}):\n")
            for m in models:
                status = ""
                if m.benchmarks:
                    passed = m.benchmarks.get("passed", False)
                    status = " [PASSED]" if passed else " [FAILED]"
                tags = f" ({', '.join(m.tags)})" if m.tags else ""
                params = f" - {m.params // 1_000_000}M params" if m.params else ""
                print(f"  {m.name}{status}{params}{tags}")
                print(f"    checkpoint: {m.checkpoint}")
                if m.benchmarks:
                    scores = m.benchmarks
                    print(
                        f"    scores: coherence={scores.get('coherence_score', 'N/A')} "
                        f"repetition={scores.get('repetition_score', 'N/A')} "
                        f"overall={scores.get('overall_score', 'N/A')}"
                    )
                print()
        return

    # Remove model
    if args.remove:
        if registry.remove(args.remove):
            print(f"Removed '{args.remove}' from registry.")
        else:
            print(f"Model '{args.remove}' not found.")
        return

    # Evaluate checkpoint
    if not args.ckpt:
        parser.error("--ckpt required for evaluation")

    print(f"\nLoading checkpoint: {args.ckpt}")
    loaded = load_checkpoint(args.ckpt, device=args.device)
    print(f"Model loaded: {loaded.params:,} parameters")

    # Create generation function using inference module
    def generate_fn(prompt: str, max_tokens: int) -> str:
        return generate(
            loaded.model,
            loaded.tokenizer,
            prompt,
            max_new_tokens=max_tokens,
            temperature=0.8,
            top_p=0.9,
        )

    # Run evaluation
    print("\nRunning functional evaluation...")
    evaluator = FunctionalEvaluator(
        coherence_threshold=args.coherence_threshold,
        repetition_threshold=args.repetition_threshold,
        completion_threshold=args.completion_threshold,
        gen_length=args.gen_length,
    )

    result = evaluator.evaluate(generate_fn, verbose=True)

    # Print summary
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"  Coherence:  {result.coherence_score:.3f} (threshold: {args.coherence_threshold})")
    print(f"  Repetition: {result.repetition_score:.3f} (threshold: {args.repetition_threshold})")
    print(f"  Completion: {result.completion_score:.3f} (threshold: {args.completion_threshold})")
    print(f"  Overall:    {result.overall_score:.3f}")
    print()
    if result.passed:
        print("  STATUS: PASSED")
    else:
        print("  STATUS: FAILED")
        for reason in result.failure_reasons:
            print(f"    - {reason}")
    print("=" * 50)

    # Register if requested
    if args.register or args.force:
        if not result.passed and not args.force:
            print("\nModel did not pass evaluation. Use --force to register anyway.")
            return

        name = args.name
        if not name:
            # Auto-generate name
            arch = loaded.config.get("architecture", "model")
            params_m = loaded.params // 1_000_000
            date = datetime.now().strftime("%Y%m%d")
            name = f"{arch}-{params_m}M-{date}"

        # Check if already exists
        if registry.get(name) and not args.force:
            print(f"\nModel '{name}' already exists. Use --force to overwrite.")
            return

        entry = ModelEntry(
            name=name,
            architecture=loaded.config.get("architecture", "unknown"),
            checkpoint=str(Path(args.ckpt).resolve()),
            created=datetime.now().isoformat(),
            params=loaded.params,
            dim=loaded.config.get("dim"),
            n_layers=loaded.config.get("n_layers"),
            n_heads=loaded.config.get("n_heads"),
            vocab_size=loaded.config.get("vocab_size"),
            steps=loaded.checkpoint.get("step"),
            val_loss=loaded.checkpoint.get("best_val_loss"),
            benchmarks=result.to_dict(),
            tags=args.tags,
            notes=args.notes,
        )

        registry.add(entry, overwrite=args.force)
        print(f"\nRegistered model as '{name}'")


if __name__ == "__main__":
    main()
