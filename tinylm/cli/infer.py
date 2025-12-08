"""TinyLM inference script.

Usage:
    # Load by checkpoint path
    uv run python infer.py --ckpt outputs/.../best.pt --prompt "Once upon"

    # Load by model name from registry
    uv run python infer.py --model llama-13M-v1 --prompt "Once upon"

    # Load by tag (uses first match)
    uv run python infer.py --tag baseline --prompt "Once upon"

    # List available models
    uv run python infer.py --list
"""

import argparse

from tinylm.inference import (
    generate,
    list_models,
    load_checkpoint,
    load_from_registry,
)
from tinylm.kernels import available_backends, set_backend
from tinylm.registry import ModelRegistry


def main():
    ap = argparse.ArgumentParser(description="TinyLM text generation")

    # Model selection (mutually exclusive)
    model_group = ap.add_mutually_exclusive_group()
    model_group.add_argument('--ckpt', type=str, help='Path to checkpoint file')
    model_group.add_argument('--model', type=str, help='Model name from registry')
    model_group.add_argument('--tag', type=str, help='Load first model with this tag')
    model_group.add_argument('--list', action='store_true', help='List registered models')

    # Generation parameters
    ap.add_argument('--prompt', type=str, default='Once upon a time')
    ap.add_argument('--max_new_tokens', type=int, default=128)
    ap.add_argument('--temperature', type=float, default=0.9)
    ap.add_argument('--top_p', type=float, default=0.9)
    ap.add_argument('--repetition_penalty', type=float, default=1.1)
    ap.add_argument('--freq_penalty', type=float, default=0.0)
    ap.add_argument('--presence_penalty', type=float, default=0.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--stream', action='store_true')

    # Device and backend
    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--kernel-backend', type=str, default='auto',
                    choices=['auto', 'cuda', 'triton', 'pytorch'])

    args = ap.parse_args()

    # List models
    if args.list:
        registry = ModelRegistry()
        models = registry.list()
        if not models:
            print("No models registered.")
        else:
            print(f"\nRegistered models ({len(models)}):\n")
            for m in models:
                tags = f" [{', '.join(m.tags)}]" if m.tags else ""
                params = f"{m.params // 1_000_000}M" if m.params else "?"
                print(f"  {m.name} ({params} params){tags}")
        return

    # Require a model source
    if not args.ckpt and not args.model and not args.tag:
        ap.error("Must specify --ckpt, --model, or --tag (or use --list)")

    # Setup kernel backend
    set_backend(args.kernel_backend)
    print(f"Kernel backend: {args.kernel_backend} (available: {available_backends()})")

    # Load model
    if args.ckpt:
        loaded = load_checkpoint(args.ckpt, device=args.device)
        print(f"Loaded from checkpoint: {loaded.params:,} parameters")
    else:
        loaded = load_from_registry(
            name=args.model,
            tag=args.tag,
            device=args.device
        )
        source = args.model or f"tag:{args.tag}"
        print(f"Loaded '{source}' from registry: {loaded.params:,} parameters")

    # Generate
    txt = generate(
        loaded.model,
        loaded.tokenizer,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        freq_penalty=args.freq_penalty,
        presence_penalty=args.presence_penalty,
        seed=args.seed,
        stream=args.stream
    )
    print(txt)


if __name__ == "__main__":
    main()
