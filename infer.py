"""TinyLM inference script."""

import argparse
import os

import torch
from tokenizers import Tokenizer

from tinylm import TinyLM, generate
from tinylm.quant import QuantConfig
from tinylm.kernels import set_backend, available_backends


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=str, required=True)
    ap.add_argument('--prompt', type=str, default='Once upon a time')
    ap.add_argument('--max_new_tokens', type=int, default=128)
    ap.add_argument('--temperature', type=float, default=0.9, help='Sampling temperature (0=greedy, >0=sampling)')
    ap.add_argument('--top_p', type=float, default=0.9)
    ap.add_argument('--repetition_penalty', type=float, default=1.1)
    ap.add_argument('--freq_penalty', type=float, default=0.0)
    ap.add_argument('--presence_penalty', type=float, default=0.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--stream', action='store_true')
    ap.add_argument('--kernel-backend', type=str, default='auto',
                    choices=['auto', 'cuda', 'triton', 'pytorch'],
                    help='Kernel backend to use (default: auto)')
    args = ap.parse_args()

    # Setup kernel backend
    set_backend(args.kernel_backend)
    print(f"Kernel backend: {args.kernel_backend} (available: {available_backends()})")

    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    ckpt = torch.load(args.ckpt, map_location='cpu')

    # Load tokenizer
    if 'tokenizer' in ckpt:
        tok = Tokenizer.from_str(ckpt['tokenizer'])
    elif 'tok' in ckpt:
        tok = Tokenizer.from_str(ckpt['tok'])
    else:
        raise ValueError("Checkpoint missing tokenizer.")

    # Load config
    cfg = ckpt.get('config', {})
    if 'model' in cfg:
        model_cfg = cfg['model']
    else:
        model_cfg = {'dim': 384, 'n_layers': 6, 'n_heads': 6}

    # Load quantization config if present
    quant_config = None
    if 'quant_config' in ckpt and ckpt['quant_config'] is not None:
        quant_config = QuantConfig.from_dict(ckpt['quant_config'])
        print(f"Loading quantized model: {quant_config}")

    model = TinyLM(
        vocab_size=tok.get_vocab_size(),
        dim=model_cfg.get('dim', 384),
        n_layers=model_cfg.get('n_layers', 6),
        n_heads=model_cfg.get('n_heads', 6),
        quant_config=quant_config
    ).cuda().eval()

    state = ckpt['model']
    if any(k.startswith('_orig_mod.') for k in state):
        state = {k.replace('_orig_mod.', '', 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)

    txt = generate(
        model, tok, args.prompt,
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
