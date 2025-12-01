# TinyLM Lab

Multi-architecture language model framework for research. Build tiny versions of LLaMA, GPT, and other architectures.

## Features

- **Multi-Architecture**: LLaMA (RMSNorm, RoPE, SwiGLU) and GPT (LayerNorm, learned pos, GELU)
- **Component System**: Modular building blocks with registry pattern
- **Clean API**: Model handles positional embeddings internally
- **KV Cache**: Efficient autoregressive generation
- **Hydra Config**: Flexible configuration management

## Setup

```bash
uv sync
uv run python setup.py build_ext --inplace  # Optional: CUDA RMSNorm
```

## Architectures

| Feature | LLaMA | GPT |
|---------|-------|-----|
| Normalization | RMSNorm (pre) | LayerNorm (post) |
| Positional | RoPE | Learned |
| MLP | Gated (SwiGLU) | Standard |
| Activation | SiLU | GELU |
| Bias | No | Yes |

```python
from tinylm import TinyLM

# LLaMA-style model
model = TinyLM(vocab_size=32000, dim=512, n_layers=8, n_heads=8, architecture="llama")

# GPT-style model
model = TinyLM(vocab_size=32000, dim=512, n_layers=8, n_heads=8, architecture="gpt")

# Forward pass (clean API - no sin/cos needed)
logits = model(tokens)

# Generation with KV cache
cache = model.create_kv_cache(batch_size=1, max_seq_len=512)
logits = model(tokens, cache=cache, start_pos=0)
```

## Training

```bash
# Prepare data
uv run python scripts/prepare_tinyshakespeare.py
uv run python scripts/prepare_tinystories.py

# Train LLaMA-style (default)
uv run python train.py model=small

# Train GPT-style
uv run python train.py model=small model.architecture=gpt

# Train with options
uv run python train.py model=medium model.architecture=llama data=tinystories training=long

# Fine-tune from checkpoint
uv run python train.py finetune=full finetune.checkpoint_path=outputs/.../best.pt

# Resume interrupted training
uv run python train.py resume.enabled=true resume.checkpoint_path=outputs/.../best.pt
```

## Inference

```bash
uv run python infer.py --ckpt outputs/.../best.pt --prompt "Once upon a time"
uv run python infer.py --ckpt outputs/.../best.pt --prompt "The king" --temperature 0.8
```

## Monitor

```bash
uv run tensorboard --logdir=outputs
```

## Configuration

| Group | Options |
|-------|---------|
| `model` | `tiny`, `small`, `medium`, `large` |
| `model.architecture` | `llama`, `gpt` |
| `data` | `tinyshakespeare`, `tinystories`, `combined` |
| `training` | `default`, `long`, `quick_test` |
| `tokenizer` | `bytelevel` (default), `whitespace` |
| `finetune` | `full`, `freeze_early`, `freeze_embeddings` |
| `quant` | `none`, `ternary` |

## Project Structure

```
tinylm/
├── architectures/     # Architecture configs (llama, gpt)
├── components/        # Building blocks
│   ├── normalization/ # RMSNorm (with CUDA kernel), LayerNorm
│   ├── positional/    # RoPE, Learned
│   ├── attention/     # MHA
│   ├── mlp/           # Standard, Gated (SwiGLU)
│   └── activations.py # SiLU, GELU, ReLU
├── model/
│   ├── transformer.py # TinyLM main class
│   └── blocks.py      # PreNorm, PostNorm blocks
├── inference/         # Generation utilities, KV cache
├── training/          # Training utilities
└── quant/             # Ternary quantization (BitTorch)
```

## Tests

```bash
uv run pytest tests/test_architectures.py -v
```

## License

MIT
