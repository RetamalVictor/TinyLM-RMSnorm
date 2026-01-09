# TinyLM Lab

A minimal transformer framework for research prototyping. ~6,800 lines of Python.

## Features

- **Architectures**: LLaMA (RMSNorm, RoPE, SwiGLU) and GPT (LayerNorm, learned pos, GELU)
- **Attention**: MHA, GQA, MQA with KV cache
- **Quantization**: Ternary weights via BitTorch
- **Export**: Browser deployment (SafeTensors + WebGPU)

## Setup

```bash
uv sync
uv run python setup.py build_ext --inplace  # Optional: CUDA RMSNorm
```

## Quick Start

```python
from tinylm import TinyLM

model = TinyLM(
    vocab_size=32000,
    dim=512,
    n_layers=8,
    n_heads=8,
    architecture="llama",
)

# Forward pass
logits = model(tokens)

# Generation with KV cache
cache = model.create_kv_cache(batch_size=1, max_seq_len=512)
logits = model(tokens, cache=cache, start_pos=0)
```

## Training

```bash
# Prepare data
uv run python scripts/prepare_tinystories.py

# Train
uv run python -m tinylm.cli.train model=small
uv run python -m tinylm.cli.train model=small model.architecture=gpt

# With ternary quantization
uv run python -m tinylm.cli.train model=small quant=ternary
```

## Inference

```bash
uv run python -m tinylm.cli.infer --ckpt outputs/.../best.pt --prompt "Once upon a time"
```

```python
from tinylm.inference import load_checkpoint, generate

loaded = load_checkpoint("outputs/.../best.pt")
text = generate(loaded.model, loaded.tokenizer, "The robot said", max_new_tokens=50)
print(text)
```

## Configuration

| Option | Values |
|--------|--------|
| `model` | `tiny`, `small`, `medium`, `large` |
| `model.architecture` | `llama`, `gpt` |
| `data` | `tinyshakespeare`, `tinystories`, `wikitext` |
| `quant` | `none`, `ternary` |

## Tests

```bash
uv run pytest tests/ -v
```

## License

MIT
