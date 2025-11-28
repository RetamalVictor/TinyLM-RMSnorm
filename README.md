# TinyLM Lab

A minimal language model framework for research, featuring custom CUDA kernels and BitTorch integration.

## Setup

```bash
# Install dependencies
uv sync

# Build CUDA extension (optional, enables faster RMSNorm)
uv run python setup.py build_ext --inplace
```

## Training

```bash
# Prepare data
uv run python data/prepare_tinyshakespeare.py

# Train with defaults (small model, tinyshakespeare)
uv run python train.py

# Train with custom config
uv run python train.py model=medium training=long data=tinystories

# Override specific parameters
uv run python train.py training.steps=5000 model.dim=512
```

Outputs go to `outputs/<date>/<time>/` with:
- `checkpoints/` - model checkpoints
- `tensorboard/` - training metrics
- `.hydra/` - config snapshot

### TensorBoard

```bash
uv run tensorboard --logdir=outputs
```

## Inference

```bash
uv run python infer.py --ckpt outputs/.../checkpoints/best.pt --prompt "Once upon a time"
```

## Configuration

Uses [Hydra](https://hydra.cc/) for configuration. See `conf/` for options:

| Config | Options |
|--------|---------|
| `model` | `small`, `medium`, `large` |
| `training` | `default`, `long` |
| `data` | `tinyshakespeare`, `tinystories` |
| `quant` | `none`, `ternary` (BitTorch) |

## Project Structure

```
tinylm/
├── model/          # Transformer, RoPE, KV-cache, RMSNorm
├── inference/      # Text generation
├── quant/          # Quantization (BitTorch integration)
└── _ext/           # CUDA extensions
kernels/            # CUDA source files
conf/               # Hydra configs
```

## Requirements

- Python 3.10+
- PyTorch 2.2+
- CUDA 12.1+ (for GPU acceleration)

## License

MIT
