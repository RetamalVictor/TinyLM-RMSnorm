# TinyLM Lab

Minimal language model framework for research.

## Setup

```bash
uv sync
uv run python setup.py build_ext --inplace  # Optional: CUDA RMSNorm
```

## Training

```bash
# Prepare data
uv run python scripts/prepare_tinyshakespeare.py
uv run python scripts/prepare_tinystories.py

# Train from scratch
uv run python train.py
uv run python train.py data=tinystories model=medium training=long
uv run python train.py training.steps=50000 training.lr=1e-4

# Fine-tune from checkpoint
uv run python train.py finetune=full finetune.checkpoint_path=outputs/.../best.pt data=tinystories
uv run python train.py finetune=freeze_early finetune.checkpoint_path=outputs/.../best.pt
uv run python train.py finetune=freeze_embeddings finetune.checkpoint_path=outputs/.../best.pt

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
| `model` | `small`, `medium`, `large` |
| `data` | `tinyshakespeare`, `tinystories`, `combined` |
| `training` | `default`, `long` |
| `tokenizer` | `bytelevel` (default), `whitespace` |
| `finetune` | `full`, `freeze_early`, `freeze_embeddings` |
| `quant` | `none`, `ternary` |

Outputs: `outputs/<date>/<time>/` (checkpoints, tensorboard, config)

## License

MIT
