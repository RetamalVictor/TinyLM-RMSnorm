# TinyLM-Lab

A tiny, hackable, highly engineered LM research testbed.

## Project Identity

TinyLM-Lab is a **minimal but complete transformer framework** designed for:
- Experimenting with exotic LM inference ideas
- Building tiny-scale proofs of concept for research
- Testing optimizations (CUDA, Triton, quantization, routing, caching)
- Learning and teaching modern LM architecture

**Core Philosophy:**
1. Small enough to understand fully
2. Engineered enough to feel real
3. Modular enough to test wild ideas

## Architecture Overview

```
                           ┌──────────────────────────┐
                           │        TinyLM-Lab        │
                           │   (Research Testbed)     │
                           └─────────────┬────────────┘
                                         │
                ┌────────────────────────┼────────────────────────┐
                │                        │                        │
        ┌───────▼────────┐      ┌────────▼────────┐      ┌────────▼────────┐
        │   Core Model    │      │  Inference Core │      │  Training Core  │
        │ (Transformer)   │      │ (Modular Engine)│      │   (Trainer)     │
        └───────┬────────┘      └────────┬─────────┘      └───────┬────────┘
                │                         │                        │
     ┌──────────▼─────────┐    ┌──────────▼─────────┐    ┌─────────▼─────────┐
     │   AttentionEngine   │    │   CacheManager     │    │ Distributed (DDP/ │
     │  (plugin attention) │    │ (KV, blocks, C2C)  │    │ FSDP, Grad Accum) │
     └─────────┬──────────┘    └──────────┬─────────┘    └─────────┬─────────┘
               │                          │                        │
   ┌───────────▼──────────┐    ┌──────────▼──────────┐    ┌────────▼─────────┐
   │ Kernels Backend       │    │ RoutingEngine       │    │ Optimizers, LR   │
   │ (CUDA, Triton, PyTorch)│    │ (token/head routing)│    │ Schedulers       │
   └───────────┬──────────┘    └──────────┬──────────┘    └────────┬─────────┘
               │                          │                        │
   ┌───────────▼──────────┐    ┌──────────▼──────────┐    ┌────────▼──────────┐
   │  Quantization Layer   │    │ Experiments Module  │    │ Dataset + Tokenizer│
   │ (INT8/INT4/QAT)       │    │ (research sandbox)  │    │  (text, images)    │
   └───────────┬──────────┘    └──────────┬──────────┘    └────────┬──────────┘
               │                          │                        │
   ┌───────────▼──────────┐    ┌──────────▼───────────┐   ┌────────▼───────────┐
   │ Benchmarks & Profiling│    │ Multi-Modality       │   │ Pretrained Models  │
   │ (latency, mem, etc.)  │    │ (image/audio→tokens) │   │ (tiny-scale demos) │
   └───────────────────────┘    └──────────────────────┘   └────────────────────┘
```

## Directory Structure

```
tinylm/
├── architectures/       # Architecture configs (LLaMA, GPT)
│   ├── __init__.py      # get_architecture(), register_architecture()
│   └── config.py        # ArchitectureConfig dataclass
├── components/          # Modular building blocks
│   ├── registry.py      # ComponentRegistry (factory pattern)
│   ├── activations.py   # SiLU, GELU, etc.
│   ├── attention/       # MHA implementations
│   ├── mlp/             # Standard, Gated (SwiGLU)
│   ├── normalization/   # RMSNorm, LayerNorm
│   └── positional/      # RoPE, Learned
├── model/
│   ├── transformer.py   # TinyLM main class
│   └── blocks.py        # PreNormBlock, PostNormBlock
├── inference/
│   ├── generate.py      # Text generation
│   └── cache_manager.py # CacheManager abstraction (StandardCache, etc.)
├── training/            # Training utilities
├── quant/               # Quantization (stub)
└── _ext/                # CUDA extensions
    └── rmsnorm_cuda.so  # Custom RMSNorm kernel

kernels/
├── rmsnorm_cuda.cu      # CUDA RMSNorm (forward + backward)
└── rmsnorm_binding.cpp  # PyTorch bindings

conf/                    # Hydra configs
├── model/               # tiny, small, medium, large
├── training/            # default, long, quick_test
├── data/                # tinyshakespeare, tinystories
└── tokenizer/           # bytelevel, whitespace
```

## Key Patterns

### Component Registry
All components use a registry pattern for extensibility:
```python
@NORM_REGISTRY.register("rmsnorm")
class RMSNorm(nn.Module): ...

# Usage
norm = build_norm("rmsnorm", dim=512)
```

### Architecture Abstraction
Switch between LLaMA and GPT styles via config:
```python
model = TinyLM(vocab_size=32000, architecture="llama")  # RMSNorm, RoPE, SwiGLU
model = TinyLM(vocab_size=32000, architecture="gpt")    # LayerNorm, learned pos, GELU
```

### CUDA Kernels
Custom RMSNorm with autograd support:
- Forward: block-parallel, warp reductions
- Backward: fused dx + dw computation
- FP16/FP32 support, graceful CPU fallback

## Development Roadmap

See `ISSUES.md` for the full roadmap organized as GitHub-ready issues.

**Priority areas:**
1. Benchmarks & Profiling - measure before optimizing
2. CUDA/Triton optimizations - vectorized loads, kernel registry
3. Quantization - INT8/INT4 inference
4. Distributed training - DDP, FSDP
5. Experiments module - attention variants, caching strategies
6. Multi-modality - tiny image/audio encoders

## Commands

```bash
# Setup
uv sync
uv run python setup.py build_ext --inplace  # CUDA RMSNorm

# Training
uv run python train.py model=small
uv run python train.py model=medium model.architecture=gpt

# Inference
uv run python infer.py --ckpt outputs/.../best.pt --prompt "Once upon"

# Tests
uv run pytest tests/
```

## Code Style

- Type hints on all public APIs
- Docstrings for modules and classes
- Registry pattern for extensible components
- Hydra for configuration management
- pytest for testing
