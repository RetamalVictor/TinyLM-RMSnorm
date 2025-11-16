# TinyLM with Custom CUDA RMSNorm

**A GPT-style transformer with a custom fused CUDA kernel for RMSNorm, demonstrating end-to-end ML systems development from CUDA programming to training pipelines.**

This project showcases:
- Writing custom CUDA kernels with PyBind11 integration
- Implementing performance-critical transformer optimizations (KV-cache, mixed precision)
- Systematic benchmarking and performance analysis
- Production-ready ML infrastructure (Docker, CI/CD, comprehensive testing)

## Performance Results

### KV-Cache: 5× Faster at Scale

The KV-cache eliminates redundant computation during autoregressive generation. As context length grows, the speedup becomes dramatic:

![KV cache throughput](plots/fig_kv_curve_panels.png)

| Context | Without Cache | With Cache | Speedup |
|---------|--------------|------------|---------|
| 32      | 100 tok/s    | 103 tok/s  | 1.03×   |
| 128     | 50 tok/s     | 102 tok/s  | 2.04×   |
| 256     | 21 tok/s     | 102 tok/s  | **4.88×** |

Data: [`plots/kv_curve.csv`](plots/kv_curve.csv)

### Custom RMSNorm Kernel: 19% Faster

Fused CUDA implementation outperforms PyTorch's native operations in end-to-end generation:

![RMSNorm benchmark](plots/fig_rmsnorm.png)

**Real-world impact:**
- PyTorch reference: 11.86 ms/token
- Fused CUDA kernel: 10.00 ms/token
- **18.6% improvement** in generation throughput

Data: [`plots/ablation_rmsnorm.csv`](plots/ablation_rmsnorm.csv)

### Memory Scaling

KV-cache memory grows linearly with sequence length, as expected:

![VRAM vs sequence length](plots/fig_vram_seq.png)

Data: [`plots/vram_seq.csv`](plots/vram_seq.csv)

### Training Curve

Model training on TinyShakespeare dataset showing convergence:

![Training curve](plots/fig_training_curve.png)

Data: [`plots/train_log.csv`](plots/train_log.csv)

## CUDA Kernel Implementation

The RMSNorm kernel (`kernels/rmsnorm_cuda.cu`) implements both forward and backward passes with:

- **Block-wise parallel reduction** for RMS computation
- **Coalesced memory access** patterns for GPU efficiency
- **FP32 accumulation** in gradients for numerical stability
- **Shared memory** utilization for fast reductions

RMSNorm formula (ε=1e-6):

![RMSNorm equation](plots/eq_rmsnorm.png)

The fused kernel computes RMS and scaling in a single pass, avoiding multiple kernel launches.

## Architecture

**Model:** 6-layer GPT-style transformer (384 dim, 6 heads)
- Rotary Position Embeddings (RoPE) instead of learned positions
- RMSNorm instead of LayerNorm
- SiLU activations
- No bias terms (following modern LLM practices)

**KV-Cache Strategy:**
- Pre-allocated tensors (no reallocation during generation)
- Incremental updates per token
- Reduces complexity from O(T²) to O(T) per step

**Training Features:**
- Mixed precision (FP16) with automatic loss scaling
- Gradient accumulation for larger effective batch sizes
- Cosine LR scheduling with warmup
- Gradient clipping for stability

## Quick Start

### Prerequisites
- NVIDIA GPU with CUDA 12.1+
- PyTorch 2.2+
- Docker (recommended) or local Python 3.9+

### Docker (Recommended)

```bash
docker compose run --rm tinylm bash
```

### Build & Run

```bash
# 1. Build CUDA extension
python setup_cuda.py build_ext --inplace
pytest -q  # Validate kernel correctness

# 2. Prepare data
python data/prepare_tinyshakespeare.py

# 3. Train
python train.py \
  --data tinyshakespeare \
  --steps 1500 \
  --batch_size 8 \
  --seq_len 192 \
  --compile \
  --log_csv plots/train_log.csv

# 4. Generate text
python infer.py \
  --ckpt out/best.pt \
  --prompt "Once upon a time" \
  --max_new_tokens 100
```

### Run All Benchmarks

```bash
# Generate all plots and CSV data
OUTDIR=plots DO_TRAIN=0 bash scripts/run_all.sh
```

Outputs all figures and raw data to `plots/`:
- `fig_kv_curve_panels.png` - KV-cache scaling analysis
- `fig_rmsnorm.png` - Kernel microbenchmark
- `fig_training_curve.png` - Loss curves
- `fig_vram_seq.png` - Memory analysis
- Plus corresponding CSV files for reproducibility

## Repository Structure

```
TinyLM-RMSnorm/
├── kernels/
│   ├── rmsnorm_cuda.cu        # 195 lines of CUDA kernel code
│   └── rmsnorm_binding.cpp    # PyBind11 wrapper
├── model.py                   # Transformer with type hints
├── train.py                   # Training pipeline
├── infer.py                   # Generation with sampling
├── setup_cuda.py              # CUDA extension build
├── tests/test_rmsnorm.py      # Kernel validation
├── scripts/                   # Benchmarks and plotting
├── plots/                     # Generated figures + CSV
└── docker-compose.yml         # Development environment
```

## Testing

```bash
# Validate CUDA kernel
pytest tests/test_rmsnorm.py -v

# Tests verify:
# - Forward pass accuracy (atol=1e-4)
# - Backward pass gradients (atol=1e-3)
# - Numerical stability across dtypes
```

## Hardware Requirements

**Minimum:** NVIDIA GPU with 4GB VRAM, CUDA Compute Capability 7.0+

**Tested on:** RTX 2070, RTX 3090, RTX 4090

The codebase generates consistent results across different GPUs. Use `--label` flag to compare hardware:

```bash
LABEL=RTX4090 OUTDIR=plots bash scripts/run_all.sh
```

## Technical Highlights

This project demonstrates:

**CUDA/C++ Programming:**
- Custom kernel development with proper autograd integration
- PyBind11 for Python↔C++ interoperability
- Memory-efficient GPU code with coalesced access

**ML Systems:**
- Complete training pipeline from tokenization to inference
- Production features: mixed precision, gradient accumulation, checkpointing
- Comprehensive benchmarking methodology

**Software Engineering:**
- Type hints throughout Python code
- Unit tests with reference implementations
- Docker containerization
- CI/CD with GitHub Actions
- Clear documentation and reproducibility

## References

1. **RMSNorm:** Zhang & Sennrich (2019) - [arXiv:1910.07467](https://arxiv.org/abs/1910.07467)
2. **RoPE:** Su et al. (2024) - [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)
3. **GPT:** Radford et al. (2019) - Language Models are Unsupervised Multitask Learners
4. **LLaMA:** Touvron et al. (2023) - [arXiv:2302.13971](https://arxiv.org/abs/2302.13971)

## License

MIT - See [LICENSE](LICENSE)
