# TinyLM-Lab Roadmap Issues

GitHub-ready issue list for the TinyLM-Lab development roadmap.

---

## Refactoring (Prerequisites)

These issues should be completed first to enable the rest of the roadmap.

### Issue R1: Extract Cache Logic from Attention into CacheManager ✅ COMPLETED

**Labels:** `refactor`, `priority-critical`, `blocking`

**Status:** COMPLETED

**Description:**
Currently KV cache manipulation is hardcoded inside `MHA.forward()` (mha.py:107-111). This blocks implementing different caching strategies (block-based, compressed, paged).

**Current code (mha.py:107-111):**
```python
if cache is not None:
    cache['k'][:, :, start_pos:start_pos+T] = k
    cache['v'][:, :, start_pos:start_pos+T] = v
    k = cache['k'][:, :, :start_pos+T]
    v = cache['v'][:, :, :start_pos+T]
```

**Tasks:**
- [x] Create `tinylm/inference/cache_manager.py` with `CacheManager` base class
- [x] Implement `StandardCache` that replicates current behavior
- [x] Define interface: `allocate()`, `update()`, `get()`, `reset()`
- [x] Refactor `MHA` to accept `CacheManager` instead of raw dict
- [x] Update `TinyLM.create_kv_cache()` to return `CacheManager`
- [x] Add tests for cache manager (`tests/test_cache_manager.py`)

**Interface:**
```python
class CacheManager(ABC):
    @abstractmethod
    def allocate(self, batch_size: int, max_seq_len: int) -> None: ...

    @abstractmethod
    def update(self, layer_idx: int, k: Tensor, v: Tensor, positions: int) -> None: ...

    @abstractmethod
    def get(self, layer_idx: int, end_pos: int) -> Tuple[Tensor, Tensor]: ...
```

**Blocked by:** Nothing
**Blocks:** Issue 14 (CacheManager Abstraction), Issue 18 (Block-Based KV), Issue 19 (KV Compression)

---

### Issue R2: Create Kernel Backend Registry ✅ COMPLETED

**Labels:** `refactor`, `priority-critical`, `blocking`

**Status:** COMPLETED

**Description:**
CUDA kernels are loaded directly in component files (e.g., `rmsnorm.py` imports `rmsnorm_cuda`). Need a registry to hot-swap between PyTorch, CUDA, and Triton backends.

**Tasks:**
- [x] Create `tinylm/kernels/__init__.py` with `KernelBackendRegistry`
- [x] Create `tinylm/kernels/backends/` with `pytorch.py`, `cuda.py`, `triton.py`
- [x] Define interface for kernel implementations (`KernelBackend`, `RMSNormKernel`)
- [x] Add global backend selection: `set_backend("cuda")`
- [x] Add automatic fallback chain: CUDA → Triton → PyTorch
- [x] Refactor RMSNorm to use kernel registry
- [x] Add Hydra config for kernel backend selection (`conf/kernels/`)
- [x] Add tests for kernel registry (`tests/test_kernel_registry.py`)

**Interface:**
```python
# Usage
from tinylm.kernels import set_backend, get_backend, available_backends

set_backend("cuda")       # Force CUDA kernels
set_backend("pytorch")    # Force PyTorch fallback
set_backend("auto")       # Auto-select best available (default)

# Check available backends
print(available_backends())  # ["cuda", "pytorch"]

# Hydra config
# uv run python train.py kernels=cuda
# uv run python train.py kernels=pytorch
```

**Implementation:**
- `tinylm/kernels/base.py` - ABC interfaces (`KernelBackend`, `RMSNormKernel`)
- `tinylm/kernels/__init__.py` - Registry with fallback chain
- `tinylm/kernels/backends/pytorch.py` - Always-available fallback
- `tinylm/kernels/backends/cuda.py` - Wraps compiled CUDA kernels
- `tinylm/kernels/backends/triton.py` - Stub for Issue 3
- `conf/kernels/{auto,cuda,triton,pytorch}.yaml` - Hydra configs

**Blocked by:** Nothing
**Blocks:** Issue 3 (Triton RMSNorm), Issue 4 (Kernel Registry)

---

### Issue R3: Refactor train.py for Distributed Training Hooks ✅ COMPLETED

**Labels:** `refactor`, `priority-high`, `blocking`

**Status:** COMPLETED

**Description:**
`train.py` is monolithic with no hooks for distributed training. Need to extract trainer logic to support DDP/FSDP without major rewrites.

**Tasks:**
- [x] Create `tinylm/training/trainer.py` with `Trainer` class
- [x] Extract training loop from `train.py` into `Trainer.train_step()`
- [x] Add hooks: `on_train_start`, `on_step_start`, `on_step_end`, `on_eval`, `on_train_end`
- [x] Add `Trainer.setup_distributed()` method (no-op by default)
- [x] Add `Trainer.wrap_model()` hook for DDP/FSDP (stubs)
- [x] Refactor `train.py` to use `Trainer` with hooks
- [x] Ensure single-GPU training still works identically
- [x] Add tests for Trainer (`tests/test_trainer.py`)

**Implementation:**
```python
class Trainer:
    def __init__(self, model, optimizer, config, scheduler, scaler): ...

    # Distributed (stubs for Issue 10/11)
    def setup_distributed(self, backend="nccl"): ...
    def wrap_model(self, wrapper=None): ...  # "ddp", "fsdp", None

    # Training
    def train_step(self, batch): ...  # Single step with grad accumulation
    def train(self, train_dl, val_dl, start_step): ...  # Full loop
    def evaluate(self, dataloader, max_batches): ...

    # Hooks
    def add_hook(self, event, callback): ...  # Register callbacks

    # Checkpointing
    def state_dict(self): ...
    def load_state_dict(self, state): ...
```

**Files:**
- `tinylm/training/trainer.py` - Trainer, TrainerConfig, TrainerState
- `tinylm/training/__init__.py` - Updated exports
- `train.py` - Refactored to use Trainer with hooks
- `tests/test_trainer.py` - 18 tests

**Blocked by:** Nothing
**Blocks:** Issue 10 (DDP), Issue 11 (FSDP), Issue 12 (Gradient Accumulation)

---

### Issue R4: Separate Attention Computation from Position/Cache Logic ✅ COMPLETED

**Labels:** `refactor`, `priority-medium`

**Status:** COMPLETED

**Description:**
`MHA.forward()` mixes three concerns: QKV projection, positional embedding application, and attention computation. This makes it hard to swap attention implementations (flash, sparse).

**Tasks:**
- [x] Extract attention computation into separate function/class
- [x] Create `AttentionOp` interface with `compute(q, k, v, mask)` method
- [x] Register implementations: `standard`, `flash` (via F.scaled_dot_product_attention flags)
- [x] Keep MHA as orchestrator that composes: projection → pos_emb → cache → attention_op → output
- [x] Add config option to select attention implementation

**Implementation:**
- `tinylm/components/attention/ops/base.py` - `AttentionOp` ABC
- `tinylm/components/attention/ops/standard.py` - `StandardAttentionOp` (auto-select best kernel)
- `tinylm/components/attention/ops/flash.py` - `FlashAttentionOp`, `MemoryEfficientAttentionOp`
- `tinylm/components/registry.py` - `ATTENTION_OP_REGISTRY`
- `tinylm/components/attention/mha.py` - MHA now accepts `attention_op` parameter
- `tests/test_attention_ops.py` - 31 tests

**Usage:**
```python
from tinylm.components.attention import MHA, build_attention_op

# Create MHA with specific attention op
mha = MHA(dim=512, n_heads=8, attention_op="flash")

# Swap at runtime
mha.attention_op = build_attention_op("standard")
```

**Blocked by:** Issue R1 (CacheManager)
**Blocks:** Issue 15 (AttentionEngine)

---

### Issue R5: Add Quantization Registry Pattern ✅ COMPLETED

**Labels:** `refactor`, `priority-medium`

**Status:** COMPLETED

**Description:**
Current quantization only supports ternary via bittorch. Need registry pattern to support INT8, INT4, and future methods.

**Current limitation:** `make_linear()` only has ternary path.

**Tasks:**
- [x] Create `QUANT_REGISTRY` similar to `NORM_REGISTRY`
- [x] Register quantization methods: `none`, `ternary`, `int8`, `int4`
- [x] Update `make_linear()` to use registry
- [x] Add `QuantMethod` base class with `quantize()`, `dequantize()` methods
- [x] Support both weight-only and weight+activation quantization

**Implementation:**
- `tinylm/quant/base.py` - `QuantMethod` ABC, `QuantParams`, `QuantRegistry`, `QUANT_REGISTRY`
- `tinylm/quant/methods/none.py` - `NoneQuantMethod` (always available)
- `tinylm/quant/methods/ternary.py` - `TernaryQuantMethod` (BitTorch wrapper)
- `tinylm/quant/methods/int8.py` - `Int8QuantMethod` (stub for Issue 6)
- `tinylm/quant/methods/int4.py` - `Int4QuantMethod` (stub for Issue 8)
- `tinylm/quant/factory.py` - `make_linear()` using registry
- `tests/test_quant_registry.py` - 44 tests

**Interface:**
```python
@QUANT_REGISTRY.register("int8")
class Int8Quantizer(QuantMethod):
    def quantize_weights(self, weight): ...
    def forward(self, x, weight_int8, scale): ...
```

**Blocked by:** Nothing
**Blocks:** Issue 6 (INT8), Issue 7 (Calibration), Issue 8 (INT4)

---

## Summary: Refactor Order

All critical refactoring issues are now **COMPLETED**:

```
R1 (CacheManager) ✅ ───────────────┬──→ Issue 14, 18, 19
                                    │
R2 (Kernel Registry) ✅ ────────────┼──→ Issue 3, 4
                                    │
R3 (Trainer Refactor) ✅ ───────────┼──→ Issue 10, 11, 12
                                    │
R4 (Attention Separation) ✅ ───────┼──→ Issue 15
        ↑                           │
        └── depends on R1           │
                                    │
R5 (Quant Registry) ✅ ─────────────┴──→ Issue 6, 7, 8
```

**Completed order:** R1 → R2 → R3 → R5 → R4

All prerequisite refactors are done! The codebase now has:
- **CacheManager**: Abstracted KV cache with StandardCache implementation
- **Kernel Backend Registry**: Hot-swappable kernels (PyTorch, CUDA, Triton)
- **Trainer**: Modular trainer with hooks for DDP/FSDP
- **AttentionOp**: Composable attention backends (standard, flash, memory_efficient)
- **Quant Registry**: Pluggable quantization methods (none, ternary, int8/int4 stubs)

---

## Benchmarks & Profiling

### Issue 1: Implement Benchmark Framework

**Labels:** `benchmarks`, `priority-high`

**Description:**
Create benchmark scripts measuring latency, memory, and tokens/sec. Output JSON + charts. Add torch.profiler support.

**Tasks:**
- [ ] Create `benchmarks/bench_forward.py` - forward pass latency across batch sizes
- [ ] Create `benchmarks/bench_generate.py` - generation throughput (tokens/sec with KV cache)
- [ ] Create `benchmarks/bench_memory.py` - peak memory usage tracking
- [ ] Create `benchmarks/bench_kernels.py` - CUDA vs PyTorch comparison
- [ ] Add `torch.profiler` integration with CUDA activity tracing
- [ ] Output results to `benchmarks/results/benchmark_results.json`
- [ ] Add matplotlib charts for visualization
- [ ] Document usage in README

**Acceptance Criteria:**
```bash
uv run python benchmarks/bench_forward.py --model small --batch-sizes 1,8,32
# Outputs: latency table + JSON results
```

---

## CUDA & Kernel Optimizations

### Issue 2: Implement Vectorized CUDA RMSNorm Kernel

**Labels:** `cuda`, `optimization`, `priority-high`

**Description:**
Rewrite RMSNorm with float4 vectorized loads for 4x memory bandwidth improvement. Use shared memory for reductions.

**Tasks:**
- [ ] Add `float4` vectorized loads in `kernels/rmsnorm_cuda.cu`
- [ ] Ensure proper alignment handling for non-divisible dimensions
- [ ] Add shared memory optimization for partial sums
- [ ] Benchmark against current implementation
- [ ] Update backward pass with vectorization

**Technical Details:**
```cuda
// Current (scalar)
float xi = to_float(x_row[i]);

// Optimized (vectorized)
float4 x4 = reinterpret_cast<const float4*>(x_row)[i];
float sumsq = x4.x*x4.x + x4.y*x4.y + x4.z*x4.z + x4.w*x4.w;
```

---

### Issue 3: Implement Triton RMSNorm Kernel

**Labels:** `triton`, `optimization`, `priority-medium`

**Description:**
Create Triton version of RMSNorm for easier iteration and comparison. Triton provides better portability and often competitive performance.

**Tasks:**
- [ ] Create `tinylm/kernels/triton/rmsnorm.py`
- [ ] Implement forward pass with Triton
- [ ] Implement backward pass with Triton
- [ ] Add autograd Function wrapper
- [ ] Benchmark against CUDA and PyTorch versions
- [ ] Document performance comparison

---

### Issue 4: Add Kernel Registry for Swappable Backends

**Labels:** `architecture`, `kernels`, `priority-medium`

**Description:**
Design registry to easily hot-swap between PyTorch, CUDA, and Triton kernels at runtime.

**Tasks:**
- [ ] Create `tinylm/kernels/__init__.py` with `KernelRegistry`
- [ ] Support backend selection via config: `kernel_backend: "cuda" | "triton" | "pytorch"`
- [ ] Add runtime fallback chain (CUDA → Triton → PyTorch)
- [ ] Integrate with existing `NORM_REGISTRY`
- [ ] Add benchmarking hooks per backend

**API Design:**
```python
from tinylm.kernels import set_backend, get_backend

set_backend("triton")  # or "cuda", "pytorch"
model = TinyLM(...)    # uses selected backend
```

---

### Issue 5: Implement Fused LayerNorm + Linear Kernel

**Labels:** `cuda`, `optimization`, `priority-low`

**Description:**
Fuse final LayerNorm with output projection for reduced memory bandwidth.

**Tasks:**
- [ ] Implement fused kernel in CUDA
- [ ] Add Triton version
- [ ] Benchmark memory savings
- [ ] Integrate with transformer forward pass

---

## Quantization

### Issue 6: Implement Weight-Only INT8 Quantization

**Labels:** `quantization`, `priority-high`

**Description:**
Add INT8 quantizer for weights with dequantization-on-the-fly. Integrate with infer.py.

**Tasks:**
- [ ] Create `tinylm/quant/int8.py`
- [ ] Implement per-channel weight quantization
- [ ] Add dequantization in forward pass
- [ ] Support saving/loading quantized checkpoints
- [ ] Add `--quantize int8` flag to `infer.py`
- [ ] Benchmark memory reduction and latency

**Implementation:**
```python
class Int8Linear(nn.Module):
    def __init__(self, in_features, out_features):
        self.weight_int8 = ...  # int8 storage
        self.scale = ...        # fp32 per-channel scale

    def forward(self, x):
        w = self.weight_int8.float() * self.scale
        return F.linear(x, w)
```

---

### Issue 7: Add Activation Calibration for Static INT8

**Labels:** `quantization`, `priority-medium`

**Description:**
Implement calibration pass to determine activation ranges for static quantization.

**Tasks:**
- [ ] Create `tinylm/quant/calibrate.py`
- [ ] Implement histogram-based range estimation
- [ ] Support percentile clipping (99.9%)
- [ ] Save calibration data to checkpoint
- [ ] Add calibration script `scripts/calibrate.py`

---

### Issue 8: Prototype INT4 (GPTQ/AWQ) Quantization

**Labels:** `quantization`, `priority-low`, `research`

**Description:**
Implement 4-bit quantization backend using GPTQ or AWQ algorithm.

**Tasks:**
- [ ] Research GPTQ vs AWQ tradeoffs
- [ ] Implement weight quantization with Hessian-based optimization
- [ ] Create custom INT4 matmul kernel (or use existing libraries)
- [ ] Benchmark against INT8 and FP16
- [ ] Document accuracy vs compression tradeoffs

---

### Issue 9: Quantization-Aware Training (QAT)

**Labels:** `quantization`, `training`, `priority-low`

**Description:**
Add fake quantization during training for better quantized model accuracy.

**Tasks:**
- [ ] Implement fake quantization ops
- [ ] Add QAT config to Hydra
- [ ] Support gradual quantization schedule
- [ ] Compare QAT vs post-training quantization accuracy

---

## Distributed Training

### Issue 10: Add DDP Training Support

**Labels:** `distributed`, `training`, `priority-high`

**Description:**
Enable multi-GPU training via PyTorch DDP with torchrun launcher.

**Tasks:**
- [ ] Add DDP wrapper in `train.py`
- [ ] Add Hydra config for distributed settings
- [ ] Support `torchrun --nproc_per_node=N train.py`
- [ ] Handle gradient synchronization
- [ ] Add proper logging for rank 0 only
- [ ] Test on 2+ GPUs

**Config:**
```yaml
# conf/training/distributed.yaml
distributed:
  enabled: true
  backend: nccl
  find_unused_parameters: false
```

---

### Issue 11: Implement FSDP Support

**Labels:** `distributed`, `training`, `priority-medium`

**Description:**
Add Fully Sharded Data Parallel for training models larger than single GPU memory.

**Tasks:**
- [ ] Add FSDP wrapping with auto_wrap_policy
- [ ] Configure sharding strategy (FULL_SHARD, SHARD_GRAD_OP)
- [ ] Handle FSDP-compatible checkpointing
- [ ] Add mixed precision with FSDP
- [ ] Benchmark memory savings

---

### Issue 12: Add Gradient Accumulation

**Labels:** `training`, `priority-medium`

**Description:**
Implement gradient accumulation for simulating larger batch sizes.

**Tasks:**
- [ ] Add `accumulation_steps` config
- [ ] Modify training loop to accumulate gradients
- [ ] Adjust learning rate scaling
- [ ] Update logging to show effective batch size

---

### Issue 13: Add Activation Checkpointing

**Labels:** `training`, `memory`, `priority-medium`

**Description:**
Enable memory-efficient training via activation checkpointing (gradient checkpointing).

**Tasks:**
- [ ] Add `torch.utils.checkpoint` to transformer blocks
- [ ] Make checkpointing configurable per-layer
- [ ] Benchmark memory savings vs compute overhead
- [ ] Add to Hydra config

---

## Architecture Abstractions

### Issue 14: Create CacheManager Abstraction

**Labels:** `architecture`, `inference`, `priority-high`

**Description:**
Implement unified KV cache interface supporting multiple caching strategies.

**Tasks:**
- [ ] Create `tinylm/inference/cache_manager.py`
- [ ] Abstract current KV cache behind interface
- [ ] Support: standard, block-based, compressed
- [ ] Add cache statistics (hit rate, memory usage)
- [ ] Design for future C2C communication

**Interface:**
```python
class CacheManager(ABC):
    def allocate(self, batch_size, max_seq_len) -> Cache
    def update(self, cache, layer_idx, k, v, positions)
    def get(self, cache, layer_idx, positions) -> Tuple[K, V]
    def compress(self, cache)  # optional
```

---

### Issue 15: Create AttentionEngine Abstraction

**Labels:** `architecture`, `priority-medium`

**Description:**
Allow swappable attention implementations (standard, sparse, flash, custom).

**Tasks:**
- [ ] Create `tinylm/components/attention/engine.py`
- [ ] Register attention variants: `standard`, `flash`, `sparse`
- [ ] Support attention mask customization
- [ ] Add config-driven selection
- [ ] Benchmark different implementations

---

### Issue 16: Create RoutingEngine Abstraction

**Labels:** `architecture`, `research`, `priority-low`

**Description:**
Add modular routing logic for experiments (token routing, head routing, early exit).

**Tasks:**
- [ ] Create `tinylm/routing/` module
- [ ] Implement token router (for MoE experiments)
- [ ] Implement head router (sparse attention)
- [ ] Add skip routing (early exit)
- [ ] Design clean integration with transformer blocks

---

## Experiments Module

### Issue 17: Create Experiments Sandbox Structure

**Labels:** `research`, `priority-high`

**Description:**
Set up `experiments/` directory with plug-and-play research modules.

**Tasks:**
- [ ] Create `experiments/` directory structure
- [ ] Add base experiment class/interface
- [ ] Create example experiment template
- [ ] Document how to add new experiments
- [ ] Add experiment runner script

**Structure:**
```
experiments/
├── __init__.py
├── base.py              # ExperimentBase class
├── attention/
│   ├── sparse_attention.py
│   └── linear_attention.py
├── caching/
│   ├── block_cache.py
│   ├── compressed_cache.py
│   └── c2c_prototype.py
├── routing/
│   └── early_exit.py
└── README.md
```

---

### Issue 18: Implement Block-Based KV Cache

**Labels:** `research`, `caching`, `priority-medium`

**Description:**
Prototype block-structured KV caching (PagedAttention-style) to reduce memory fragmentation.

**Tasks:**
- [ ] Implement block allocator
- [ ] Create block table for KV mapping
- [ ] Modify attention to use block indices
- [ ] Benchmark memory efficiency vs standard cache
- [ ] Document tradeoffs

---

### Issue 19: Implement KV Cache Compression

**Labels:** `research`, `caching`, `priority-low`

**Description:**
Experiment with KV cache compression techniques (quantization, pruning, merging).

**Tasks:**
- [ ] Implement KV quantization (FP16 → INT8)
- [ ] Add attention-score-based pruning
- [ ] Experiment with token merging
- [ ] Benchmark accuracy vs memory tradeoffs

---

### Issue 20: Implement Cache-to-Cache Communication Prototype

**Labels:** `research`, `experimental`, `priority-low`

**Description:**
Experiment with novel cache exchange mechanisms for multi-model or multi-request scenarios.

**Tasks:**
- [ ] Design C2C protocol
- [ ] Implement cache serialization
- [ ] Create prototype communication layer
- [ ] Benchmark latency overhead
- [ ] Document findings

---

### Issue 21: Add Speculative Decoding Module

**Labels:** `research`, `inference`, `priority-medium`

**Description:**
Prototype small-draft/large-target speculative decoding on tiny models.

**Tasks:**
- [ ] Implement draft model generation
- [ ] Add verification step with main model
- [ ] Handle token rejection and resampling
- [ ] Benchmark speedup vs overhead
- [ ] Support configurable draft length

---

### Issue 22: Implement Sparse/Top-K Attention

**Labels:** `research`, `attention`, `priority-low`

**Description:**
Add sparse attention variants for efficiency experiments.

**Tasks:**
- [ ] Implement top-k attention selection
- [ ] Add local + global attention pattern
- [ ] Benchmark vs full attention
- [ ] Measure accuracy impact

---

## Multi-Modality

### Issue 23: Add Tiny Image Encoder

**Labels:** `multimodal`, `priority-low`

**Description:**
Implement minimal CNN or ViT to produce image tokens for multimodal experiments.

**Tasks:**
- [ ] Create `tinylm/encoders/image.py`
- [ ] Implement small CNN encoder (ResNet-style)
- [ ] Implement tiny ViT encoder option
- [ ] Add patch embedding layer
- [ ] Output compatible token embeddings
- [ ] Add example dataset (tiny images)

---

### Issue 24: Add Tiny Audio Encoder

**Labels:** `multimodal`, `priority-low`

**Description:**
Implement small 1D conv encoder for audio → tokens.

**Tasks:**
- [ ] Create `tinylm/encoders/audio.py`
- [ ] Implement 1D convolutional encoder
- [ ] Add mel-spectrogram preprocessing
- [ ] Output token-compatible embeddings
- [ ] Add example with small audio clips

---

### Issue 25: Create Multimodal TinyLM Demonstrator

**Labels:** `multimodal`, `demo`, `priority-low`

**Description:**
Combine tiny encoders + text decoder for image captioning or audio transcription demo.

**Tasks:**
- [ ] Create multimodal model wrapper
- [ ] Implement cross-attention between modalities
- [ ] Train on tiny captioning dataset
- [ ] Create demo script
- [ ] Document architecture and results

---

## Models & Releases

### Issue 26: Release Pretrained Tiny Models

**Labels:** `models`, `release`, `priority-medium`

**Description:**
Train and upload reference tiny models for benchmarking and demos.

**Tasks:**
- [ ] Train TinyLM-Shakespeare (character-level)
- [ ] Train TinyLM-Stories (BPE tokenizer)
- [ ] Upload to HuggingFace Hub or GitHub releases
- [ ] Add download script
- [ ] Document model cards

**Models:**
- `tinylm-shakespeare-small` (~10M params)
- `tinylm-stories-small` (~10M params)
- `tinylm-stories-medium` (~50M params)

---

## Documentation

### Issue 27: Documentation Overhaul

**Labels:** `documentation`, `priority-medium`

**Description:**
Add comprehensive documentation including diagrams, tutorials, and API reference.

**Tasks:**
- [ ] Add architecture diagrams to README
- [ ] Write tutorial: adding custom components
- [ ] Write tutorial: adding custom kernels
- [ ] Write tutorial: running experiments
- [ ] Add API reference docs
- [ ] Add contributing guide

---

### Issue 28: Technical Report Draft

**Labels:** `documentation`, `research`, `priority-low`

**Description:**
Write PDF/markdown report outlining architecture, experiments, benchmarks, and findings.

**Tasks:**
- [ ] Outline report structure
- [ ] Document architecture decisions
- [ ] Include benchmark results
- [ ] Summarize experiment findings
- [ ] Add figures and diagrams
- [ ] Publish as PDF or arXiv-style doc

---

## Infrastructure

### Issue 29: Add CI Performance Tracking

**Labels:** `ci`, `benchmarks`, `priority-low`

**Description:**
Add CI job that tracks performance regressions on each PR.

**Tasks:**
- [ ] Create benchmark CI workflow
- [ ] Store historical benchmark results
- [ ] Add performance regression detection
- [ ] Comment on PRs with benchmark changes

---

### Issue 30: RunPod Training Templates

**Labels:** `infrastructure`, `priority-low`

**Description:**
Add RunPod training scripts with auto-resume and profiling.

**Tasks:**
- [ ] Create RunPod setup script
- [ ] Add auto-resume from S3/GCS
- [ ] Include profiling configuration
- [ ] Document cloud training workflow

---

## Summary by Priority

### Critical (Refactors - Do First)
- **R1**: Extract Cache Logic into CacheManager → unblocks caching experiments
- **R2**: Create Kernel Backend Registry → unblocks Triton/CUDA work
- **R3**: Refactor train.py for Distributed Hooks → unblocks DDP/FSDP
- **R5**: Add Quantization Registry Pattern → unblocks INT8/INT4
- **R4**: Separate Attention from Cache Logic → unblocks attention experiments

**Recommended order:** R1 → R2 → R3 → R5 → R4

### High Priority (After Refactors)
1. Benchmark Framework
2. Vectorized CUDA RMSNorm
3. Weight-Only INT8 Quantization
4. DDP Training Support
5. Experiments Sandbox Structure

### Medium Priority
6. Triton RMSNorm
7. Activation Calibration
8. FSDP Support
9. Gradient Accumulation
10. Activation Checkpointing
11. AttentionEngine Abstraction
12. Block-Based KV Cache
13. Speculative Decoding
14. Pretrained Models
15. Documentation Overhaul

### Low Priority
16. Fused LayerNorm + Linear
17. INT4 Quantization
18. QAT
19. RoutingEngine
20. KV Compression
21. C2C Communication
22. Sparse Attention
25. Image Encoder
26. Audio Encoder
27. Multimodal Demo
28. Technical Report
29. CI Performance Tracking
30. RunPod Templates
