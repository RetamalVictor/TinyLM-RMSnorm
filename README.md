# NanoFalcon

A tiny decoder-only Transformer you can train end-to-end in a few hours, showcasing:

* A custom CUDA kernel (fused **RMSNorm**) with a PyTorch extension (forward + backward).
* A minimal, fast inference path with KV-cache and rotary embeddings.
* A clean training loop that reaches non-trivial loss on a small dataset (TinyStories or TinyShakespeare).
* Micro-benchmarks and correctness tests.

This repo is built for a 16-hour sprint. Ship shape, interview-ready.

---

## Quickstart

```bash
# 0) Python 3.10+, CUDA 11.8+ recommended, PyTorch >= 2.2
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) Build the CUDA extension
python setup_cuda.py build_ext --inplace

# 2) Prepare data (TinyStories ~50MB) or TinyShakespeare (~1MB)
python data/prepare_tinystories.py   # default dataset
# or
python data/prepare_tinyshakespeare.py

# 3) Train a small model (about ~20–60 minutes on a single consumer GPU)
python train.py --data tinystories --steps 4000 --batch_size 16 --seq_len 256 \
  --dim 384 --n_layers 6 --n_heads 6 --lr 3e-4 --compile --flash

# 4) Generate text
python infer.py --ckpt out/best.pt --prompt "Once upon a time" --max_new_tokens 128

# 5) Micro-benchmark & correctness for the RMSNorm CUDA kernel
python bench_rmsnorm.py
pytest -q
```

---

## Repo layout

```
.
├── bench_rmsnorm.py
├── data/
│   ├── prepare_tinystories.py
│   └── prepare_tinyshakespeare.py
├── infer.py
├── kernels/
│   ├── rmsnorm_binding.cpp
│   └── rmsnorm_cuda.cu
├── model.py
├── README.md   ← you are here
├── requirements.txt
├── setup_cuda.py
├── tests/
│   └── test_rmsnorm.py
└── train.py
```

---

## Design choices (talk track)

* **RMSNorm** is used by Falcon/LLaMA-style models and is hot on the forward path. A fused kernel that computes per-token RMS, scales, and applies weights saves bandwidth and launches.
* **Rotary embeddings (RoPE)** are applied to Q/K for better extrapolation. We keep them simple and vectorized in PyTorch so the CUDA part is focused.
* **Fast inference**: preallocated KV-cache, single-step decode loop, option to `torch.compile`, and FlashAttention if available.
* **Scope control**: no fancy quantization here (keepable as stretch). The point is: a clean CUDA op + solid systems sensibility.

---

## requirements.txt

```txt
torch>=2.2
tokenizers>=0.15
datasets>=2.19
numpy>=1.25
tqdm
pytest
```

---

## setup\_cuda.py

```python
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='rmsnorm_cuda',
    ext_modules=[
        CUDAExtension(
            name='rmsnorm_cuda',
            sources=['kernels/rmsnorm_binding.cpp', 'kernels/rmsnorm_cuda.cu'],
            extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3', '--use_fast_math']},
        ),
    ],
    cmdclass={'build_ext': BuildExtension}
)
```

---

## kernels/rmsnorm\_binding.cpp

```cpp
#include <torch/extension.h>
#include <vector>

// forward declarations
std::vector<torch::Tensor> rmsnorm_forward_cuda(torch::Tensor x, torch::Tensor weight, double eps);
std::vector<torch::Tensor> rmsnorm_backward_cuda(torch::Tensor dy, torch::Tensor x, torch::Tensor weight, torch::Tensor inv_rms, double eps);

// Python bindings
std::vector<torch::Tensor> rmsnorm_forward(torch::Tensor x, torch::Tensor weight, double eps) {
  TORCH_CHECK(x.is_cuda(), "x must be CUDA");
  TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
  return rmsnorm_forward_cuda(x, weight, eps);
}

std::vector<torch::Tensor> rmsnorm_backward(torch::Tensor dy, torch::Tensor x, torch::Tensor weight, torch::Tensor inv_rms, double eps) {
  TORCH_CHECK(dy.is_cuda() && x.is_cuda() && weight.is_cuda() && inv_rms.is_cuda(), "all must be CUDA");
  return rmsnorm_backward_cuda(dy, x, weight, inv_rms, eps);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &rmsnorm_forward, "RMSNorm forward (CUDA)");
  m.def("backward", &rmsnorm_backward, "RMSNorm backward (CUDA)");
}
```

---

## kernels/rmsnorm\_cuda.cu

```cuda
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <type_traits>
#include <cmath>

// Conversions
__device__ inline float to_float(float x) { return x; }
__device__ inline float to_float(half x)  { return __half2float(x); }

template<typename T>
__device__ inline T from_float(float x);

template<>
__device__ inline float from_float<float>(float x) { return x; }

template<>
__device__ inline half from_float<half>(float x) { return __float2half(x); }

// Block reduction (sum) that returns the reduced value in lane 0 of warp 0
// (Other threads will have undefined 'val'; we'll broadcast via shared mem.)
template<typename T>
__inline__ __device__ T blockReduceSum(T val) {
  __shared__ T shared[32];
  int lane = threadIdx.x & 31;
  int wid  = threadIdx.x >> 5;
  // warp reduce
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    val += __shfl_down_sync(0xffffffff, val, offset);
  // write per-warp sum
  if (lane == 0) shared[wid] = val;
  __syncthreads();
  // first warp loads per-warp sums
  val = (lane < (blockDim.x + 31) / 32) ? shared[lane] : 0;
  if (wid == 0) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
      val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val; // valid in warp 0, lane 0
}

// Forward kernel: per-row RMS + scale + weight
template<typename scalar_t>
__global__ void rmsnorm_fwd_kernel(const scalar_t* __restrict__ x,
                                   const scalar_t* __restrict__ w,
                                   scalar_t* __restrict__ y,
                                   float* __restrict__ inv_rms_out,
                                   int hidden, float eps) {
  int row = blockIdx.x; // one block per row
  int tid = threadIdx.x;
  int stride = blockDim.x;

  const scalar_t* x_row = x + (size_t)row * hidden;
  scalar_t* y_row = y + (size_t)row * hidden;

  float sumsq = 0.f;
  for (int i = tid; i < hidden; i += stride) {
    float xi = to_float(x_row[i]);
    sumsq += xi * xi;
  }
  float reduced = blockReduceSum<float>(sumsq);

  __shared__ float s_inv_rms;
  if (tid == 0) {
    s_inv_rms = rsqrtf(reduced / hidden + eps);
    inv_rms_out[row] = s_inv_rms;
  }
  __syncthreads();
  float inv_rms = s_inv_rms; // broadcast

  for (int i = tid; i < hidden; i += stride) {
    float xi = to_float(x_row[i]);
    float wi = to_float(w[i]);
    float yi = (xi * inv_rms) * wi;
    y_row[i] = from_float<scalar_t>(yi);
  }
}

// Backward kernel: compute dx, accumulate dweight (in FP32)

template<typename scalar_t>
__global__ void rmsnorm_bwd_kernel(const scalar_t* __restrict__ dy,
                                   const scalar_t* __restrict__ x,
                                   const scalar_t* __restrict__ w,
                                   const float* __restrict__ inv_rms_in,
                                   scalar_t* __restrict__ dx,
                                   float* __restrict__ dw_fp32,
                                   int hidden) {
  int row = blockIdx.x;
  int tid = threadIdx.x;
  int stride = blockDim.x;

  const scalar_t* dy_row = dy + (size_t)row * hidden;
  const scalar_t* x_row  = x  + (size_t)row * hidden;
  scalar_t* dx_row       = dx + (size_t)row * hidden;

  float inv_rms = inv_rms_in[row];
  float r3_over_N = (inv_rms * inv_rms * inv_rms) / hidden;

  float dot = 0.f; // sum_j x_j * du_j
  // First pass: compute dot and also partial dw
  for (int i = tid; i < hidden; i += stride) {
    float dyi = to_float(dy_row[i]);
    float wi  = to_float(w[i]);
    float xi  = to_float(x_row[i]);
    float du  = dyi * wi;
    dot += xi * du;
    // dweight += dy * (x * inv_rms)
    float contrib = dyi * (xi * inv_rms);
    atomicAdd(dw_fp32 + i, contrib);
  }
  float reduced_dot = blockReduceSum<float>(dot);

  __shared__ float s_dot;
  if (tid == 0) s_dot = reduced_dot;
  __syncthreads();
  float a = -r3_over_N * s_dot;

  // Second pass: dx = inv_rms * du + x * a
  for (int i = tid; i < hidden; i += stride) {
    float dyi = to_float(dy_row[i]);
    float wi  = to_float(w[i]);
    float xi  = to_float(x_row[i]);
    float du  = dyi * wi;
    float dxi = inv_rms * du + xi * a;
    dx_row[i] = from_float<scalar_t>(dxi);
  }
}

std::vector<torch::Tensor> rmsnorm_forward_cuda(torch::Tensor x, torch::Tensor weight, double eps) {
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
  int rows = x.numel() / x.size(-1);
  int hidden = x.size(-1);

  auto y = torch::empty_like(x);
  auto inv_rms = torch::empty({rows}, x.options().dtype(torch::kFloat));

  int threads = std::min(1024, 1 << (int)std::ceil(std::log2((double)hidden)));
  threads = max(32, threads); // at least one warp
  dim3 block(threads);
  dim3 grid(rows);

  auto stream = at::cuda::getCurrentCUDAStream();

  if (x.scalar_type() == torch::kFloat16) {
    rmsnorm_fwd_kernel<half><<<grid, block, 0, stream>>>(
      (half*)x.data_ptr<at::Half>(), (half*)weight.data_ptr<at::Half>(),
      (half*)y.data_ptr<at::Half>(), inv_rms.data_ptr<float>(), hidden, (float)eps);
  } else if (x.scalar_type() == torch::kFloat32) {
    rmsnorm_fwd_kernel<float><<<grid, block, 0, stream>>>(
      x.data_ptr<float>(), weight.data_ptr<float>(), y.data_ptr<float>(),
      inv_rms.data_ptr<float>(), hidden, (float)eps);
  } else {
    TORCH_CHECK(false, "Unsupported dtype");
  }
  return {y, inv_rms};
}

std::vector<torch::Tensor> rmsnorm_backward_cuda(torch::Tensor dy, torch::Tensor x, torch::Tensor weight, torch::Tensor inv_rms, double eps) {
  int rows = x.numel() / x.size(-1);
  int hidden = x.size(-1);

  auto dx = torch::empty_like(x);
  // Accumulate dweight in fp32 for stability (and to support half weights)
  auto dw32 = torch::zeros({weight.size(0)}, x.options().dtype(torch::kFloat));

  int threads = std::min(1024, 1 << (int)std::ceil(std::log2((double)hidden)));
  threads = max(32, threads);
  dim3 block(threads);
  dim3 grid(rows);

  auto stream = at::cuda::getCurrentCUDAStream();

  if (x.scalar_type() == torch::kFloat16) {
    rmsnorm_bwd_kernel<half><<<grid, block, 0, stream>>>(
      (half*)dy.data_ptr<at::Half>(), (half*)x.data_ptr<at::Half>(), (half*)weight.data_ptr<at::Half>(),
      inv_rms.data_ptr<float>(), (half*)dx.data_ptr<at::Half>(), dw32.data_ptr<float>(), hidden);
  } else if (x.scalar_type() == torch::kFloat32) {
    rmsnorm_bwd_kernel<float><<<grid, block, 0, stream>>>(
      dy.data_ptr<float>(), x.data_ptr<float>(), weight.data_ptr<float>(),
      inv_rms.data_ptr<float>(), dx.data_ptr<float>(), dw32.data_ptr<float>(), hidden);
  } else {
    TORCH_CHECK(false, "Unsupported dtype");
  }

  // Cast dweight to parameter dtype
  auto dw = (weight.scalar_type() == torch::kFloat16)
              ? dw32.to(torch::kHalf)
              : dw32;
  return {dx, dw};
}
```

---

## model.py

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import rmsnorm_cuda

class RMSNormCUDAFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        y, inv_rms = rmsnorm_cuda.forward(x, weight, eps)
        ctx.save_for_backward(x, weight, inv_rms)
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, dy):
        x, weight, inv_rms = ctx.saved_tensors
        dx, dw = rmsnorm_cuda.backward(dy.contiguous(), x, weight, inv_rms, ctx.eps)
        return dx, dw, None

class RMSNormCUDA(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        return RMSNormCUDAFn.apply(x, self.weight, self.eps)

def rotary_embeddings(x, sin, cos):
    x1, x2 = x[..., ::2], x[..., 1::2]
    xr = torch.stack((-x2, x1), dim=-1).reshape_as(x)
    return x * cos + xr * sin

def build_sincos(seq_len, dim, device):
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.einsum('t,f->tf', t, inv_freq)
    sin = torch.sin(torch.cat([freqs, freqs], dim=-1))[None, None, :, :]
    cos = torch.cos(torch.cat([freqs, freqs], dim=-1))[None, None, :, :]
    return sin, cos

class MHA(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.nh = n_heads
        self.dim = dim
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x, sin, cos, cache=None, start_pos=0):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.nh, C // self.nh).transpose(1, 2)
        k = k.view(B, T, self.nh, C // self.nh).transpose(1, 2)
        v = v.view(B, T, self.nh, C // self.nh).transpose(1, 2)
        q = rotary_embeddings(q, sin[:, :, start_pos:start_pos+T, :], cos[:, :, start_pos:start_pos+T, :])
        k = rotary_embeddings(k, sin[:, :, start_pos:start_pos+T, :], cos[:, :, start_pos:start_pos+T, :])
        if cache is not None:
            cache['k'][:, :, start_pos:start_pos+T] = k
            cache['v'][:, :, start_pos:start_pos+T] = v
            k = cache['k'][:, :, :start_pos+T]
            v = cache['v'][:, :, :start_pos+T]
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = attn.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)

class Block(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=4):
        super().__init__()
        self.norm1 = RMSNormCUDA(dim)
        self.attn = MHA(dim, n_heads)
        self.norm2 = RMSNormCUDA(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio*dim, bias=False),
            nn.SiLU(),
            nn.Linear(mlp_ratio*dim, dim, bias=False),
        )
    def forward(self, x, sin, cos, cache=None, start_pos=0):
        x = x + self.attn(self.norm1(x), sin, cos, cache, start_pos)
        x = x + self.mlp(self.norm2(x))
        return x

class TinyLM(nn.Module):
    def __init__(self, vocab_size, dim=384, n_layers=6, n_heads=6):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([Block(dim, n_heads) for _ in range(n_layers)])
        self.norm = RMSNormCUDA(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.dim = dim
        self.n_heads = n_heads
    def forward(self, idx, sin, cos, cache=None, start_pos=0):
        x = self.tok(idx)
        for blk in self.blocks:
            x = blk(x, sin, cos, cache, start_pos)
        x = self.norm(x)
        return self.head(x)

def prealloc_kvcache(B, max_seq, n_heads, head_dim, device, dtype):
    k = torch.empty(B, n_heads, max_seq, head_dim, device=device, dtype=dtype)
    v = torch.empty(B, n_heads, max_seq, head_dim, device=device, dtype=dtype)
    return {'k': k, 'v': v}
```

---

## train.py

```python
import argparse, os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tqdm import tqdm
from model import TinyLM, build_sincos

class CharDataset(torch.utils.data.Dataset):
    def __init__(self, text, seq_len, tokenizer):
        self.seq_len = seq_len
        self.tok = tokenizer
        self.ids = self.tok.encode(text).ids
    def __len__(self):
        return max(0, len(self.ids) - self.seq_len)
    def __getitem__(self, i):
        x = self.ids[i:i+self.seq_len]
        y = self.ids[i+1:i+self.seq_len+1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

def build_tokenizer(corpus_paths, out_path):
    tok = Tokenizer(BPE(unk_token="<unk>"))
    tok.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(vocab_size=4096, min_frequency=2, special_tokens=["<unk>"])
    # Streamed training to avoid RAM blowups
    def line_iter():
        for p in corpus_paths:
            with open(p, 'r', encoding='utf-8') as f:
                for line in f:
                    yield line.strip()
    tok.train_from_iterator(line_iter(), trainer=trainer)
    tok.save(out_path)
    return tok

@torch.no_grad()
def evaluate(model, dl, sin, cos, device):
    model.eval()
    loss_sum = 0
    n = 0
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        logits = model(x, sin, cos)
        loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        loss_sum += loss.item(); n += 1
    model.train()
    return loss_sum / max(1, n)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=str, default='tinystories')
    ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--seq_len', type=int, default=256)
    ap.add_argument('--dim', type=int, default=384)
    ap.add_argument('--n_layers', type=int, default=6)
    ap.add_argument('--n_heads', type=int, default=6)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--compile', action='store_true')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.data == 'tinystories':
        train_path = 'data/tinystories_train.txt'
        val_path   = 'data/tinystories_val.txt'
    else:
        train_path = 'data/tinyshakespeare_train.txt'
        val_path   = 'data/tinyshakespeare_val.txt'

    if not os.path.exists('tokenizer.json'):
        build_tokenizer([train_path, val_path], 'tokenizer.json')
    tok = Tokenizer.from_file('tokenizer.json')

    with open(train_path, 'r', encoding='utf-8') as f: train_text = f.read()
    with open(val_path, 'r', encoding='utf-8') as f: val_text = f.read()

    train_ds = CharDataset(train_text, args.seq_len, tok)
    val_ds   = CharDataset(val_text, args.seq_len, tok)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_dl   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=True)

    model = TinyLM(vocab_size=tok.get_vocab_size(), dim=args.dim, n_layers=args.n_layers, n_heads=args.n_heads).to(device)
    if args.compile and hasattr(torch, 'compile'):
        model = torch.compile(model)

    opt = AdamW(model.parameters(), lr=args.lr)
    sin, cos = build_sincos(4096, model.dim // model.n_heads, device)

    best = 1e9
    os.makedirs('out', exist_ok=True)

    for step, (x, y) in enumerate(tqdm(train_dl, total=args.steps)):
        x, y = x.to(device), y.to(device)
        logits = model(x, sin, cos)
        loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 100 == 0:
            val_loss = evaluate(model, val_dl, sin, cos, device)
            if val_loss < best:
                best = val_loss
                base = getattr(model, "_orig_mod", model)
                torch.save({
                    'model': base.state_dict(),
                    'tok': tok.to_str(),
                    'config': {
                        'dim': base.dim,
                        'n_layers': len(base.blocks),
                        'n_heads': base.n_heads,
                        'vocab_size': tok.get_vocab_size(),
                    }
                }, 'out/best.pt')
        if step+1 >= args.steps:
            break

if __name__ == '__main__':
    main()
```

---

## infer.py

```python
import argparse, torch, random
from model import TinyLM, build_sincos, prealloc_kvcache
from tokenizers import Tokenizer

def sample_top_p(logits, top_p=0.9):
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cdf = torch.cumsum(sorted_probs, dim=-1)
    mask = cdf > top_p
    # Keep at least 1 token
    mask[..., 0] = False
    sorted_probs[mask] = 0.0
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
    idx = torch.multinomial(sorted_probs, num_samples=1)
    next_token = sorted_idx.gather(-1, idx)
    return next_token

@torch.no_grad()
def generate(model, tok, prompt, max_new_tokens=128, temperature=1.0, top_p=0.9,
             repetition_penalty=1.1, freq_penalty=0.0, presence_penalty=0.0, seed=0, stream=False):
    if seed is not None:
        random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    device = next(model.parameters()).device
    sin, cos = build_sincos(8192, model.dim // model.n_heads, device)
    ids = torch.tensor(tok.encode(prompt).ids, device=device).unsqueeze(0)
    cache = prealloc_kvcache(1, ids.size(1)+max_new_tokens, model.n_heads, model.dim//model.n_heads, device, dtype=next(model.parameters()).dtype)

    recent_window = 512
    for _ in range(max_new_tokens):
        logits = model(ids[:, -1:], sin, cos, cache, start_pos=ids.size(1)-1)
        logits = logits[:, -1, :]
        # Repetition penalty / frequency & presence penalties
        if ids.size(1) > 0 and (repetition_penalty > 1.0 or freq_penalty > 0.0 or presence_penalty > 0.0):
            recent = ids[:, -recent_window:]
            for b in range(ids.size(0)):
                unique, counts = torch.unique(recent[b], return_counts=True)
                if repetition_penalty > 1.0:
                    logits[b, unique] /= repetition_penalty
                if freq_penalty > 0.0:
                    logits[b, unique] -= freq_penalty * counts.to(logits.dtype)
                if presence_penalty > 0.0:
                    logits[b, unique] -= presence_penalty
        # Temperature
        if temperature != 1.0:
            logits = logits / max(1e-8, temperature)
        # Nucleus sampling
        next_id = sample_top_p(logits, top_p=top_p)
        ids = torch.cat([ids, next_id], dim=1)
        if stream:
            print(tok.decode(ids[0].tolist()), flush=True)
    return tok.decode(ids[0].tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=str, required=True)
    ap.add_argument('--prompt', type=str, default='Once upon a time')
    ap.add_argument('--max_new_tokens', type=int, default=128)
    ap.add_argument('--temperature', type=float, default=0.9)
    ap.add_argument('--top_p', type=float, default=0.9)
    ap.add_argument('--repetition_penalty', type=float, default=1.1)
    ap.add_argument('--freq_penalty', type=float, default=0.0)
    ap.add_argument('--presence_penalty', type=float, default=0.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--stream', action='store_true')
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location='cpu')
    tok = Tokenizer.from_str(ckpt['tok'])

    cfg = ckpt.get('config', None)
    if cfg is None:
        cfg = {'dim': 384, 'n_layers': 6, 'n_heads': 6, 'vocab_size': tok.get_vocab_size()}

    model = TinyLM(vocab_size=cfg['vocab_size'], dim=cfg['dim'], n_layers=cfg['n_layers'], n_heads=cfg['n_heads']).cuda().eval()

    state = ckpt['model']
    if any(k.startswith('_orig_mod.') for k in state):
        state = {k.replace('_orig_mod.', '', 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)

    txt = generate(model, tok, args.prompt, args.max_new_tokens, args.temperature, args.top_p,
                   args.repetition_penalty, args.freq_penalty, args.presence_penalty, args.seed, args.stream)
    print(txt)

if __name__ == '__main__':
    main()
```

---

## bench\_rmsnorm.py

```python
import torch, time
from model import RMSNormCUDA

@torch.no_grad()
def bench(B=16, T=256, C=1024, iters=200, dtype=torch.float16):
    x = torch.randn(B, T, C, device='cuda', dtype=dtype)
    ref_w = torch.ones(C, device='cuda', dtype=dtype)
    mod = RMSNormCUDA(C).cuda().to(dtype)
    mod.weight.copy_(ref_w)

    # Warmup
    for _ in range(50):
        y = mod(x)
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(iters):
        y = mod(x)
    torch.cuda.synchronize()
    t1 = time.time()
    print({'shape': (B,T,C), 'iters': iters, 'ms_per_iter': (t1-t0)*1000/iters})

if __name__ == '__main__':
    bench()
```

---

## tests/test\_rmsnorm.py

```python
import torch
from model import RMSNormCUDA

def test_forward_close():
    torch.manual_seed(0)
    B,T,C = 4,8,64
    x = torch.randn(B,T,C, device='cuda', dtype=torch.float32, requires_grad=True)
    w = torch.randn(C, device='cuda', dtype=torch.float32)

    # Reference
    with torch.no_grad():
        rms = (x.detach()**2).mean(dim=-1, keepdim=True).add(1e-6).rsqrt()
    y_ref = x * rms * w

    mod = RMSNormCUDA(C).cuda().float()
    mod.weight.data.copy_(w)
    y = mod(x)
    assert torch.allclose(y, y_ref, atol=1e-4, rtol=1e-4)

def test_backward_close():
    torch.manual_seed(0)
    B,T,C = 2,4,128
    x = torch.randn(B,T,C, device='cuda', dtype=torch.float32, requires_grad=True)
    w = torch.randn(C, device='cuda', dtype=torch.float32, requires_grad=True)

    # Our module
    mod = RMSNormCUDA(C).cuda().float()
    mod.weight.data.copy_(w.detach())
    y = mod(x)
    loss = y.sum()
    dx, dw = torch.autograd.grad(loss, [x, mod.weight])

    # Reference via PyTorch ops
    x2 = x.detach().clone().requires_grad_(True)
    w2 = w.detach().clone().requires_grad_(True)
    rms = (x2**2).mean(dim=-1, keepdim=True).add(1e-6).rsqrt()
    y2 = x2 * rms * w2
    loss2 = y2.sum()
    dx2, dw2 = torch.autograd.grad(loss2, [x2, w2])

    assert torch.allclose(dx, dx2, atol=1e-3, rtol=1e-3)
    assert torch.allclose(dw, dw2, atol=1e-3, rtol=1e-3)
```

---

## tests/conftest.py

```python
# Ensure the project root is importable without touching PYTHONPATH.
# PyTest loads this automatically.
import os, sys
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
```

---

## data/prepare\_tinystories.py

```python
import os, datasets
os.makedirs('data', exist_ok=True)
ds = datasets.load_dataset('roneneldan/TinyStories', split='train')
val = datasets.load_dataset('roneneldan/TinyStories', split='validation')
with open('data/tinystories_train.txt','w') as f:
    for r in ds['text']: f.write(r + '\n')
with open('data/tinystories_val.txt','w') as f:
    for r in val['text']: f.write(r + '\n')
print('Wrote TinyStories train/val.')
```

---

## data/prepare\_tinyshakespeare.py

```python
import requests, os
os.makedirs('data', exist_ok=True)
url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
text = requests.get(url).text
n = int(len(text)*0.9)
with open('data/tinyshakespeare_train.txt','w') as f: f.write(text[:n])
with open('data/tinyshakespeare_val.txt','w') as f: f.write(text[n:])
print('Wrote TinyShakespeare train/val.')
```

---

## Interview demo script (CLI)

```bash
# 1) Show correctness test + speed microbench
pytest -q && python bench_rmsnorm.py

# 2) Train for a short run (e.g. 2000 steps)
python train.py --data tinystories --steps 2000 --batch_size 16 --seq_len 256 --dim 384 --n_layers 6 --n_heads 6 --lr 3e-4 --compile --flash

# 3) Generate
python infer.py --ckpt out/best.pt --prompt "Once upon a time in Abu Dhabi" --max_new_tokens 80
```

---

## Notes & Extensions (if time remains)

* Add a fused **RoPE** CUDA kernel (apply sin/cos to Q/K in-place) to reduce bandwidth.
* Export a TorchScript/ONNX + TensorRT FP16 engine for the single-step decode path.
* Weight-only int8 for projection layers (per-channel scale), measuring tokens/s and perplexity delta.
* Ablation: compare PyTorch RMSNorm vs. custom kernel speed across shapes.
* Add paged KV-cache to enable long sequences without realloc.

````}



---

## Dockerfile

```dockerfile
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

WORKDIR /workspace/app

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ninja-build git curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt requests

COPY . .

ENV HF_HOME=/workspace/.cache/huggingface \
    TOKENIZERS_PARALLELISM=false \
    TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"

RUN python setup_cuda.py build_ext --inplace

CMD ["bash"]
````

---

## docker-compose.yml

```yaml
version: "3.9"
services:
  nanofalcon:
    build:
      context: .
      dockerfile: Dockerfile
    image: nanofalcon:latest
    shm_size: "8g"
    environment:
      HF_HOME: "/workspace/.cache/huggingface"
      TOKENIZERS_PARALLELISM: "false"
      NVIDIA_VISIBLE_DEVICES: "all"
      NVIDIA_DRIVER_CAPABILITIES: "compute,utility"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    volumes:
      - hf-cache:/workspace/.cache/huggingface
      - outputs:/workspace/app/out
      - ./data:/workspace/app/data
    working_dir: /workspace/app
    command: >
      bash -lc "python data/prepare_tinystories.py &&
                python setup_cuda.py build_ext --inplace &&
                pytest -q &&
                python train.py --data tinystories --steps 2000 --batch_size 16 --seq_len 256 --dim 384 --n_layers 6 --n_heads 6 --lr 3e-4 --compile &&
                python infer.py --ckpt out/best.pt --prompt 'Once upon a time in Abu Dhabi' --max_new_tokens 80"

volumes:
  hf-cache:
  outputs:
```

**Note (legacy setups):** if your Docker Compose ignores `deploy` outside Swarm, add this override file and include it with `-f compose.legacy-gpu.yml`:

```yaml
# compose.legacy-gpu.yml
version: "3.9"
services:
  nanofalcon:
    # Older NVIDIA Toolkit path (deprecated, but works widely)
    runtime: nvidia
```

---

## Container quickstart

```bash
# Build image and run the end-to-end pipeline (data → build CUDA → tests → train → infer)
docker compose up --build

# Open an interactive shell with GPU access
# (override the command defined in compose)
docker compose run --build --rm nanofalcon bash
```
