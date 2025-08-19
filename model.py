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