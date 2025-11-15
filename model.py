"""
TinyLM: A minimal GPT-style transformer implementation with custom CUDA RMSNorm.

This module implements a small-scale language model with the following key components:
- Custom CUDA-accelerated RMSNorm (Root Mean Square Layer Normalization)
- Rotary Position Embeddings (RoPE) for positional encoding
- Multi-Head Attention with KV-cache support for efficient inference
- Standard transformer blocks with pre-normalization

References:
    RMSNorm: Zhang & Sennrich (2019) "Root Mean Square Layer Normalization"
    RoPE: Su et al. (2024) "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    GPT Architecture: Radford et al. (2019) "Language Models are Unsupervised Multitask Learners"
"""

import math
from typing import Optional, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

import rmsnorm_cuda


class RMSNormCUDAFn(torch.autograd.Function):
    """Custom autograd Function for CUDA-accelerated RMSNorm.

    This implements the forward and backward passes using a fused CUDA kernel
    for improved performance compared to native PyTorch operations.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        """Forward pass of RMSNorm.

        Args:
            ctx: Autograd context for saving tensors for backward pass
            x: Input tensor of shape [batch_size, seq_len, hidden_dim]
            weight: Learnable scale parameters of shape [hidden_dim]
            eps: Small constant for numerical stability

        Returns:
            Normalized tensor of same shape as input
        """
        y, inv_rms = rmsnorm_cuda.forward(x, weight, eps)
        ctx.save_for_backward(x, weight, inv_rms)
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, None]:
        """Backward pass of RMSNorm.

        Args:
            ctx: Autograd context with saved tensors
            dy: Gradient w.r.t output

        Returns:
            Tuple of (dx, dweight, deps) where deps is None (non-differentiable)
        """
        x, weight, inv_rms = ctx.saved_tensors
        dx, dw = rmsnorm_cuda.backward(dy.contiguous(), x, weight, inv_rms, ctx.eps)
        return dx, dw, None


class RMSNormCUDA(nn.Module):
    """CUDA-accelerated Root Mean Square Layer Normalization.

    RMSNorm is a simplification of LayerNorm that normalizes by RMS statistics
    without mean centering, reducing computational cost while maintaining
    comparable performance.

    Attributes:
        weight: Learnable scale parameters
        eps: Small constant for numerical stability (default: 1e-6)
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        """Initialize RMSNorm layer.

        Args:
            dim: Dimension to normalize over
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMSNorm to input tensor.

        Args:
            x: Input tensor of shape [..., dim]

        Returns:
            Normalized tensor of same shape
        """
        return RMSNormCUDAFn.apply(x, self.weight, self.eps)


def rotary_embeddings(
    x: torch.Tensor,
    sin: torch.Tensor,
    cos: torch.Tensor
) -> torch.Tensor:
    """Apply Rotary Position Embeddings (RoPE) to input tensor.

    RoPE encodes positional information by rotating pairs of dimensions
    in the feature space, providing better extrapolation to longer sequences
    than traditional position embeddings.

    Args:
        x: Input tensor of shape [batch, heads, seq_len, head_dim]
        sin: Sine values for rotation of shape [1, 1, seq_len, head_dim]
        cos: Cosine values for rotation of shape [1, 1, seq_len, head_dim]

    Returns:
        Tensor with rotary embeddings applied, same shape as input
    """
    x1, x2 = x[..., ::2], x[..., 1::2]
    xr = torch.stack((-x2, x1), dim=-1).reshape_as(x)
    return x * cos + xr * sin


def build_sincos(
    seq_len: int,
    dim: int,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute sine and cosine values for Rotary Position Embeddings.

    Args:
        seq_len: Maximum sequence length to compute embeddings for
        dim: Dimension of each attention head
        device: Device to create tensors on

    Returns:
        Tuple of (sin, cos) tensors, each of shape [1, 1, seq_len, dim]
    """
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.einsum('t,f->tf', t, inv_freq)
    sin = torch.sin(torch.cat([freqs, freqs], dim=-1))[None, None, :, :]
    cos = torch.cos(torch.cat([freqs, freqs], dim=-1))[None, None, :, :]
    return sin, cos


class MHA(nn.Module):
    """Multi-Head Attention with RoPE and KV-cache support.

    Implements scaled dot-product attention with:
    - Rotary position embeddings for positional encoding
    - KV-cache for efficient autoregressive generation
    - No bias terms (following modern LLM practices)

    Attributes:
        nh: Number of attention heads
        dim: Model dimension
        qkv: Linear projection for queries, keys, values
        proj: Output projection
    """

    def __init__(self, dim: int, n_heads: int):
        """Initialize Multi-Head Attention layer.

        Args:
            dim: Model dimension (must be divisible by n_heads)
            n_heads: Number of attention heads
        """
        super().__init__()
        assert dim % n_heads == 0, f"dim {dim} must be divisible by n_heads {n_heads}"
        self.nh = n_heads
        self.dim = dim
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        sin: torch.Tensor,
        cos: torch.Tensor,
        cache: Optional[Dict[str, torch.Tensor]] = None,
        start_pos: int = 0
    ) -> torch.Tensor:
        """Forward pass of multi-head attention.

        Args:
            x: Input tensor of shape [batch_size, seq_len, dim]
            sin: Sine values for RoPE
            cos: Cosine values for RoPE
            cache: Optional KV-cache dict with 'k' and 'v' tensors
            start_pos: Starting position for KV-cache updates

        Returns:
            Output tensor of shape [batch_size, seq_len, dim]
        """
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for multi-head attention
        q = q.view(B, T, self.nh, C // self.nh).transpose(1, 2)
        k = k.view(B, T, self.nh, C // self.nh).transpose(1, 2)
        v = v.view(B, T, self.nh, C // self.nh).transpose(1, 2)

        # Apply rotary embeddings
        q = rotary_embeddings(q, sin[:, :, start_pos:start_pos+T, :],
                            cos[:, :, start_pos:start_pos+T, :])
        k = rotary_embeddings(k, sin[:, :, start_pos:start_pos+T, :],
                            cos[:, :, start_pos:start_pos+T, :])

        # Update and use KV-cache if provided
        if cache is not None:
            cache['k'][:, :, start_pos:start_pos+T] = k
            cache['v'][:, :, start_pos:start_pos+T] = v
            k = cache['k'][:, :, :start_pos+T]
            v = cache['v'][:, :, :start_pos+T]

        # Scaled dot-product attention with causal mask
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # Reshape and project output
        y = attn.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class Block(nn.Module):
    """Transformer block with pre-normalization and SiLU activation.

    Architecture:
        x -> LayerNorm -> Multi-Head Attention -> Residual ->
        x -> LayerNorm -> MLP (Linear -> SiLU -> Linear) -> Residual

    Attributes:
        norm1: RMSNorm for attention input
        attn: Multi-head attention layer
        norm2: RMSNorm for MLP input
        mlp: Feed-forward network with SiLU activation
    """

    def __init__(self, dim: int, n_heads: int, mlp_ratio: int = 4):
        """Initialize transformer block.

        Args:
            dim: Model dimension
            n_heads: Number of attention heads
            mlp_ratio: MLP hidden dimension ratio (hidden_dim = dim * mlp_ratio)
        """
        super().__init__()
        self.norm1 = RMSNormCUDA(dim)
        self.attn = MHA(dim, n_heads)
        self.norm2 = RMSNormCUDA(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio*dim, bias=False),
            nn.SiLU(),
            nn.Linear(mlp_ratio*dim, dim, bias=False),
        )

    def forward(
        self,
        x: torch.Tensor,
        sin: torch.Tensor,
        cos: torch.Tensor,
        cache: Optional[Dict[str, torch.Tensor]] = None,
        start_pos: int = 0
    ) -> torch.Tensor:
        """Forward pass of transformer block.

        Args:
            x: Input tensor of shape [batch_size, seq_len, dim]
            sin: Sine values for RoPE
            cos: Cosine values for RoPE
            cache: Optional KV-cache dict
            start_pos: Starting position for KV-cache updates

        Returns:
            Output tensor of shape [batch_size, seq_len, dim]
        """
        x = x + self.attn(self.norm1(x), sin, cos, cache, start_pos)
        x = x + self.mlp(self.norm2(x))
        return x


class TinyLM(nn.Module):
    """Small-scale GPT-style language model with custom RMSNorm.

    Architecture:
        Token Embedding -> N x Transformer Blocks -> RMSNorm -> Output Projection

    Features:
        - Custom CUDA RMSNorm for improved performance
        - Rotary Position Embeddings (no learned position embeddings)
        - KV-cache support for efficient generation
        - No bias terms in linear layers

    Attributes:
        tok: Token embedding layer
        blocks: List of transformer blocks
        norm: Final RMSNorm before output
        head: Output projection to vocabulary
        dim: Model dimension
        n_heads: Number of attention heads
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int = 384,
        n_layers: int = 6,
        n_heads: int = 6
    ):
        """Initialize TinyLM model.

        Args:
            vocab_size: Size of vocabulary
            dim: Model dimension (default: 384)
            n_layers: Number of transformer blocks (default: 6)
            n_heads: Number of attention heads (default: 6)
        """
        super().__init__()
        self.tok = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([Block(dim, n_heads) for _ in range(n_layers)])
        self.norm = RMSNormCUDA(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.dim = dim
        self.n_heads = n_heads

    def forward(
        self,
        idx: torch.Tensor,
        sin: torch.Tensor,
        cos: torch.Tensor,
        cache: Optional[Dict[str, torch.Tensor]] = None,
        start_pos: int = 0
    ) -> torch.Tensor:
        """Forward pass of TinyLM.

        Args:
            idx: Token indices of shape [batch_size, seq_len]
            sin: Sine values for RoPE
            cos: Cosine values for RoPE
            cache: Optional KV-cache dict for incremental decoding
            start_pos: Starting position for KV-cache updates

        Returns:
            Logits tensor of shape [batch_size, seq_len, vocab_size]
        """
        x = self.tok(idx)
        for blk in self.blocks:
            x = blk(x, sin, cos, cache, start_pos)
        x = self.norm(x)
        return self.head(x)


def prealloc_kvcache(
    B: int,
    max_seq: int,
    n_heads: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype
) -> Dict[str, torch.Tensor]:
    """Pre-allocate KV-cache tensors for efficient autoregressive generation.

    The KV-cache stores key and value tensors from previous timesteps to avoid
    recomputation during incremental decoding, reducing quadratic complexity
    to linear for generation.

    Args:
        B: Batch size
        max_seq: Maximum sequence length to cache
        n_heads: Number of attention heads
        head_dim: Dimension of each attention head
        device: Device to allocate tensors on
        dtype: Data type for tensors (e.g., float16 for memory efficiency)

    Returns:
        Dictionary with 'k' and 'v' keys containing pre-allocated tensors
        of shape [batch_size, n_heads, max_seq, head_dim]
    """
    k = torch.empty(B, n_heads, max_seq, head_dim, device=device, dtype=dtype)
    v = torch.empty(B, n_heads, max_seq, head_dim, device=device, dtype=dtype)
    return {'k': k, 'v': v}