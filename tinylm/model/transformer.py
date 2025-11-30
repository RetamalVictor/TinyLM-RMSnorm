"""Transformer model components for TinyLM."""

from typing import Optional, Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from tinylm.model.normalization import RMSNormCUDA
from tinylm.model.rope import rotary_embeddings
from tinylm.quant import QuantConfig, make_linear


class MHA(nn.Module):
    """Multi-Head Attention with RoPE and KV-cache support."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        dropout: float = 0.0,
        quant_config: Optional[QuantConfig] = None
    ):
        super().__init__()
        assert dim % n_heads == 0, f"dim {dim} must be divisible by n_heads {n_heads}"
        self.nh = n_heads
        self.dim = dim
        self.qkv = make_linear(dim, dim * 3, bias=False, quant_config=quant_config, layer_type="attention")
        self.proj = make_linear(dim, dim, bias=False, quant_config=quant_config, layer_type="attention")
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        sin: torch.Tensor,
        cos: torch.Tensor,
        cache: Optional[Dict[str, torch.Tensor]] = None,
        start_pos: int = 0
    ) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.nh, C // self.nh).transpose(1, 2)
        k = k.view(B, T, self.nh, C // self.nh).transpose(1, 2)
        v = v.view(B, T, self.nh, C // self.nh).transpose(1, 2)

        q = rotary_embeddings(q, sin[:, :, start_pos:start_pos+T, :],
                            cos[:, :, start_pos:start_pos+T, :])
        k = rotary_embeddings(k, sin[:, :, start_pos:start_pos+T, :],
                            cos[:, :, start_pos:start_pos+T, :])

        if cache is not None:
            cache['k'][:, :, start_pos:start_pos+T] = k
            cache['v'][:, :, start_pos:start_pos+T] = v
            k = cache['k'][:, :, :start_pos+T]
            v = cache['v'][:, :, :start_pos+T]

        # Only use causal mask for multi-token sequences (prefill)
        # Single-token generation (T=1) doesn't need masking - it attends to all cached positions
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=(T > 1))
        y = attn.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)
        return self.dropout(y)


class Block(nn.Module):
    """Transformer block with pre-normalization and SiLU activation."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        quant_config: Optional[QuantConfig] = None
    ):
        super().__init__()
        self.norm1 = RMSNormCUDA(dim)
        self.attn = MHA(dim, n_heads, dropout=dropout, quant_config=quant_config)
        self.norm2 = RMSNormCUDA(dim)
        self.mlp = nn.Sequential(
            make_linear(dim, mlp_ratio*dim, bias=False, quant_config=quant_config, layer_type="mlp"),
            nn.SiLU(),
            nn.Dropout(dropout),
            make_linear(mlp_ratio*dim, dim, bias=False, quant_config=quant_config, layer_type="mlp"),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x: torch.Tensor,
        sin: torch.Tensor,
        cos: torch.Tensor,
        cache: Optional[Dict[str, torch.Tensor]] = None,
        start_pos: int = 0
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), sin, cos, cache, start_pos)
        x = x + self.mlp(self.norm2(x))
        return x


class TinyLM(nn.Module):
    """Small-scale GPT-style language model with custom RMSNorm.

    Features:
        - Custom CUDA RMSNorm for improved performance
        - Rotary Position Embeddings (no learned position embeddings)
        - KV-cache support for efficient generation
        - Optional ternary quantization
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int = 384,
        n_layers: int = 6,
        n_heads: int = 6,
        dropout: float = 0.0,
        quant_config: Optional[QuantConfig] = None
    ):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, dim)
        self.tok_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            Block(dim, n_heads, dropout=dropout, quant_config=quant_config)
            for _ in range(n_layers)
        ])
        self.norm = RMSNormCUDA(dim)
        self.head = make_linear(dim, vocab_size, bias=False, quant_config=quant_config, layer_type="head")
        self.dim = dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout
        self.quant_config = quant_config

    def forward(
        self,
        idx: torch.Tensor,
        sin: torch.Tensor,
        cos: torch.Tensor,
        cache: Optional[List[Dict[str, torch.Tensor]]] = None,
        start_pos: int = 0
    ) -> torch.Tensor:
        x = self.tok(idx)
        x = self.tok_dropout(x)
        for i, blk in enumerate(self.blocks):
            layer_cache = cache[i] if cache is not None else None
            x = blk(x, sin, cos, layer_cache, start_pos)
        x = self.norm(x)
        return self.head(x)
