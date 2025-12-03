"""Transformer model for TinyLM."""

from typing import Optional

import torch
import torch.nn as nn

from tinylm.architectures import ArchitectureConfig, get_architecture
from tinylm.components import (
    PositionalContext,
    build_norm,
    build_pos_emb,
)
from tinylm.inference.cache_manager import CacheManager, StandardCache
from tinylm.model.blocks import build_block
from tinylm.quant import QuantConfig, make_linear


class TinyLM(nn.Module):
    """Multi-architecture language model.

    Supports different architectures by composing building blocks:
    - LLaMA: RMSNorm (pre), RoPE, SiLU, Gated MLP
    - GPT: LayerNorm (post), Learned pos emb, GELU, Standard MLP

    Features:
        - Configurable architecture via ArchitectureConfig
        - Internal positional embedding handling (clean API)
        - KV-cache support for efficient generation
        - Optional quantization
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int = 384,
        n_layers: int = 6,
        n_heads: int = 6,
        max_seq_len: int = 4096,
        dropout: float = 0.0,
        architecture: str = "llama",
        arch_config: Optional[ArchitectureConfig] = None,
        quant_config: Optional[QuantConfig] = None,
    ):
        super().__init__()

        # Get architecture config
        if arch_config is not None:
            self.arch = arch_config
        else:
            self.arch = get_architecture(architecture)

        # Store model configuration
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        self.dropout_rate = dropout
        self.quant_config = quant_config

        # Derived values
        self.head_dim = dim // n_heads

        # Token embeddings
        self.tok = nn.Embedding(vocab_size, dim)
        self.tok_dropout = nn.Dropout(dropout)

        # Positional embeddings
        pos_dim = self.head_dim if self.arch.pos_emb_type == "rope" else dim
        self.pos_emb = build_pos_emb(
            self.arch.pos_emb_type,
            dim=pos_dim,
            max_seq_len=max_seq_len,
            base=self.arch.rope_base,
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            build_block(
                norm_position=self.arch.norm_position,
                dim=dim,
                n_heads=n_heads,
                n_kv_heads=self.arch.n_kv_heads,
                norm_type=self.arch.norm_type,
                attention_type=self.arch.attention_type,
                mlp_type=self.arch.mlp_type,
                activation=self.arch.activation,
                mlp_ratio=self.arch.mlp_ratio,
                dropout=dropout,
                bias=self.arch.use_bias,
                norm_eps=self.arch.norm_eps,
                quant_config=quant_config,
                pos_emb=self.pos_emb if self.arch.pos_emb_type == "rope" else None,
            )
            for _ in range(n_layers)
        ])

        # Final normalization
        self.norm = build_norm(
            self.arch.norm_type,
            dim,
            eps=self.arch.norm_eps,
        )

        # Output head
        self.head = make_linear(
            dim, vocab_size, bias=False,
            quant_config=quant_config, layer_type="head"
        )

        # Precompute positional info and register as buffer
        self._init_pos_cache()

    def _init_pos_cache(self):
        """Initialize positional embedding cache."""
        pos_cache = self.pos_emb.precompute(self.max_seq_len, torch.device('cpu'))
        for key, tensor in pos_cache.items():
            self.register_buffer(f'_pos_{key}', tensor, persistent=False)

    def _get_pos_ctx(
        self,
        seq_len: int,
        start_pos: int,
        device: torch.device,
    ) -> PositionalContext:
        """Build positional context for forward pass."""
        ctx = PositionalContext(
            seq_len=seq_len,
            start_pos=start_pos,
            device=device,
        )

        # Add cached positional values
        if hasattr(self, '_pos_sin'):
            ctx.sin = self._pos_sin.to(device)
            ctx.cos = self._pos_cos.to(device)
        if hasattr(self, '_pos_positions'):
            ctx.positions = self._pos_positions.to(device)

        return ctx

    def forward(
        self,
        idx: torch.Tensor,
        cache: Optional[CacheManager] = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            idx: Token indices [batch, seq_len]
            cache: Optional CacheManager for KV caching
            start_pos: Starting position for generation

        Returns:
            Logits [batch, seq_len, vocab_size]
        """
        B, T = idx.shape

        # Token embeddings
        x = self.tok(idx)
        x = self.tok_dropout(x)

        # Add learned positional embeddings if applicable
        if self.arch.pos_emb_type == "learned":
            pos_ctx = self._get_pos_ctx(T, start_pos, idx.device)
            x = x + self.pos_emb.apply(x, pos_ctx)

        # Build positional context for attention
        pos_ctx = self._get_pos_ctx(T, start_pos, idx.device)

        # Forward through blocks
        for i, block in enumerate(self.blocks):
            x = block(x, pos_ctx, cache=cache, layer_idx=i, start_pos=start_pos)

        # Final norm and head
        x = self.norm(x)
        return self.head(x)

    def create_kv_cache(
        self,
        batch_size: int,
        max_seq_len: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> CacheManager:
        """Create KV cache for generation.

        Args:
            batch_size: Batch size
            max_seq_len: Maximum sequence length (defaults to self.max_seq_len)
            device: Device for cache tensors
            dtype: Data type for cache tensors

        Returns:
            CacheManager instance (StandardCache)
        """
        if max_seq_len is None:
            max_seq_len = self.max_seq_len
        if device is None:
            device = next(self.parameters()).device
        if dtype is None:
            dtype = next(self.parameters()).dtype

        cache = StandardCache(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            head_dim=self.head_dim,
            device=device,
            dtype=dtype,
        )
        cache.allocate(batch_size, max_seq_len)
        return cache

    def get_num_params(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())


__all__ = ["TinyLM"]
