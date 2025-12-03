"""Transformer blocks for TinyLM."""

from typing import Optional, TYPE_CHECKING
import torch
import torch.nn as nn

from tinylm.components import (
    build_norm,
    build_attention,
    build_mlp,
    PositionalContext,
)
from tinylm.quant import QuantConfig

if TYPE_CHECKING:
    from tinylm.inference.cache_manager import CacheManager


class PreNormBlock(nn.Module):
    """Pre-normalization transformer block.

    Structure: x -> Norm -> Attn -> Add -> Norm -> MLP -> Add

    Used by: LLaMA, Mistral, Falcon
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        norm_type: str = "rmsnorm",
        attention_type: str = "mha",
        mlp_type: str = "gated",
        activation: str = "silu",
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        bias: bool = False,
        norm_eps: float = 1e-6,
        quant_config: Optional[QuantConfig] = None,
        pos_emb: Optional[nn.Module] = None,
    ):
        super().__init__()

        # Build components using factories
        self.norm1 = build_norm(norm_type, dim, eps=norm_eps)
        self.attn = build_attention(
            attention_type,
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            dropout=dropout,
            bias=bias,
            quant_config=quant_config,
        )
        self.norm2 = build_norm(norm_type, dim, eps=norm_eps)
        self.mlp = build_mlp(
            mlp_type,
            dim=dim,
            hidden_ratio=mlp_ratio,
            activation=activation,
            dropout=dropout,
            bias=bias,
            quant_config=quant_config,
        )

        # Store positional embedding reference
        self.pos_emb = pos_emb

    def forward(
        self,
        x: torch.Tensor,
        pos_ctx: PositionalContext,
        cache: Optional["CacheManager"] = None,
        layer_idx: int = 0,
        start_pos: int = 0,
    ) -> torch.Tensor:
        # Pre-norm: norm before attention and MLP
        x = x + self.attn(
            self.norm1(x),
            pos_ctx,
            cache=cache,
            layer_idx=layer_idx,
            start_pos=start_pos,
            pos_emb=self.pos_emb,
        )
        x = x + self.mlp(self.norm2(x))
        return x


class PostNormBlock(nn.Module):
    """Post-normalization transformer block.

    Structure: x -> Attn -> Add -> Norm -> MLP -> Add -> Norm

    Used by: GPT-2, original Transformer
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        norm_type: str = "layernorm",
        attention_type: str = "mha",
        mlp_type: str = "standard",
        activation: str = "gelu",
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        bias: bool = True,
        norm_eps: float = 1e-5,
        quant_config: Optional[QuantConfig] = None,
        pos_emb: Optional[nn.Module] = None,
    ):
        super().__init__()

        # Build components using factories
        self.attn = build_attention(
            attention_type,
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            dropout=dropout,
            bias=bias,
            quant_config=quant_config,
        )
        self.norm1 = build_norm(norm_type, dim, eps=norm_eps, bias=bias)
        self.mlp = build_mlp(
            mlp_type,
            dim=dim,
            hidden_ratio=mlp_ratio,
            activation=activation,
            dropout=dropout,
            bias=bias,
            quant_config=quant_config,
        )
        self.norm2 = build_norm(norm_type, dim, eps=norm_eps, bias=bias)

        # Store positional embedding reference
        self.pos_emb = pos_emb

    def forward(
        self,
        x: torch.Tensor,
        pos_ctx: PositionalContext,
        cache: Optional["CacheManager"] = None,
        layer_idx: int = 0,
        start_pos: int = 0,
    ) -> torch.Tensor:
        # Post-norm: norm after attention and MLP
        x = self.norm1(x + self.attn(
            x, pos_ctx,
            cache=cache,
            layer_idx=layer_idx,
            start_pos=start_pos,
            pos_emb=self.pos_emb
        ))
        x = self.norm2(x + self.mlp(x))
        return x


def build_block(
    norm_position: str,
    dim: int,
    n_heads: int,
    **kwargs,
) -> nn.Module:
    """Factory function to build transformer block.

    Args:
        norm_position: "pre" for PreNormBlock, "post" for PostNormBlock
        dim: Model dimension
        n_heads: Number of attention heads
        **kwargs: Additional block arguments

    Returns:
        Transformer block
    """
    if norm_position == "pre":
        return PreNormBlock(dim=dim, n_heads=n_heads, **kwargs)
    elif norm_position == "post":
        return PostNormBlock(dim=dim, n_heads=n_heads, **kwargs)
    else:
        raise ValueError(f"Unknown norm_position: {norm_position}")


__all__ = [
    "PreNormBlock",
    "PostNormBlock",
    "build_block",
]
