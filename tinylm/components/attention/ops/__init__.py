"""Attention operation implementations.

Provides swappable attention computation backends:
- standard: PyTorch's scaled_dot_product_attention (auto-selects best kernel)
- flash: Explicit Flash Attention (via SDPA with math kernel disabled)
- memory_efficient: Memory-efficient attention (via SDPA settings)

Example:
    >>> from tinylm.components.attention.ops import build_attention_op
    >>> attn_op = build_attention_op("standard")
    >>> output = attn_op(q, k, v, is_causal=True)
"""

from typing import Optional

from tinylm.components.attention.ops.base import AttentionOp
from tinylm.components.attention.ops.flash import FlashAttentionOp, MemoryEfficientAttentionOp
from tinylm.components.attention.ops.standard import StandardAttentionOp
from tinylm.components.registry import ATTENTION_OP_REGISTRY


def build_attention_op(
    op_type: str = "standard",
    dropout: float = 0.0,
    scale: Optional[float] = None,
) -> AttentionOp:
    """Factory function to build attention operation.

    Args:
        op_type: Type of attention operation ("standard", "flash", "memory_efficient")
        dropout: Dropout probability for attention weights
        scale: Optional scale factor for attention (default: 1/sqrt(head_dim))

    Returns:
        AttentionOp instance
    """
    return ATTENTION_OP_REGISTRY.build(
        op_type,
        dropout=dropout,
        scale=scale,
    )


def available_attention_ops() -> list:
    """List available attention operation types."""
    return ATTENTION_OP_REGISTRY.available()


__all__ = [
    "AttentionOp",
    "StandardAttentionOp",
    "FlashAttentionOp",
    "MemoryEfficientAttentionOp",
    "build_attention_op",
    "available_attention_ops",
    "ATTENTION_OP_REGISTRY",
]
