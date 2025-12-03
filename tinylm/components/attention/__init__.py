"""Attention components for TinyLM.

This module provides:
- MHA: Multi-Head Attention module (orchestrates projection, position, cache, attention)
- AttentionOp: Composable attention computation backends (standard, flash, memory_efficient)

Example:
    >>> from tinylm.components.attention import MHA, build_attention_op
    >>>
    >>> # Create MHA with default standard attention
    >>> mha = MHA(dim=512, n_heads=8)
    >>>
    >>> # Create MHA with flash attention
    >>> mha = MHA(dim=512, n_heads=8, attention_op="flash")
    >>>
    >>> # Swap attention op at runtime
    >>> mha.attention_op = build_attention_op("memory_efficient")
"""

from typing import Optional
import torch.nn as nn

from tinylm.components.registry import ATTENTION_REGISTRY, ATTENTION_OP_REGISTRY
from tinylm.quant import QuantConfig

# Import attention module to register with registry
from tinylm.components.attention.mha import MHA, make_attention

# Import attention ops
from tinylm.components.attention.ops import (
    AttentionOp,
    StandardAttentionOp,
    FlashAttentionOp,
    MemoryEfficientAttentionOp,
    build_attention_op,
    available_attention_ops,
)


def build_attention(
    attention_type: str,
    dim: int,
    n_heads: int,
    n_kv_heads: Optional[int] = None,
    dropout: float = 0.0,
    bias: bool = False,
    quant_config: Optional[QuantConfig] = None,
    attention_op: str = "standard",
) -> nn.Module:
    """Factory function to build attention module.

    Args:
        attention_type: Type of attention ("mha", "mqa", "gqa")
        dim: Model dimension
        n_heads: Number of query heads
        n_kv_heads: Number of KV heads (for MQA/GQA, None = same as n_heads)
        dropout: Dropout rate
        bias: Whether to use bias
        quant_config: Quantization configuration
        attention_op: Attention operation backend ("standard", "flash", "memory_efficient")

    Returns:
        Attention module
    """
    return ATTENTION_REGISTRY.build(
        attention_type,
        dim=dim,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        dropout=dropout,
        bias=bias,
        quant_config=quant_config,
        attention_op=attention_op,
    )


__all__ = [
    # Attention modules
    "MHA",
    "make_attention",
    "build_attention",
    "ATTENTION_REGISTRY",
    # Attention ops
    "AttentionOp",
    "StandardAttentionOp",
    "FlashAttentionOp",
    "MemoryEfficientAttentionOp",
    "build_attention_op",
    "available_attention_ops",
    "ATTENTION_OP_REGISTRY",
]
