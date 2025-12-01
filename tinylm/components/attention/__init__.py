"""Attention components for TinyLM."""

from typing import Optional
import torch.nn as nn

from tinylm.components.registry import ATTENTION_REGISTRY
from tinylm.quant import QuantConfig

# Import to register with registry
from tinylm.components.attention.mha import MHA, make_attention


def build_attention(
    attention_type: str,
    dim: int,
    n_heads: int,
    n_kv_heads: Optional[int] = None,
    dropout: float = 0.0,
    bias: bool = False,
    quant_config: Optional[QuantConfig] = None,
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
    )


__all__ = [
    "MHA",
    "make_attention",
    "build_attention",
    "ATTENTION_REGISTRY",
]
