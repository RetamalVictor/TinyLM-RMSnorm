"""Normalization components for TinyLM."""

from tinylm.components.registry import NORM_REGISTRY
from tinylm.components.normalization.rmsnorm import RMSNorm
from tinylm.components.normalization.layernorm import LayerNorm


def build_norm(norm_type: str, dim: int, **kwargs) -> "nn.Module":
    """Factory function to build normalization layer.

    Args:
        norm_type: Type of normalization ("rmsnorm" or "layernorm")
        dim: Hidden dimension
        **kwargs: Additional arguments (eps, bias, etc.)

    Returns:
        Normalization module
    """
    return NORM_REGISTRY.build(norm_type, dim=dim, **kwargs)


__all__ = [
    "RMSNorm",
    "LayerNorm",
    "build_norm",
    "NORM_REGISTRY",
]
