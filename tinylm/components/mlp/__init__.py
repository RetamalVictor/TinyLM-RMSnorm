"""MLP components for TinyLM."""

from typing import Optional

import torch.nn as nn

from tinylm.components.mlp.gated import GatedMLP

# Import to register with registry
from tinylm.components.mlp.standard import StandardMLP
from tinylm.components.registry import MLP_REGISTRY
from tinylm.quant import QuantConfig


def build_mlp(
    mlp_type: str,
    dim: int,
    hidden_dim: Optional[int] = None,
    hidden_ratio: float = 4.0,
    activation: str = "silu",
    dropout: float = 0.0,
    bias: bool = False,
    quant_config: Optional[QuantConfig] = None,
) -> nn.Module:
    """Factory function to build MLP.

    Args:
        mlp_type: Type of MLP ("standard" or "gated")
        dim: Model dimension
        hidden_dim: Hidden dimension (if None, computed from hidden_ratio)
        hidden_ratio: Hidden dim multiplier (default 4x)
        activation: Activation function name
        dropout: Dropout rate
        bias: Whether to use bias
        quant_config: Quantization configuration

    Returns:
        MLP module
    """
    return MLP_REGISTRY.build(
        mlp_type,
        dim=dim,
        hidden_dim=hidden_dim,
        hidden_ratio=hidden_ratio,
        activation=activation,
        dropout=dropout,
        bias=bias,
        quant_config=quant_config,
    )


__all__ = [
    "StandardMLP",
    "GatedMLP",
    "build_mlp",
    "MLP_REGISTRY",
]
