"""Activation functions for TinyLM."""

import torch
import torch.nn as nn

from tinylm.components.registry import ACTIVATION_REGISTRY


@ACTIVATION_REGISTRY.register("silu")
class SiLU(nn.Module):
    """SiLU (Swish) activation function.

    f(x) = x * sigmoid(x)

    Used by: LLaMA, Mistral
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.silu(x)


@ACTIVATION_REGISTRY.register("gelu")
class GELU(nn.Module):
    """Gaussian Error Linear Unit activation.

    Used by: GPT-2, BERT, Falcon
    """

    def __init__(self, approximate: str = "none"):
        super().__init__()
        self.approximate = approximate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.gelu(x, approximate=self.approximate)


@ACTIVATION_REGISTRY.register("relu")
class ReLU(nn.Module):
    """Rectified Linear Unit activation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.relu(x)


def build_activation(activation_type: str, **kwargs) -> nn.Module:
    """Factory function to build activation.

    Args:
        activation_type: Type of activation ("silu", "gelu", "relu")
        **kwargs: Additional arguments

    Returns:
        Activation module
    """
    return ACTIVATION_REGISTRY.build(activation_type, **kwargs)


def get_activation_fn(activation_type: str) -> callable:
    """Get activation function (not module).

    Args:
        activation_type: Type of activation

    Returns:
        Activation function
    """
    if activation_type == "silu":
        return nn.functional.silu
    elif activation_type == "gelu":
        return nn.functional.gelu
    elif activation_type == "relu":
        return nn.functional.relu
    else:
        raise ValueError(f"Unknown activation: {activation_type}")


__all__ = [
    "SiLU",
    "GELU",
    "ReLU",
    "build_activation",
    "get_activation_fn",
    "ACTIVATION_REGISTRY",
]
