"""LayerNorm wrapper for TinyLM components."""

import torch
import torch.nn as nn

from tinylm.components.registry import NORM_REGISTRY


@NORM_REGISTRY.register("layernorm")
class LayerNorm(nn.Module):
    """Standard Layer Normalization.

    Wraps PyTorch's nn.LayerNorm with unified interface.

    Used by: GPT-2, Falcon
    """

    def __init__(self, dim: int, eps: float = 1e-5, bias: bool = True):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=True)
        self.dim = dim
        self.eps = eps
        # Handle bias parameter (LayerNorm always has bias by default)
        if not bias:
            self.norm.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)
