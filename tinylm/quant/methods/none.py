"""No quantization - standard linear layer."""

from typing import Optional

import torch.nn as nn

from tinylm.quant.base import QuantMethod, QuantParams, QUANT_REGISTRY


@QUANT_REGISTRY.register("none")
class NoneQuantMethod(QuantMethod):
    """Standard linear layer without quantization.

    This is the default method that creates standard nn.Linear layers.
    Always available as it has no external dependencies.
    """

    @classmethod
    def is_available(cls) -> bool:
        """Always available."""
        return True

    @classmethod
    def create_linear(
        cls,
        in_features: int,
        out_features: int,
        bias: bool = False,
        params: Optional[QuantParams] = None,
    ) -> nn.Module:
        """Create a standard nn.Linear layer.

        Args:
            in_features: Size of each input sample.
            out_features: Size of each output sample.
            bias: If True, adds a learnable bias.
            params: Ignored for standard linear.

        Returns:
            Standard nn.Linear module.
        """
        return nn.Linear(in_features, out_features, bias=bias)
