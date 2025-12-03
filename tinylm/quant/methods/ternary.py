"""Ternary quantization using BitTorch."""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from tinylm.quant.base import QUANT_REGISTRY, QuantMethod, QuantParams

# Check if BitTorch is available
_BITTORCH_AVAILABLE = False
try:
    from bittorch.nn import TernaryLinear
    _BITTORCH_AVAILABLE = True
except ImportError:
    TernaryLinear = None


@QUANT_REGISTRY.register("ternary")
class TernaryQuantMethod(QuantMethod):
    """Ternary quantization using BitTorch TernaryLinear.

    Ternary quantization represents weights using three values: {-1, 0, +1}
    multiplied by a learned or computed scale factor. This provides significant
    memory reduction while maintaining reasonable accuracy.

    Requires BitTorch library to be installed.
    """

    @classmethod
    def is_available(cls) -> bool:
        """Check if BitTorch is available."""
        return _BITTORCH_AVAILABLE

    @classmethod
    def create_linear(
        cls,
        in_features: int,
        out_features: int,
        bias: bool = False,
        params: Optional[QuantParams] = None,
    ) -> nn.Module:
        """Create a TernaryLinear layer.

        Args:
            in_features: Size of each input sample.
            out_features: Size of each output sample.
            bias: If True, adds a learnable bias.
            params: Quantization parameters containing threshold_factor,
                per_channel, and backend settings.

        Returns:
            TernaryLinear module from BitTorch.

        Raises:
            ImportError: If BitTorch is not installed.
        """
        if not cls.is_available():
            raise ImportError(
                "BitTorch is required for ternary quantization. "
                "Install it with: pip install bittorch"
            )

        # Use defaults if params not provided
        if params is None:
            params = QuantParams()

        return TernaryLinear(
            in_features,
            out_features,
            bias=bias,
            threshold_factor=params.threshold_factor,
            per_channel=params.per_channel,
            backend=params.backend,
        )

    @classmethod
    def quantize_weights(
        cls,
        weight: Tensor,
        params: Optional[QuantParams] = None,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """Quantize weights to ternary representation.

        This performs post-training quantization to ternary values.

        Args:
            weight: Full-precision weight tensor.
            params: Quantization parameters.

        Returns:
            Tuple of (quantized_weight, metadata) where metadata contains
            the scale factor used.
        """
        if params is None:
            params = QuantParams()

        threshold_factor = params.threshold_factor

        # Compute threshold and scale
        if params.per_channel:
            # Per-output-channel quantization
            abs_weight = weight.abs()
            mean_abs = abs_weight.mean(dim=1, keepdim=True)
            threshold = threshold_factor * mean_abs
            scale = abs_weight.sum(dim=1, keepdim=True) / (
                (abs_weight > threshold).sum(dim=1, keepdim=True).clamp(min=1)
            )
        else:
            # Global quantization
            abs_weight = weight.abs()
            mean_abs = abs_weight.mean()
            threshold = threshold_factor * mean_abs
            scale = abs_weight.sum() / (abs_weight > threshold).sum().clamp(min=1)

        # Ternarize: values > threshold -> +1, < -threshold -> -1, else -> 0
        ternary = torch.zeros_like(weight)
        ternary[weight > threshold] = 1.0
        ternary[weight < -threshold] = -1.0

        return ternary, {"scale": scale, "threshold": threshold}

    @classmethod
    def dequantize_weights(
        cls,
        quantized_weight: Tensor,
        metadata: Dict[str, Any],
    ) -> Tensor:
        """Dequantize ternary weights back to full precision approximation.

        Args:
            quantized_weight: Ternary weight tensor ({-1, 0, +1}).
            metadata: Metadata containing scale factor.

        Returns:
            Dequantized weight tensor.
        """
        scale = metadata.get("scale", 1.0)
        return quantized_weight * scale
