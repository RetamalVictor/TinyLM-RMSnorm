"""INT8 weight-only quantization (stub for Issue 6)."""

from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
from torch import Tensor

from tinylm.quant.base import QuantMethod, QuantParams, QUANT_REGISTRY


@QUANT_REGISTRY.register("int8")
class Int8QuantMethod(QuantMethod):
    """INT8 weight-only quantization.

    This is a stub implementation for Issue 6: Weight-Only INT8 Quantization.
    Currently returns standard linear layers until full implementation.

    Planned features:
    - Per-channel weight quantization to INT8
    - Dequantization-on-the-fly during forward pass
    - Support for saving/loading quantized checkpoints
    """

    @classmethod
    def is_available(cls) -> bool:
        """INT8 quantization is available (uses PyTorch native ops)."""
        # Will return True once implemented
        # Currently returns False to indicate stub status
        return False

    @classmethod
    def create_linear(
        cls,
        in_features: int,
        out_features: int,
        bias: bool = False,
        params: Optional[QuantParams] = None,
    ) -> nn.Module:
        """Create an INT8 quantized linear layer.

        Args:
            in_features: Size of each input sample.
            out_features: Size of each output sample.
            bias: If True, adds a learnable bias.
            params: Quantization parameters.

        Returns:
            Linear module (currently standard nn.Linear, will be Int8Linear).

        Note:
            This is a stub. Full implementation will create Int8Linear with:
            - weight_int8: INT8 weight storage
            - scale: FP32 per-channel scale
            - Forward: w = weight_int8.float() * scale; F.linear(x, w)
        """
        if not cls.is_available():
            raise NotImplementedError(
                "INT8 quantization is not yet implemented. "
                "See Issue 6 for implementation roadmap."
            )

        # TODO: Implement Int8Linear
        # class Int8Linear(nn.Module):
        #     def __init__(self, in_features, out_features, bias=False):
        #         super().__init__()
        #         self.weight_int8 = nn.Parameter(
        #             torch.zeros(out_features, in_features, dtype=torch.int8),
        #             requires_grad=False
        #         )
        #         self.scale = nn.Parameter(
        #             torch.ones(out_features, 1), requires_grad=False
        #         )
        #         if bias:
        #             self.bias = nn.Parameter(torch.zeros(out_features))
        #         else:
        #             self.register_parameter('bias', None)
        #
        #     def forward(self, x):
        #         w = self.weight_int8.float() * self.scale
        #         return F.linear(x, w, self.bias)

        return nn.Linear(in_features, out_features, bias=bias)

    @classmethod
    def quantize_weights(
        cls,
        weight: Tensor,
        params: Optional[QuantParams] = None,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """Quantize weights to INT8.

        Stub implementation - will perform per-channel symmetric quantization.

        Args:
            weight: Full-precision weight tensor [out_features, in_features].
            params: Quantization parameters.

        Returns:
            Tuple of (quantized_weight_int8, metadata).
        """
        if params is None:
            params = QuantParams(bits=8, symmetric=True)

        # Per-channel symmetric quantization
        # Find max abs value per output channel
        max_val = weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)

        # Scale to [-127, 127] range (symmetric INT8)
        scale = max_val / 127.0

        # Quantize
        weight_int8 = torch.round(weight / scale).clamp(-127, 127).to(torch.int8)

        return weight_int8, {"scale": scale, "dtype": "int8"}

    @classmethod
    def dequantize_weights(
        cls,
        quantized_weight: Tensor,
        metadata: Dict[str, Any],
    ) -> Tensor:
        """Dequantize INT8 weights to full precision.

        Args:
            quantized_weight: INT8 weight tensor.
            metadata: Metadata containing scale.

        Returns:
            Dequantized FP32 weight tensor.
        """
        scale = metadata.get("scale", 1.0)
        return quantized_weight.float() * scale
