"""INT4 quantization using GPTQ/AWQ (stub for Issue 8)."""

from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
from torch import Tensor

from tinylm.quant.base import QuantMethod, QuantParams, QUANT_REGISTRY


@QUANT_REGISTRY.register("int4")
class Int4QuantMethod(QuantMethod):
    """INT4 quantization using GPTQ or AWQ algorithm.

    This is a stub implementation for Issue 8: INT4 (GPTQ/AWQ) Quantization.
    Currently returns standard linear layers until full implementation.

    Planned features:
    - GPTQ or AWQ weight quantization algorithm
    - 4-bit weight storage with efficient packing
    - Custom INT4 matmul kernel for inference
    - Grouped quantization support for better accuracy
    """

    @classmethod
    def is_available(cls) -> bool:
        """INT4 quantization availability.

        Returns False until implementation is complete.
        """
        return False

    @classmethod
    def create_linear(
        cls,
        in_features: int,
        out_features: int,
        bias: bool = False,
        params: Optional[QuantParams] = None,
    ) -> nn.Module:
        """Create an INT4 quantized linear layer.

        Args:
            in_features: Size of each input sample.
            out_features: Size of each output sample.
            bias: If True, adds a learnable bias.
            params: Quantization parameters including group_size.

        Returns:
            Linear module (currently standard nn.Linear).

        Note:
            This is a stub. Full implementation will create Int4Linear with:
            - weight_int4: Packed INT4 storage (2 weights per byte)
            - scale: FP16 per-group scale
            - zeros: FP16 per-group zero point (for asymmetric)
            - group_size: Number of weights per group (e.g., 128)
        """
        if not cls.is_available():
            raise NotImplementedError(
                "INT4 quantization is not yet implemented. "
                "See Issue 8 for implementation roadmap."
            )

        # TODO: Implement Int4Linear
        # Will use either GPTQ or AWQ algorithm for weight quantization
        # Key differences:
        # - GPTQ: Hessian-based optimization, layer-by-layer
        # - AWQ: Activation-aware scaling, protects salient weights

        return nn.Linear(in_features, out_features, bias=bias)

    @classmethod
    def quantize_weights(
        cls,
        weight: Tensor,
        params: Optional[QuantParams] = None,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """Quantize weights to INT4 (stub).

        Full implementation will use GPTQ/AWQ algorithm with:
        - Hessian-based or activation-aware weight selection
        - Grouped quantization for better accuracy
        - Efficient 4-bit packing

        Args:
            weight: Full-precision weight tensor.
            params: Quantization parameters.

        Returns:
            Tuple of (quantized_weight, metadata).
        """
        if params is None:
            params = QuantParams(bits=4, symmetric=False, group_size=128)

        group_size = params.group_size if params.group_size > 0 else weight.shape[1]

        # Simple grouped quantization (placeholder for GPTQ/AWQ)
        out_features, in_features = weight.shape
        num_groups = (in_features + group_size - 1) // group_size

        # Pad if needed
        if in_features % group_size != 0:
            pad_size = group_size - (in_features % group_size)
            weight = torch.nn.functional.pad(weight, (0, pad_size))

        # Reshape for grouped quantization
        weight_grouped = weight.view(out_features, num_groups, group_size)

        # Find min/max per group (asymmetric quantization)
        min_val = weight_grouped.amin(dim=2, keepdim=True)
        max_val = weight_grouped.amax(dim=2, keepdim=True)

        # Scale and zero point
        scale = (max_val - min_val) / 15.0  # 4-bit range [0, 15]
        scale = scale.clamp(min=1e-8)
        zero = -min_val / scale

        # Quantize to [0, 15]
        weight_int4 = torch.round((weight_grouped - min_val) / scale).clamp(0, 15)

        return weight_int4.to(torch.uint8), {
            "scale": scale.squeeze(-1),
            "zero": zero.squeeze(-1),
            "group_size": group_size,
            "original_in_features": in_features,
            "dtype": "int4",
        }

    @classmethod
    def dequantize_weights(
        cls,
        quantized_weight: Tensor,
        metadata: Dict[str, Any],
    ) -> Tensor:
        """Dequantize INT4 weights to full precision.

        Args:
            quantized_weight: INT4 weight tensor (stored as uint8).
            metadata: Metadata containing scale, zero, group_size.

        Returns:
            Dequantized weight tensor.
        """
        scale = metadata["scale"]
        zero = metadata["zero"]
        group_size = metadata["group_size"]
        original_in_features = metadata.get("original_in_features", None)

        # Reshape scale and zero to broadcast
        out_features = quantized_weight.shape[0]
        num_groups = scale.shape[1]

        # Dequantize
        weight = (quantized_weight.float() - zero.unsqueeze(-1)) * scale.unsqueeze(-1)

        # Reshape back
        weight = weight.view(out_features, -1)

        # Remove padding if needed
        if original_in_features is not None:
            weight = weight[:, :original_in_features]

        return weight
