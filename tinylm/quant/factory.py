"""Factory functions for creating quantized or standard linear layers."""

from typing import Optional

import torch.nn as nn

from .config import QuantConfig


def make_linear(
    in_features: int,
    out_features: int,
    bias: bool = False,
    quant_config: Optional[QuantConfig] = None,
    layer_type: str = "default"
) -> nn.Module:
    """Create Linear or TernaryLinear based on config.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        bias: If True, adds a learnable bias. Default: False.
        quant_config: Quantization configuration. If None or disabled,
            returns standard nn.Linear.
        layer_type: Type of layer being created. One of:
            - "attention": For qkv and proj layers in MHA
            - "mlp": For fc1 and fc2 layers in MLP block
            - "head": For output projection head
            - "default": Uses enabled flag only

    Returns:
        nn.Linear or TernaryLinear module.
    """
    # Check if quantization should be applied for this layer type
    should_quantize = False
    if quant_config and quant_config.enabled:
        if layer_type == "attention":
            should_quantize = quant_config.quantize_attention
        elif layer_type == "mlp":
            should_quantize = quant_config.quantize_mlp
        elif layer_type == "head":
            should_quantize = quant_config.quantize_head
        else:  # default
            should_quantize = True

    if should_quantize:
        try:
            from bittorch.nn import TernaryLinear
            return TernaryLinear(
                in_features,
                out_features,
                bias=bias,
                threshold_factor=quant_config.threshold_factor,
                per_channel=quant_config.per_channel,
                backend=quant_config.backend,
            )
        except ImportError as e:
            raise ImportError(
                "BitTorch is required for ternary quantization. "
                "Install it with: pip install -e /path/to/bittorch"
            ) from e

    return nn.Linear(in_features, out_features, bias=bias)
