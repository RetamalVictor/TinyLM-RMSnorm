"""Factory functions for creating quantized or standard linear layers."""

from typing import Optional

import torch.nn as nn

from tinylm.quant.base import QUANT_REGISTRY, QuantParams
from tinylm.quant.config import QuantConfig


def make_linear(
    in_features: int,
    out_features: int,
    bias: bool = False,
    quant_config: Optional[QuantConfig] = None,
    layer_type: str = "default"
) -> nn.Module:
    """Create Linear or quantized Linear based on config.

    Uses the QUANT_REGISTRY to create the appropriate linear layer
    based on the quantization method specified in the config.

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
        nn.Linear or quantized linear module.

    Examples:
        >>> # Standard linear
        >>> layer = make_linear(512, 512)

        >>> # Ternary quantization
        >>> config = QuantConfig(enabled=True, method="ternary")
        >>> layer = make_linear(512, 512, quant_config=config)

        >>> # Selective quantization (attention only)
        >>> config = QuantConfig(enabled=True, method="ternary",
        ...                      quantize_mlp=False)
        >>> attention_layer = make_linear(512, 512, quant_config=config,
        ...                               layer_type="attention")  # quantized
        >>> mlp_layer = make_linear(512, 512, quant_config=config,
        ...                         layer_type="mlp")  # NOT quantized
    """
    # Determine if quantization should be applied for this layer type
    should_quantize = False
    method = "none"

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
            method = quant_config.method

    # Use registry to create the layer
    if method == "none" or not should_quantize:
        return nn.Linear(in_features, out_features, bias=bias)

    # Get the method from registry
    method_cls = QUANT_REGISTRY.get(method)

    # Check availability
    if not method_cls.is_available():
        raise RuntimeError(
            f"Quantization method '{method}' is not available. "
            f"Available methods: {QUANT_REGISTRY.available_and_ready()}"
        )

    # Create QuantParams from config
    params = QuantParams(
        enabled=quant_config.enabled,
        threshold_factor=quant_config.threshold_factor,
        per_channel=quant_config.per_channel,
        backend=quant_config.backend,
    )

    return method_cls.create_linear(in_features, out_features, bias, params)


def available_methods() -> list:
    """List all registered quantization methods.

    Returns:
        List of method names.
    """
    return QUANT_REGISTRY.available()


def available_and_ready_methods() -> list:
    """List quantization methods that are available and have dependencies installed.

    Returns:
        List of method names that can be used immediately.
    """
    return QUANT_REGISTRY.available_and_ready()
