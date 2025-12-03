"""Quantization utilities for TinyLM with pluggable backends.

This module provides a registry-based system for quantization methods,
supporting:
- none: Standard nn.Linear (no quantization)
- ternary: Ternary quantization via BitTorch

Example:
    >>> from tinylm.quant import QuantConfig, make_linear, available_methods
    >>>
    >>> # List available methods
    >>> print(available_methods())  # ['none', 'ternary']
    >>>
    >>> # Create standard linear
    >>> layer = make_linear(512, 512)
    >>>
    >>> # Create with ternary quantization
    >>> config = QuantConfig(enabled=True, method="ternary")
    >>> layer = make_linear(512, 512, quant_config=config)
"""

from tinylm.quant.base import (
    QUANT_REGISTRY,
    QuantMethod,
    QuantParams,
    QuantRegistry,
)
from tinylm.quant.config import QuantConfig, QuantMethodType
from tinylm.quant.factory import (
    available_and_ready_methods,
    available_methods,
    make_linear,
)

# Import methods to register them
from tinylm.quant.methods import (  # noqa: F401
    NoneQuantMethod,
    TernaryQuantMethod,
)

__all__ = [
    # Config
    "QuantConfig",
    "QuantMethodType",
    # Base classes
    "QuantMethod",
    "QuantParams",
    "QuantRegistry",
    "QUANT_REGISTRY",
    # Factory
    "make_linear",
    "available_methods",
    "available_and_ready_methods",
    # Methods
    "NoneQuantMethod",
    "TernaryQuantMethod",
]
