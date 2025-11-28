"""Quantization utilities for TinyLM with BitTorch integration."""

from .config import QuantConfig
from .factory import make_linear

__all__ = ["QuantConfig", "make_linear"]
