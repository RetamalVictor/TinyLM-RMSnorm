"""Quantization method implementations."""

from tinylm.quant.methods.none import NoneQuantMethod
from tinylm.quant.methods.ternary import TernaryQuantMethod

__all__ = [
    "NoneQuantMethod",
    "TernaryQuantMethod",
]
