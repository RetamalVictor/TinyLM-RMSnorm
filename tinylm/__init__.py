"""TinyLM Lab: Multi-architecture language model framework for research."""

from tinylm.model import TinyLM
from tinylm.inference import generate, sample_top_p
from tinylm.quant import QuantConfig, make_linear

__version__ = "0.1.0"
__all__ = [
    "TinyLM",
    "generate",
    "sample_top_p",
    "QuantConfig",
    "make_linear",
]
