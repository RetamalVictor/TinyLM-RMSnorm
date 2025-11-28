"""TinyLM Lab: A minimal language model framework for multi-agent research."""

from tinylm.model import TinyLM, build_sincos, prealloc_kvcache, RMSNormCUDA, Block, MHA
from tinylm.inference import generate, sample_top_p
from tinylm.quant import QuantConfig, make_linear

__version__ = "0.1.0"
__all__ = [
    "TinyLM",
    "build_sincos",
    "prealloc_kvcache",
    "RMSNormCUDA",
    "Block",
    "MHA",
    "generate",
    "sample_top_p",
    "QuantConfig",
    "make_linear",
]
