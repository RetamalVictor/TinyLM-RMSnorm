"""Inference utilities for TinyLM."""

from tinylm.inference.generate import generate, sample_top_p
from tinylm.inference.kv_cache import prealloc_kvcache

__all__ = ["generate", "sample_top_p", "prealloc_kvcache"]
