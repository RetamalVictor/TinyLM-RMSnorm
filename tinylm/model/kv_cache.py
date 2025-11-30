"""KV-cache utilities for efficient autoregressive generation."""

from typing import Dict, List
import torch


def prealloc_kvcache(
    B: int,
    max_seq: int,
    n_heads: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    n_layers: int = 1
) -> List[Dict[str, torch.Tensor]]:
    """Pre-allocate KV-cache tensors for efficient autoregressive generation.

    The KV-cache stores key and value tensors from previous timesteps to avoid
    recomputation during incremental decoding, reducing quadratic complexity
    to linear for generation.

    Args:
        B: Batch size
        max_seq: Maximum sequence length to cache
        n_heads: Number of attention heads
        head_dim: Dimension of each attention head
        device: Device to allocate tensors on
        dtype: Data type for tensors (e.g., float16 for memory efficiency)
        n_layers: Number of transformer layers (each needs its own cache)

    Returns:
        List of dictionaries (one per layer), each with 'k' and 'v' keys
        containing pre-allocated tensors of shape [batch_size, n_heads, max_seq, head_dim]
    """
    caches = []
    for _ in range(n_layers):
        k = torch.empty(B, n_heads, max_seq, head_dim, device=device, dtype=dtype)
        v = torch.empty(B, n_heads, max_seq, head_dim, device=device, dtype=dtype)
        caches.append({'k': k, 'v': v})
    return caches
