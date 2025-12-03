"""KV Cache Manager abstraction for TinyLM.

Provides a clean interface for KV cache management, enabling:
- Swappable cache implementations (standard, block-based, compressed)
- Consistent API across different caching strategies
- Future extensions like paged attention, cache compression, etc.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, List
import torch


class CacheManager(ABC):
    """Abstract base class for KV cache management.

    Interface matches the plan from ISSUES.md R1:
    - allocate(batch_size, max_seq_len) -> None
    - update(layer_idx, k, v, positions) -> None
    - get(layer_idx, end_pos) -> Tuple[Tensor, Tensor]
    - reset() -> None
    """

    @abstractmethod
    def allocate(self, batch_size: int, max_seq_len: int) -> None:
        """Allocate cache storage for all layers.

        Args:
            batch_size: Number of sequences in batch
            max_seq_len: Maximum sequence length to cache
        """
        pass

    @abstractmethod
    def update(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
        start_pos: int,
    ) -> None:
        """Update cache with new key-value pairs.

        Args:
            layer_idx: Which transformer layer (0-indexed)
            k: Key tensor [B, H, T, D]
            v: Value tensor [B, H, T, D]
            start_pos: Starting position in sequence
        """
        pass

    @abstractmethod
    def get(
        self,
        layer_idx: int,
        end_pos: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve cached key-value pairs.

        Args:
            layer_idx: Which transformer layer (0-indexed)
            end_pos: End position (exclusive) to retrieve up to

        Returns:
            Tuple of (keys, values) tensors [B, H, end_pos, D]
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset cache state for new generation."""
        pass

    @property
    @abstractmethod
    def n_layers(self) -> int:
        """Number of layers this cache supports."""
        pass

    @property
    @abstractmethod
    def is_allocated(self) -> bool:
        """Whether cache has been allocated."""
        pass


class StandardCache(CacheManager):
    """Standard pre-allocated contiguous KV cache.

    Replicates the original TinyLM caching behavior:
    - Pre-allocates full [B, H, max_seq, D] tensors per layer
    - O(1) update and retrieval via slicing
    """

    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        head_dim: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """Initialize cache configuration.

        Args:
            n_layers: Number of transformer layers
            n_heads: Number of attention heads
            head_dim: Dimension per attention head
            device: Device for cache tensors (default: cpu)
            dtype: Data type for cache tensors (default: float32)
        """
        self._n_layers = n_layers
        self._n_heads = n_heads
        self._head_dim = head_dim
        self._device = device or torch.device('cpu')
        self._dtype = dtype or torch.float32

        # Cache storage: list of {'k': Tensor, 'v': Tensor} per layer
        self._caches: List[Dict[str, torch.Tensor]] = []
        self._allocated = False
        self._batch_size = 0
        self._max_seq_len = 0

    def allocate(self, batch_size: int, max_seq_len: int) -> None:
        """Pre-allocate contiguous cache tensors for all layers."""
        self._caches = []
        self._batch_size = batch_size
        self._max_seq_len = max_seq_len

        for _ in range(self._n_layers):
            k = torch.zeros(
                batch_size, self._n_heads, max_seq_len, self._head_dim,
                device=self._device, dtype=self._dtype
            )
            v = torch.zeros(
                batch_size, self._n_heads, max_seq_len, self._head_dim,
                device=self._device, dtype=self._dtype
            )
            self._caches.append({'k': k, 'v': v})

        self._allocated = True

    def update(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
        start_pos: int,
    ) -> None:
        """Update cache at specified position."""
        if not self._allocated:
            raise RuntimeError("Cache not allocated. Call allocate() first.")
        if layer_idx >= self._n_layers:
            raise IndexError(f"layer_idx {layer_idx} >= n_layers {self._n_layers}")

        T = k.size(2)  # sequence length of new tokens
        cache = self._caches[layer_idx]
        cache['k'][:, :, start_pos:start_pos + T] = k
        cache['v'][:, :, start_pos:start_pos + T] = v

    def get(
        self,
        layer_idx: int,
        end_pos: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve cached KV up to end_pos."""
        if not self._allocated:
            raise RuntimeError("Cache not allocated. Call allocate() first.")
        if layer_idx >= self._n_layers:
            raise IndexError(f"layer_idx {layer_idx} >= n_layers {self._n_layers}")

        cache = self._caches[layer_idx]
        return cache['k'][:, :, :end_pos], cache['v'][:, :, :end_pos]

    def reset(self) -> None:
        """Reset cache to unallocated state."""
        self._caches = []
        self._allocated = False
        self._batch_size = 0
        self._max_seq_len = 0

    @property
    def n_layers(self) -> int:
        return self._n_layers

    @property
    def is_allocated(self) -> bool:
        return self._allocated

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def max_seq_len(self) -> int:
        return self._max_seq_len


__all__ = [
    "CacheManager",
    "StandardCache",
]
