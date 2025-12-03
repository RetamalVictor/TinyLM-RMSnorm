"""Base class for attention operations.

AttentionOp separates the attention computation from:
- QKV projection (handled by MHA)
- Positional embedding application (handled by PositionalEmbedding)
- KV caching (handled by CacheManager)

This allows swapping attention implementations (standard, flash, sparse, etc.)
without modifying the MHA class.
"""

from abc import ABC, abstractmethod
from typing import Optional
import torch


class AttentionOp(ABC):
    """Abstract interface for attention computation.

    All implementations receive pre-processed Q, K, V tensors and return
    the attention output. The interface assumes:

    - Q: [B, H, T, D] query tensor
    - K: [B, H, T_kv, D] key tensor (may be longer than Q due to KV cache)
    - V: [B, H, T_kv, D] value tensor
    - Output: [B, H, T, D] attention output

    Implementations should handle:
    - Causal masking (via is_causal flag)
    - Custom attention masks/biases (for ALiBi, etc.)
    - Dropout during training
    """

    def __init__(self, dropout: float = 0.0, scale: Optional[float] = None):
        """Initialize attention operation.

        Args:
            dropout: Dropout probability for attention weights
            scale: Optional scale factor (default: 1/sqrt(head_dim))
        """
        self.dropout = dropout
        self.scale = scale

    @abstractmethod
    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
        training: bool = False,
    ) -> torch.Tensor:
        """Compute attention.

        Args:
            q: Query tensor [B, H, T, D]
            k: Key tensor [B, H, T_kv, D]
            v: Value tensor [B, H, T_kv, D]
            attn_mask: Optional attention mask/bias [H, T, T_kv] or [B, H, T, T_kv]
                       Additive bias (e.g., ALiBi) or boolean mask
            is_causal: Whether to apply causal masking
            training: Whether in training mode (for dropout)

        Returns:
            Attention output [B, H, T, D]
        """
        pass

    @classmethod
    def is_available(cls) -> bool:
        """Check if this attention op is available on current hardware."""
        return True

    @property
    def name(self) -> str:
        """Return the registered name of this attention op."""
        return getattr(self, '_name', self.__class__.__name__)
