"""Base classes and protocols for positional embeddings."""

from dataclasses import dataclass
from typing import Optional, Dict, Protocol, runtime_checkable
import torch


@dataclass
class PositionalContext:
    """Context for positional embeddings, carrying all possible inputs.

    Different implementations use different fields:
    - RoPE: uses sin, cos, start_pos
    - Learned: uses positions
    - ALiBi: returns attention_bias
    """
    seq_len: int
    start_pos: int = 0
    device: Optional[torch.device] = None
    dtype: Optional[torch.dtype] = None

    # Pre-computed values (RoPE)
    sin: Optional[torch.Tensor] = None
    cos: Optional[torch.Tensor] = None

    # Position indices (Learned)
    positions: Optional[torch.Tensor] = None


@runtime_checkable
class PositionalEmbedding(Protocol):
    """Protocol for positional embeddings."""

    def precompute(
        self,
        max_seq_len: int,
        device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """Precompute positional information for caching.

        Returns:
            Dictionary of precomputed tensors (e.g., {'sin': ..., 'cos': ...})
        """
        ...

    def apply(
        self,
        x: torch.Tensor,
        ctx: PositionalContext
    ) -> torch.Tensor:
        """Apply positional information to input.

        Args:
            x: Query/Key tensor of shape [B, H, T, D] or input tensor [B, T, D]
            ctx: Positional context with precomputed values

        Returns:
            Tensor with positional information applied
        """
        ...

    @property
    def adds_to_input(self) -> bool:
        """Whether this embedding adds to token embeddings (e.g., Learned)."""
        ...

    @property
    def modifies_qk(self) -> bool:
        """Whether this embedding modifies Q/K in attention (e.g., RoPE)."""
        ...

    @property
    def modifies_attention(self) -> bool:
        """Whether this embedding modifies attention scores (e.g., ALiBi)."""
        ...

    def get_attention_bias(
        self,
        ctx: PositionalContext
    ) -> Optional[torch.Tensor]:
        """Get attention bias if applicable (for ALiBi-style embeddings)."""
        ...
