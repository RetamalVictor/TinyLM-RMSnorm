"""Positional embedding components for TinyLM."""

from tinylm.components.registry import POS_EMB_REGISTRY
from tinylm.components.positional.base import PositionalContext, PositionalEmbedding
from tinylm.components.positional.rope import RoPE
from tinylm.components.positional.learned import LearnedPositionalEmbedding


def build_pos_emb(
    pos_type: str,
    dim: int,
    max_seq_len: int = 4096,
    **kwargs
) -> "nn.Module":
    """Factory function to build positional embedding.

    Args:
        pos_type: Type of positional embedding ("rope" or "learned")
        dim: Hidden dimension (head_dim for RoPE, model dim for learned)
        max_seq_len: Maximum sequence length
        **kwargs: Additional arguments (base for RoPE, etc.)

    Returns:
        Positional embedding module
    """
    return POS_EMB_REGISTRY.build(
        pos_type,
        dim=dim,
        max_seq_len=max_seq_len,
        **kwargs
    )


__all__ = [
    "PositionalContext",
    "PositionalEmbedding",
    "RoPE",
    "LearnedPositionalEmbedding",
    "build_pos_emb",
    "POS_EMB_REGISTRY",
]
