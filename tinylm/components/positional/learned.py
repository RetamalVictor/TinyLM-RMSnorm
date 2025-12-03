"""Learned positional embeddings (GPT-style)."""

from typing import Dict, Optional

import torch
import torch.nn as nn

from tinylm.components.positional.base import PositionalContext
from tinylm.components.registry import POS_EMB_REGISTRY


@POS_EMB_REGISTRY.register("learned")
class LearnedPositionalEmbedding(nn.Module):
    """Learned positional embeddings.

    Standard learnable position embeddings that are added to token embeddings.
    Position indices are learned as an embedding table.

    Used by: GPT-2, GPT-3, BERT
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        **kwargs  # Accept but ignore extra args for compatibility
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.embedding = nn.Embedding(max_seq_len, dim)

    def precompute(
        self,
        max_seq_len: int,
        device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """Precompute position indices.

        For learned embeddings, we just prepare position indices.
        The actual embeddings are looked up during forward pass.
        """
        positions = torch.arange(max_seq_len, device=device)
        return {'positions': positions}

    def apply(
        self,
        x: torch.Tensor,
        ctx: PositionalContext
    ) -> torch.Tensor:
        """Get positional embeddings to add to input.

        Args:
            x: Input tensor of shape [B, T, D] (token embeddings)
            ctx: Positional context

        Returns:
            Positional embeddings to add, shape [B, T, D] or [1, T, D]
        """
        T = x.size(1) if x.dim() == 3 else ctx.seq_len
        start = ctx.start_pos
        positions = torch.arange(start, start + T, device=x.device)
        return self.embedding(positions).unsqueeze(0)  # [1, T, D]

    @property
    def adds_to_input(self) -> bool:
        return True

    @property
    def modifies_qk(self) -> bool:
        return False

    @property
    def modifies_attention(self) -> bool:
        return False

    def get_attention_bias(self, ctx: PositionalContext) -> Optional[torch.Tensor]:
        return None

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """Direct forward for position indices.

        Args:
            positions: Position indices of shape [T] or [B, T]

        Returns:
            Position embeddings of shape [T, D] or [B, T, D]
        """
        return self.embedding(positions)
