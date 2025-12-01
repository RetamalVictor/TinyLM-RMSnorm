"""Rotary Position Embeddings (RoPE) implementation."""

from typing import Dict, Optional
import torch
import torch.nn as nn

from tinylm.components.registry import POS_EMB_REGISTRY
from tinylm.components.positional.base import PositionalContext


@POS_EMB_REGISTRY.register("rope")
class RoPE(nn.Module):
    """Rotary Position Embeddings.

    RoPE encodes positional information by rotating pairs of dimensions
    in the feature space, providing better extrapolation to longer sequences.

    Used by: LLaMA, Mistral, Falcon (some variants)
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        base: int = 10000,
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

    def precompute(
        self,
        max_seq_len: int,
        device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """Precompute sine and cosine values for RoPE.

        Args:
            max_seq_len: Maximum sequence length
            device: Device to create tensors on

        Returns:
            Dictionary with 'sin' and 'cos' tensors of shape [1, 1, seq_len, dim]
        """
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, device=device).float() / self.dim)
        )
        t = torch.arange(max_seq_len, device=device).float()
        freqs = torch.einsum('t,f->tf', t, inv_freq)
        sin = torch.sin(torch.cat([freqs, freqs], dim=-1))[None, None, :, :]
        cos = torch.cos(torch.cat([freqs, freqs], dim=-1))[None, None, :, :]
        return {'sin': sin, 'cos': cos}

    def apply(
        self,
        x: torch.Tensor,
        ctx: PositionalContext
    ) -> torch.Tensor:
        """Apply rotary embeddings to Q or K tensor.

        Args:
            x: Query or Key tensor of shape [B, H, T, D]
            ctx: Positional context with sin/cos values

        Returns:
            Tensor with rotary embeddings applied
        """
        start = ctx.start_pos
        end = start + x.size(2)
        sin = ctx.sin[:, :, start:end, :]
        cos = ctx.cos[:, :, start:end, :]

        # Apply rotation
        x1, x2 = x[..., ::2], x[..., 1::2]
        xr = torch.stack((-x2, x1), dim=-1).reshape_as(x)
        return x * cos + xr * sin

    @property
    def adds_to_input(self) -> bool:
        return False

    @property
    def modifies_qk(self) -> bool:
        return True

    @property
    def modifies_attention(self) -> bool:
        return False

    def get_attention_bias(self, ctx: PositionalContext) -> Optional[torch.Tensor]:
        return None

    def forward(self, x: torch.Tensor, ctx: PositionalContext) -> torch.Tensor:
        """Alias for apply() for nn.Module compatibility."""
        return self.apply(x, ctx)
