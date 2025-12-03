"""Standard attention using PyTorch's scaled_dot_product_attention.

This is the default attention implementation that auto-selects the best
available kernel (Flash, Memory-Efficient, or Math) based on hardware
and input characteristics.
"""

from typing import Optional

import torch

from tinylm.components.attention.ops.base import AttentionOp
from tinylm.components.registry import ATTENTION_OP_REGISTRY


@ATTENTION_OP_REGISTRY.register("standard")
class StandardAttentionOp(AttentionOp):
    """Standard attention using F.scaled_dot_product_attention.

    Automatically selects the best available kernel:
    - Flash Attention 2 (if available and inputs are suitable)
    - Memory-efficient attention (xformers-style)
    - Math kernel (fallback)

    This is the recommended default for most use cases.
    """

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
        training: bool = False,
    ) -> torch.Tensor:
        """Compute attention using SDPA.

        Args:
            q: Query tensor [B, H, T, D]
            k: Key tensor [B, H, T_kv, D]
            v: Value tensor [B, H, T_kv, D]
            attn_mask: Optional attention bias (additive)
            is_causal: Whether to apply causal masking
            training: Whether in training mode

        Returns:
            Attention output [B, H, T, D]
        """
        return self._sdpa_fallback(q, k, v, attn_mask, is_causal, training)

    @classmethod
    def is_available(cls) -> bool:
        """Always available (uses PyTorch's SDPA)."""
        return True
