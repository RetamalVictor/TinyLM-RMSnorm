"""Standard attention using PyTorch's scaled_dot_product_attention.

This is the default attention implementation that auto-selects the best
available kernel (Flash, Memory-Efficient, or Math) based on hardware
and input characteristics.
"""

from typing import Optional
import torch
import torch.nn.functional as F

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
        # Don't use is_causal if we have an explicit mask
        # (they may conflict or double-apply causality)
        use_causal = is_causal and attn_mask is None

        return F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if training else 0.0,
            is_causal=use_causal,
            scale=self.scale,
        )

    @classmethod
    def is_available(cls) -> bool:
        """Always available (uses PyTorch's SDPA)."""
        return True
