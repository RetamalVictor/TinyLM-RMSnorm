"""Flash attention and memory-efficient attention implementations.

These provide explicit control over which SDPA kernel is used:
- FlashAttentionOp: Forces Flash Attention 2 kernel
- MemoryEfficientAttentionOp: Forces memory-efficient kernel

Use these when you need specific kernel behavior for benchmarking
or when auto-selection doesn't pick the optimal kernel.
"""

from typing import Optional

import torch
import torch.nn.functional as F

from tinylm.components.attention.ops.base import AttentionOp
from tinylm.components.registry import ATTENTION_OP_REGISTRY


def _check_sdp_kernel_available() -> bool:
    """Check if SDPA kernel selection is available (requires CUDA)."""
    if not torch.cuda.is_available():
        return False
    try:
        from torch.backends.cuda import sdp_kernel  # noqa: F401
        return True
    except ImportError:
        return False


def _check_flash_available() -> bool:
    """Check if Flash Attention is available."""
    return _check_sdp_kernel_available()


def _check_memory_efficient_available() -> bool:
    """Check if memory-efficient attention is available."""
    return _check_sdp_kernel_available()


@ATTENTION_OP_REGISTRY.register("flash")
class FlashAttentionOp(AttentionOp):
    """Flash Attention 2 via SDPA with explicit kernel selection.

    Forces the use of Flash Attention kernel by disabling other backends.
    Falls back to standard attention if Flash is not available.

    Requirements:
    - PyTorch 2.0+
    - CUDA GPU with compute capability >= 8.0 (Ampere or newer)
    - FP16 or BF16 inputs
    - Head dimension <= 128
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
        """Compute attention using Flash Attention kernel.

        Args:
            q: Query tensor [B, H, T, D]
            k: Key tensor [B, H, T_kv, D]
            v: Value tensor [B, H, T_kv, D]
            attn_mask: Optional attention bias (NOTE: Flash Attention has
                       limited mask support)
            is_causal: Whether to apply causal masking
            training: Whether in training mode

        Returns:
            Attention output [B, H, T, D]
        """
        if _check_flash_available() and attn_mask is None:
            # Use Flash Attention with explicit kernel selection
            from torch.backends.cuda import sdp_kernel

            use_causal = is_causal and attn_mask is None
            with sdp_kernel(
                enable_flash=True,
                enable_math=False,
                enable_mem_efficient=False,
            ):
                return F.scaled_dot_product_attention(
                    q, k, v,
                    dropout_p=self.dropout if training else 0.0,
                    is_causal=use_causal,
                    scale=self.scale,
                )
        else:
            # Fallback to standard SDPA (Flash doesn't support arbitrary masks)
            return self._sdpa_fallback(q, k, v, attn_mask, is_causal, training)

    @classmethod
    def is_available(cls) -> bool:
        """Check if Flash Attention is available."""
        return _check_flash_available()


@ATTENTION_OP_REGISTRY.register("memory_efficient")
class MemoryEfficientAttentionOp(AttentionOp):
    """Memory-efficient attention via SDPA with explicit kernel selection.

    Forces the use of memory-efficient attention kernel.
    Good for long sequences where Flash Attention isn't available.

    Requirements:
    - PyTorch 2.0+
    - CUDA GPU
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
        """Compute attention using memory-efficient kernel.

        Args:
            q: Query tensor [B, H, T, D]
            k: Key tensor [B, H, T_kv, D]
            v: Value tensor [B, H, T_kv, D]
            attn_mask: Optional attention bias
            is_causal: Whether to apply causal masking
            training: Whether in training mode

        Returns:
            Attention output [B, H, T, D]
        """
        if _check_memory_efficient_available():
            from torch.backends.cuda import sdp_kernel

            use_causal = is_causal and attn_mask is None
            with sdp_kernel(
                enable_flash=False,
                enable_math=False,
                enable_mem_efficient=True,
            ):
                return F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attn_mask,
                    dropout_p=self.dropout if training else 0.0,
                    is_causal=use_causal,
                    scale=self.scale,
                )
        else:
            # Fallback to standard SDPA
            return self._sdpa_fallback(q, k, v, attn_mask, is_causal, training)

    @classmethod
    def is_available(cls) -> bool:
        """Check if memory-efficient attention is available."""
        return _check_memory_efficient_available()
