"""Multi-Head Attention implementation."""

from typing import Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from tinylm.components.registry import ATTENTION_REGISTRY
from tinylm.components.positional.base import PositionalContext
from tinylm.quant import QuantConfig, make_linear


@ATTENTION_REGISTRY.register("mha")
class MHA(nn.Module):
    """Multi-Head Attention with positional embedding and KV-cache support.

    Supports different positional embedding types via PositionalContext:
    - RoPE: applies rotation to Q and K
    - Learned: no modification (added to input before attention)
    - ALiBi: adds attention bias

    Used by: All architectures (LLaMA, GPT-2, Falcon, Mistral with different configs)
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = False,
        quant_config: Optional[QuantConfig] = None,
    ):
        super().__init__()
        assert dim % n_heads == 0, f"dim {dim} must be divisible by n_heads {n_heads}"

        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads  # For MQA/GQA support later
        self.head_dim = dim // n_heads

        # Projections
        self.qkv = make_linear(
            dim, dim * 3, bias=bias,
            quant_config=quant_config, layer_type="attention"
        )
        self.proj = make_linear(
            dim, dim, bias=bias,
            quant_config=quant_config, layer_type="attention"
        )
        self.dropout = nn.Dropout(dropout)

        # Store reference to positional embedding (set by parent block)
        self.pos_emb = None

    def set_pos_emb(self, pos_emb: nn.Module):
        """Set positional embedding module for Q/K modification."""
        self.pos_emb = pos_emb

    def forward(
        self,
        x: torch.Tensor,
        pos_ctx: PositionalContext,
        cache: Optional[Dict[str, torch.Tensor]] = None,
        start_pos: int = 0,
        pos_emb: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [B, T, D]
            pos_ctx: Positional context with precomputed values
            cache: Optional KV cache dict
            start_pos: Starting position for cache
            pos_emb: Optional positional embedding module (overrides self.pos_emb)

        Returns:
            Output tensor [B, T, D]
        """
        B, T, C = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape to [B, H, T, D]
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply positional embeddings to Q and K (for RoPE)
        pos_emb_module = pos_emb or self.pos_emb
        if pos_emb_module is not None and pos_emb_module.modifies_qk:
            # Update context with current position
            ctx = PositionalContext(
                seq_len=T,
                start_pos=start_pos,
                sin=pos_ctx.sin,
                cos=pos_ctx.cos,
                device=x.device,
                dtype=x.dtype,
            )
            q = pos_emb_module.apply(q, ctx)
            k = pos_emb_module.apply(k, ctx)

        # KV cache handling
        if cache is not None:
            cache['k'][:, :, start_pos:start_pos+T] = k
            cache['v'][:, :, start_pos:start_pos+T] = v
            k = cache['k'][:, :, :start_pos+T]
            v = cache['v'][:, :, :start_pos+T]

        # Get attention bias if applicable (for ALiBi)
        attn_bias = None
        if pos_emb_module is not None and pos_emb_module.modifies_attention:
            attn_bias = pos_emb_module.get_attention_bias(pos_ctx)

        # Compute attention
        # Only use causal mask for multi-token sequences (prefill)
        # Single-token generation (T=1) doesn't need masking
        if attn_bias is not None:
            # With attention bias (ALiBi-style)
            attn = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_bias,
                is_causal=False  # Bias handles causality
            )
        else:
            attn = F.scaled_dot_product_attention(q, k, v, is_causal=(T > 1))

        # Reshape and project
        y = attn.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)
        return self.dropout(y)


# Legacy interface for backward compatibility
def make_attention(
    dim: int,
    n_heads: int,
    dropout: float = 0.0,
    bias: bool = False,
    quant_config: Optional[QuantConfig] = None,
) -> MHA:
    """Create MHA attention module."""
    return MHA(
        dim=dim,
        n_heads=n_heads,
        dropout=dropout,
        bias=bias,
        quant_config=quant_config,
    )
