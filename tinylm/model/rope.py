"""Rotary Position Embeddings (RoPE) implementation."""

from typing import Tuple
import torch


def rotary_embeddings(
    x: torch.Tensor,
    sin: torch.Tensor,
    cos: torch.Tensor
) -> torch.Tensor:
    """Apply Rotary Position Embeddings (RoPE) to input tensor.

    RoPE encodes positional information by rotating pairs of dimensions
    in the feature space, providing better extrapolation to longer sequences.

    Args:
        x: Input tensor of shape [batch, heads, seq_len, head_dim]
        sin: Sine values for rotation of shape [1, 1, seq_len, head_dim]
        cos: Cosine values for rotation of shape [1, 1, seq_len, head_dim]

    Returns:
        Tensor with rotary embeddings applied, same shape as input
    """
    x1, x2 = x[..., ::2], x[..., 1::2]
    xr = torch.stack((-x2, x1), dim=-1).reshape_as(x)
    return x * cos + xr * sin


def build_sincos(
    seq_len: int,
    dim: int,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute sine and cosine values for Rotary Position Embeddings.

    Args:
        seq_len: Maximum sequence length to compute embeddings for
        dim: Dimension of each attention head
        device: Device to create tensors on

    Returns:
        Tuple of (sin, cos) tensors, each of shape [1, 1, seq_len, dim]
    """
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.einsum('t,f->tf', t, inv_freq)
    sin = torch.sin(torch.cat([freqs, freqs], dim=-1))[None, None, :, :]
    cos = torch.cos(torch.cat([freqs, freqs], dim=-1))[None, None, :, :]
    return sin, cos
