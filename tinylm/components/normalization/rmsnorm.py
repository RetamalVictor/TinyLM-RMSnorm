"""RMSNorm with pluggable kernel backends (PyTorch, CUDA, Triton)."""

from typing import Tuple

import torch
import torch.nn as nn

from tinylm.components.registry import NORM_REGISTRY
from tinylm.kernels import get_backend


class RMSNormKernelFn(torch.autograd.Function):
    """Custom autograd Function for kernel-accelerated RMSNorm.

    This function dynamically selects the appropriate kernel backend
    based on the current global setting and device compatibility.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        eps: float
    ) -> torch.Tensor:
        # Get appropriate kernel for this device
        backend = get_backend(x.device)
        kernel = backend.rmsnorm

        # Forward pass
        y, inv_rms = kernel.forward(x, weight, eps)

        # Save for backward
        ctx.save_for_backward(x, weight)
        ctx.inv_rms = inv_rms
        ctx.eps = eps
        ctx.backend_name = backend.name

        return y

    @staticmethod
    def backward(
        ctx,
        dy: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, None]:
        x, weight = ctx.saved_tensors
        inv_rms = ctx.inv_rms
        eps = ctx.eps

        # Get backend - use same one as forward for consistency
        backend = get_backend(x.device)

        # If using PyTorch backend, we can't use the custom backward
        # since PyTorch autograd already handles it through the forward computation.
        # In this case, we need to compute gradients manually.
        if backend.name == "pytorch" or inv_rms is None:
            # Manual gradient computation for PyTorch backend
            # Forward: y = x * rsqrt(mean(x^2) + eps) * w
            # Let rms = rsqrt(mean(x^2) + eps), y = x * rms * w
            rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)

            # Gradient of weight: sum over batch and seq dimensions
            # dw = sum(dy * x * rms, dim=[0, 1, ...])
            dw = (dy * x * rms).sum(dim=tuple(range(dy.dim() - 1)))

            # Gradient of x (more complex due to mean in rms)
            # Using chain rule through the rms normalization
            N = x.shape[-1]
            dx = dy * rms * weight
            # Correction term for the gradient through rms
            dx = dx - x * rms.pow(3) * (x * dy * weight).sum(dim=-1, keepdim=True) / N

            return dx, dw, None

        # Use kernel backward
        kernel = backend.rmsnorm
        dx, dw = kernel.backward(dy.contiguous(), x, weight, inv_rms, eps)
        return dx, dw, None


@NORM_REGISTRY.register("rmsnorm")
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization with pluggable kernel backends.

    Automatically selects the best available kernel backend based on:
    1. Global backend setting (set via `tinylm.kernels.set_backend()`)
    2. Device compatibility (CUDA kernels only work on GPU)
    3. Fallback chain (CUDA -> Triton -> PyTorch)

    Used by: LLaMA, Mistral, Gemma

    Example:
        >>> from tinylm.kernels import set_backend
        >>> set_backend("cuda")  # Force CUDA kernels
        >>> norm = RMSNorm(512)
        >>> y = norm(x)  # Uses CUDA kernel if x is on GPU

        >>> set_backend("auto")  # Auto-select best available
        >>> norm = RMSNorm(512)
        >>> y = norm(x)  # Uses best available kernel
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        """Initialize RMSNorm.

        Args:
            dim: Feature dimension (last dimension of input)
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMSNorm using the best available kernel.

        Args:
            x: Input tensor of shape (..., dim)

        Returns:
            Normalized tensor of same shape
        """
        return RMSNormKernelFn.apply(x, self.weight, self.eps)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, eps={self.eps}"
