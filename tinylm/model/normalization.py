"""RMSNorm with optional CUDA acceleration."""

from typing import Tuple
import torch
import torch.nn as nn

# Try to import CUDA module, fallback to CPU implementation if not available
try:
    from tinylm._ext import rmsnorm_cuda, HAS_RMSNORM_CUDA
    HAS_CUDA_KERNEL = HAS_RMSNORM_CUDA
    if not HAS_CUDA_KERNEL:
        raise ImportError("rmsnorm_cuda not compiled")
except ImportError:
    rmsnorm_cuda = None
    HAS_CUDA_KERNEL = False
    import warnings
    warnings.warn(
        "CUDA RMSNorm kernel not found. Falling back to PyTorch implementation. "
        "To enable CUDA kernel, run: uv run python setup.py build_ext --inplace",
        RuntimeWarning,
        stacklevel=2
    )


class RMSNormCUDAFn(torch.autograd.Function):
    """Custom autograd Function for CUDA-accelerated RMSNorm."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        if not HAS_CUDA_KERNEL:
            raise RuntimeError("CUDA RMSNorm module not available")
        y, inv_rms = rmsnorm_cuda.forward(x, weight, eps)
        ctx.save_for_backward(x, weight, inv_rms)
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, None]:
        if not HAS_CUDA_KERNEL:
            raise RuntimeError("CUDA RMSNorm module not available")
        x, weight, inv_rms = ctx.saved_tensors
        dx, dw = rmsnorm_cuda.backward(dy.contiguous(), x, weight, inv_rms, ctx.eps)
        return dx, dw, None


class RMSNormCUDA(nn.Module):
    """Root Mean Square Layer Normalization with optional CUDA acceleration.

    Automatically uses the custom CUDA kernel when available and running on GPU,
    otherwise falls back to a PyTorch native implementation.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if HAS_CUDA_KERNEL and x.is_cuda:
            return RMSNormCUDAFn.apply(x, self.weight, self.eps)
        else:
            rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
            return x * rms * self.weight
