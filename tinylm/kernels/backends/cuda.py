"""CUDA kernel backend - uses custom CUDA kernels when available."""

from typing import Tuple, Optional
import torch

from tinylm.kernels.base import KernelBackend, RMSNormKernel

# Lazy import of CUDA extension
_rmsnorm_cuda = None
_cuda_available = None


def _get_rmsnorm_cuda():
    """Lazily load the CUDA RMSNorm extension."""
    global _rmsnorm_cuda, _cuda_available
    if _cuda_available is None:
        try:
            from tinylm._ext import rmsnorm_cuda
            _rmsnorm_cuda = rmsnorm_cuda
            _cuda_available = rmsnorm_cuda is not None
        except ImportError:
            _rmsnorm_cuda = None
            _cuda_available = False
    return _rmsnorm_cuda


class CUDARMSNormKernel(RMSNormKernel):
    """CUDA-accelerated RMSNorm kernel implementation."""

    @staticmethod
    def forward(
        x: torch.Tensor,
        weight: torch.Tensor,
        eps: float
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass using custom CUDA kernel.

        Returns both output and inv_rms for use in backward pass.
        """
        cuda_module = _get_rmsnorm_cuda()
        if cuda_module is None:
            raise RuntimeError("CUDA RMSNorm kernel not available")
        y, inv_rms = cuda_module.forward(x, weight, eps)
        return y, inv_rms

    @staticmethod
    def backward(
        dy: torch.Tensor,
        x: torch.Tensor,
        weight: torch.Tensor,
        inv_rms: Optional[torch.Tensor],
        eps: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Backward pass using custom CUDA kernel."""
        cuda_module = _get_rmsnorm_cuda()
        if cuda_module is None:
            raise RuntimeError("CUDA RMSNorm kernel not available")
        if inv_rms is None:
            raise RuntimeError("CUDA backward requires inv_rms from forward pass")
        dx, dw = cuda_module.backward(dy.contiguous(), x, weight, inv_rms, eps)
        return dx, dw


class CUDABackend(KernelBackend):
    """CUDA kernel backend using custom compiled kernels."""

    name = "cuda"

    # Kernel implementations
    rmsnorm = CUDARMSNormKernel

    @classmethod
    def is_available(cls) -> bool:
        """Check if CUDA kernels are compiled and CUDA is available."""
        if not torch.cuda.is_available():
            return False
        return _get_rmsnorm_cuda() is not None

    @classmethod
    def supports_device(cls, device: torch.device) -> bool:
        """CUDA backend only supports CUDA devices."""
        return device.type == "cuda" and cls.is_available()
