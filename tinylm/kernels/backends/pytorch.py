"""PyTorch native backend - always available, works on CPU and GPU."""

from typing import Optional, Tuple

import torch

from tinylm.kernels.base import KernelBackend, RMSNormKernel


class PyTorchRMSNormKernel(RMSNormKernel):
    """Pure PyTorch implementation of RMSNorm."""

    @staticmethod
    def forward(
        x: torch.Tensor,
        weight: torch.Tensor,
        eps: float
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass using native PyTorch operations.

        This implementation does not cache inv_rms since PyTorch autograd
        handles the backward pass automatically.
        """
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
        return x * rms * weight, None

    @staticmethod
    def backward(
        dy: torch.Tensor,
        x: torch.Tensor,
        weight: torch.Tensor,
        inv_rms: Optional[torch.Tensor],
        eps: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Backward pass - not used since PyTorch autograd handles it."""
        raise NotImplementedError(
            "PyTorch backend uses autograd for backward pass. "
            "This method should not be called directly."
        )


class PyTorchBackend(KernelBackend):
    """PyTorch native backend - always available fallback."""

    name = "pytorch"

    # Kernel implementations
    rmsnorm = PyTorchRMSNormKernel

    @classmethod
    def is_available(cls) -> bool:
        """PyTorch backend is always available."""
        return True

    @classmethod
    def supports_device(cls, device: torch.device) -> bool:
        """PyTorch backend supports all devices."""
        return True
