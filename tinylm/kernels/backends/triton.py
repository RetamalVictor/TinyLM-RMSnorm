"""Triton kernel backend - JIT-compiled GPU kernels.

This is a stub implementation for Issue 3 (Triton RMSNorm).
Triton provides easier kernel development with competitive performance.
"""

from typing import Optional, Tuple

import torch

from tinylm.kernels.base import KernelBackend, RMSNormKernel

# Check if Triton is available
_triton_available = None


def _check_triton_available() -> bool:
    """Check if Triton is installed and functional."""
    global _triton_available
    if _triton_available is None:
        import importlib.util
        triton_spec = importlib.util.find_spec("triton")
        _triton_available = triton_spec is not None and torch.cuda.is_available()
    return _triton_available


class TritonRMSNormKernel(RMSNormKernel):
    """Triton-based RMSNorm kernel implementation.

    TODO: Implement in Issue 3 (Triton RMSNorm Kernel)
    """

    @staticmethod
    def forward(
        x: torch.Tensor,
        weight: torch.Tensor,
        eps: float
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass using Triton kernel."""
        raise NotImplementedError(
            "Triton RMSNorm kernel not yet implemented. "
            "See Issue 3: Implement Triton RMSNorm Kernel"
        )

    @staticmethod
    def backward(
        dy: torch.Tensor,
        x: torch.Tensor,
        weight: torch.Tensor,
        inv_rms: Optional[torch.Tensor],
        eps: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Backward pass using Triton kernel."""
        raise NotImplementedError(
            "Triton RMSNorm backward not yet implemented. "
            "See Issue 3: Implement Triton RMSNorm Kernel"
        )


class TritonBackend(KernelBackend):
    """Triton kernel backend - JIT-compiled GPU kernels.

    Triton provides:
    - Easier kernel development than CUDA
    - Automatic optimization
    - Good portability across GPU architectures

    Status: Stub implementation. Full implementation in Issue 3.
    """

    name = "triton"

    # Kernel implementations
    rmsnorm = TritonRMSNormKernel

    @classmethod
    def is_available(cls) -> bool:
        """Check if Triton is installed and CUDA is available.

        Note: Returns False until Issue 3 is implemented.
        """
        # TODO: Return actual availability once Issue 3 is complete
        # return _check_triton_available()
        return False  # Disabled until implemented

    @classmethod
    def supports_device(cls, device: torch.device) -> bool:
        """Triton backend only supports CUDA devices."""
        return device.type == "cuda" and cls.is_available()
