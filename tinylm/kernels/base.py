"""Base interface for kernel backends."""

from abc import ABC, abstractmethod
from typing import Tuple, Optional
import torch


class KernelBackend(ABC):
    """Abstract base class for kernel backend implementations.

    Each backend (PyTorch, CUDA, Triton) implements this interface
    to provide optimized kernel operations.
    """

    name: str = "base"

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        """Check if this backend is available on the current system."""
        ...

    @classmethod
    @abstractmethod
    def supports_device(cls, device: torch.device) -> bool:
        """Check if this backend supports the given device."""
        ...


class RMSNormKernel(ABC):
    """Abstract interface for RMSNorm kernel implementations."""

    @staticmethod
    @abstractmethod
    def forward(
        x: torch.Tensor,
        weight: torch.Tensor,
        eps: float
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass for RMSNorm.

        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            weight: Learnable scale parameter of shape (dim,)
            eps: Small constant for numerical stability

        Returns:
            Tuple of (output, inv_rms) where inv_rms may be None if not needed
            for backward pass (e.g., PyTorch backend).
        """
        ...

    @staticmethod
    @abstractmethod
    def backward(
        dy: torch.Tensor,
        x: torch.Tensor,
        weight: torch.Tensor,
        inv_rms: Optional[torch.Tensor],
        eps: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Backward pass for RMSNorm.

        Args:
            dy: Gradient of output
            x: Original input tensor
            weight: Learnable scale parameter
            inv_rms: Cached inverse RMS from forward pass (may be None)
            eps: Small constant for numerical stability

        Returns:
            Tuple of (dx, dw) gradients for input and weight.
        """
        ...
