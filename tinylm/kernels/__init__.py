"""Kernel backend registry for TinyLM.

This module provides a unified interface for selecting and managing
kernel backends (PyTorch, CUDA, Triton) with automatic fallback.

Usage:
    from tinylm.kernels import set_backend, get_backend, get_kernel

    # Set global backend
    set_backend("cuda")  # or "triton", "pytorch", "auto"

    # Get current backend
    backend = get_backend()

    # Get kernel for specific operation
    rmsnorm_kernel = get_kernel("rmsnorm")
"""

from typing import List, Literal, Optional, Type

import torch

from tinylm.components.registry import BaseRegistry
from tinylm.kernels.backends.cuda import CUDABackend
from tinylm.kernels.backends.pytorch import PyTorchBackend
from tinylm.kernels.backends.triton import TritonBackend
from tinylm.kernels.base import KernelBackend, RMSNormKernel

__all__ = [
    "KernelBackend",
    "RMSNormKernel",
    "PyTorchBackend",
    "CUDABackend",
    "TritonBackend",
    "set_backend",
    "get_backend",
    "get_kernel",
    "available_backends",
    "BACKEND_REGISTRY",
]

# Type alias for backend names
BackendName = Literal["auto", "cuda", "triton", "pytorch"]

# Default fallback chain: CUDA -> Triton -> PyTorch
DEFAULT_FALLBACK_CHAIN: List[str] = ["cuda", "triton", "pytorch"]


class KernelBackendRegistry(BaseRegistry[KernelBackend]):
    """Registry for kernel backends with fallback support.

    Extends BaseRegistry with kernel-specific functionality:
    - Global backend selection
    - Automatic fallback chain (CUDA -> Triton -> PyTorch)
    - Per-device backend resolution
    - Runtime backend switching
    """

    def __init__(self):
        super().__init__("kernel_backend")
        self._current_backend: Optional[str] = None
        self._fallback_chain: List[str] = DEFAULT_FALLBACK_CHAIN.copy()

    def register_backend(self, name: str, backend_cls: Type[KernelBackend]) -> None:
        """Register a backend implementation (non-decorator form).

        Args:
            name: Backend name.
            backend_cls: Backend class to register.
        """
        if name in self._registry:
            raise ValueError(f"Backend '{name}' already registered")
        self._registry[name] = backend_cls

    def set_backend(self, name: str) -> None:
        """Set the active backend.

        Args:
            name: Backend name ("cuda", "triton", "pytorch") or "auto"
                  for automatic selection based on fallback chain.

        Raises:
            ValueError: If backend is not registered or not available.
        """
        if name == "auto":
            self._current_backend = None
            return

        if name not in self._registry:
            raise ValueError(
                f"Unknown backend: '{name}'. "
                f"Available: {self.available()}"
            )

        backend_cls = self._registry[name]
        if not backend_cls.is_available():
            raise ValueError(
                f"Backend '{name}' is registered but not available. "
                f"Check that required dependencies are installed."
            )

        self._current_backend = name

    def get_backend(self, device: Optional[torch.device] = None) -> Type[KernelBackend]:
        """Get the active backend class.

        Args:
            device: Optional device to check compatibility. If None, returns
                   the globally set backend or uses fallback chain.

        Returns:
            The backend class to use.

        Raises:
            RuntimeError: If no suitable backend is available.
        """
        # If a specific backend is set and supports the device, use it
        if self._current_backend is not None:
            backend_cls = self._registry[self._current_backend]
            if device is None or backend_cls.supports_device(device):
                return backend_cls

        # Otherwise, use fallback chain
        return self._resolve_backend(device)

    def _resolve_backend(self, device: Optional[torch.device]) -> Type[KernelBackend]:
        """Resolve backend using fallback chain."""
        for backend_name in self._fallback_chain:
            if backend_name not in self._registry:
                continue
            backend_cls = self._registry[backend_name]
            if backend_cls.is_available():
                if device is None or backend_cls.supports_device(device):
                    return backend_cls

        raise RuntimeError(
            f"No suitable backend available. "
            f"Tried: {self._fallback_chain}"
        )

    def get_kernel(
        self,
        operation: str,
        device: Optional[torch.device] = None
    ) -> Type:
        """Get the kernel class for a specific operation.

        Args:
            operation: Name of the operation (e.g., "rmsnorm")
            device: Optional device to check compatibility.

        Returns:
            The kernel class for the operation.

        Raises:
            ValueError: If operation is not supported.
        """
        backend_cls = self.get_backend(device)
        kernel = getattr(backend_cls, operation, None)
        if kernel is None:
            raise ValueError(
                f"Operation '{operation}' not supported by backend '{backend_cls.name}'"
            )
        return kernel

    def available_and_ready(self) -> List[str]:
        """List backends that are registered and available."""
        return [
            name for name, cls in self._registry.items()
            if cls.is_available()
        ]

    def get_current_backend_name(self) -> str:
        """Get the name of the currently selected backend."""
        if self._current_backend is not None:
            return self._current_backend
        return "auto"

    def set_fallback_chain(self, chain: List[str]) -> None:
        """Set custom fallback chain.

        Args:
            chain: List of backend names in priority order.
        """
        for name in chain:
            if name not in self._registry:
                raise ValueError(f"Unknown backend in chain: '{name}'")
        self._fallback_chain = chain.copy()


# Global registry instance
BACKEND_REGISTRY = KernelBackendRegistry()

# Register default backends
BACKEND_REGISTRY.register_backend("pytorch", PyTorchBackend)
BACKEND_REGISTRY.register_backend("cuda", CUDABackend)
BACKEND_REGISTRY.register_backend("triton", TritonBackend)


# Convenience functions for global backend management
def set_backend(name: BackendName) -> None:
    """Set the global kernel backend.

    Args:
        name: Backend name ("cuda", "triton", "pytorch") or "auto"
              for automatic selection.

    Example:
        >>> from tinylm.kernels import set_backend
        >>> set_backend("cuda")  # Use CUDA kernels
        >>> set_backend("auto")  # Auto-select best available
    """
    BACKEND_REGISTRY.set_backend(name)


def get_backend(device: Optional[torch.device] = None) -> Type[KernelBackend]:
    """Get the current kernel backend class.

    Args:
        device: Optional device to check compatibility.

    Returns:
        The backend class to use.

    Example:
        >>> from tinylm.kernels import get_backend
        >>> backend = get_backend()
        >>> print(backend.name)  # "cuda", "triton", or "pytorch"
    """
    return BACKEND_REGISTRY.get_backend(device)


def get_kernel(
    operation: str,
    device: Optional[torch.device] = None
) -> Type:
    """Get the kernel class for a specific operation.

    Args:
        operation: Name of the operation (e.g., "rmsnorm")
        device: Optional device to check compatibility.

    Returns:
        The kernel class for the operation.

    Example:
        >>> from tinylm.kernels import get_kernel
        >>> RMSNormKernel = get_kernel("rmsnorm")
        >>> y, inv_rms = RMSNormKernel.forward(x, weight, eps)
    """
    return BACKEND_REGISTRY.get_kernel(operation, device)


def available_backends() -> List[str]:
    """List all available (ready to use) backends.

    Returns:
        List of backend names that are available on this system.

    Example:
        >>> from tinylm.kernels import available_backends
        >>> print(available_backends())  # ["cuda", "pytorch"]
    """
    return BACKEND_REGISTRY.available_and_ready()
