"""Base classes and registry for quantization methods."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type

import torch.nn as nn
from torch import Tensor

from tinylm.components.registry import BaseRegistry


@dataclass
class QuantParams:
    """Parameters for quantization method configuration.

    Attributes:
        enabled: Whether quantization is enabled.
        threshold_factor: Factor for threshold-based quantization (e.g., ternary).
        per_channel: If True, use per-channel scaling.
        backend: Backend to use ("auto", "cuda", "python").
        bits: Number of bits for integer quantization (4, 8, etc.).
        symmetric: If True, use symmetric quantization.
        group_size: Group size for grouped quantization (-1 for no grouping).
    """
    enabled: bool = True
    threshold_factor: float = 0.05
    per_channel: bool = True
    backend: str = "auto"
    bits: int = 8
    symmetric: bool = True
    group_size: int = -1


class QuantMethod(ABC):
    """Abstract base class for quantization methods.

    Quantization methods define how to create quantized linear layers
    and optionally how to quantize/dequantize weights and activations.
    """

    name: str = "base"

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        """Check if this quantization method is available.

        Returns:
            True if all required dependencies are installed.
        """
        ...

    @classmethod
    @abstractmethod
    def create_linear(
        cls,
        in_features: int,
        out_features: int,
        bias: bool = False,
        params: Optional[QuantParams] = None,
    ) -> nn.Module:
        """Create a linear layer with this quantization method.

        Args:
            in_features: Size of each input sample.
            out_features: Size of each output sample.
            bias: If True, adds a learnable bias.
            params: Quantization parameters.

        Returns:
            Linear module (standard or quantized).
        """
        ...

    @classmethod
    def quantize_weights(
        cls,
        weight: Tensor,
        params: Optional[QuantParams] = None,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """Quantize weights (optional, for post-training quantization).

        Args:
            weight: Full-precision weight tensor.
            params: Quantization parameters.

        Returns:
            Tuple of (quantized_weight, metadata_dict).
        """
        # Default: return weight unchanged
        return weight, {}

    @classmethod
    def dequantize_weights(
        cls,
        quantized_weight: Tensor,
        metadata: Dict[str, Any],
    ) -> Tensor:
        """Dequantize weights back to full precision.

        Args:
            quantized_weight: Quantized weight tensor.
            metadata: Metadata from quantize_weights().

        Returns:
            Full-precision weight tensor.
        """
        # Default: return weight unchanged
        return quantized_weight

    @classmethod
    def quantize_activations(
        cls,
        x: Tensor,
        params: Optional[QuantParams] = None,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """Quantize activations (optional, for dynamic quantization).

        Args:
            x: Full-precision activation tensor.
            params: Quantization parameters.

        Returns:
            Tuple of (quantized_activation, metadata_dict).
        """
        # Default: return activations unchanged
        return x, {}


class QuantRegistry(BaseRegistry[QuantMethod]):
    """Registry for quantization method implementations.

    Extends BaseRegistry with quantization-specific functionality:
    - Sets cls.name during registration
    - create_linear() factory method
    - available_and_ready() for checking availability
    """

    def __init__(self, name: str = "quantization"):
        super().__init__(name)

    def register(self, name: str):
        """Decorator to register a quantization method.

        Also sets cls.name to the registered name.

        Usage:
            @QUANT_REGISTRY.register("ternary")
            class TernaryQuantMethod(QuantMethod):
                ...
        """
        def decorator(cls: Type[QuantMethod]) -> Type[QuantMethod]:
            if name in self._registry:
                raise ValueError(f"{self.name} method '{name}' already registered")
            cls.name = name
            self._registry[name] = cls
            return cls
        return decorator

    def create_linear(
        self,
        method: str,
        in_features: int,
        out_features: int,
        bias: bool = False,
        params: Optional[QuantParams] = None,
    ) -> nn.Module:
        """Create a linear layer using the specified quantization method.

        Args:
            method: Name of quantization method.
            in_features: Size of each input sample.
            out_features: Size of each output sample.
            bias: If True, adds a learnable bias.
            params: Quantization parameters.

        Returns:
            Linear module.

        Raises:
            RuntimeError: If method is not available.
        """
        method_cls = self.get(method)
        if not method_cls.is_available():
            raise RuntimeError(
                f"Quantization method '{method}' is not available. "
                f"Check if required dependencies are installed."
            )
        return method_cls.create_linear(in_features, out_features, bias, params)

    def available_and_ready(self) -> List[str]:
        """List quantization methods that are both registered and available."""
        return [
            name for name, cls in self._registry.items()
            if cls.is_available()
        ]


# Global registry instance
QUANT_REGISTRY = QuantRegistry()
