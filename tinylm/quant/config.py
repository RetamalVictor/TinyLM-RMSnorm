"""Configuration for quantization integration."""

from dataclasses import dataclass
from typing import Literal

# Type alias for supported quantization methods
QuantMethodType = Literal["none", "ternary"]


@dataclass
class QuantConfig:
    """Configuration for quantization.

    Attributes:
        enabled: Whether to enable quantization.
        method: Quantization method to use. One of:
            - "none": No quantization (standard linear)
            - "ternary": Ternary quantization via BitTorch
        threshold_factor: Factor for ternary quantization threshold (default: 0.05).
        per_channel: If True, use per-channel scaling (default: True).
        backend: Backend to use for forward pass ("auto", "cuda", or "python").
        quantize_attention: Whether to quantize attention layers (qkv, proj).
        quantize_mlp: Whether to quantize MLP layers (fc1, fc2).
        quantize_head: Whether to quantize output head (usually False for stability).

    Example:
        >>> # Ternary quantization for attention and MLP
        >>> config = QuantConfig(enabled=True, method="ternary")

        >>> # Selective quantization (attention only)
        >>> config = QuantConfig(
        ...     enabled=True,
        ...     method="ternary",
        ...     quantize_mlp=False,
        ...     quantize_head=False
        ... )
    """
    enabled: bool = False
    method: QuantMethodType = "none"
    threshold_factor: float = 0.05
    per_channel: bool = True
    backend: Literal["auto", "cuda", "python"] = "auto"
    # Layer group controls
    quantize_attention: bool = True
    quantize_mlp: bool = True
    quantize_head: bool = False  # Keep head in FP32 by default for stability

    def to_dict(self) -> dict:
        """Convert config to dictionary for serialization."""
        return {
            "enabled": self.enabled,
            "method": self.method,
            "threshold_factor": self.threshold_factor,
            "per_channel": self.per_channel,
            "backend": self.backend,
            "quantize_attention": self.quantize_attention,
            "quantize_mlp": self.quantize_mlp,
            "quantize_head": self.quantize_head,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "QuantConfig":
        """Create config from dictionary."""
        valid_keys = {
            "enabled", "method", "threshold_factor", "per_channel",
            "backend", "quantize_attention", "quantize_mlp", "quantize_head"
        }
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)

    def __repr__(self) -> str:
        if not self.enabled:
            return "QuantConfig(enabled=False)"
        return (
            f"QuantConfig(enabled={self.enabled}, method={self.method!r}, "
            f"threshold={self.threshold_factor}, backend={self.backend!r}, "
            f"attention={self.quantize_attention}, mlp={self.quantize_mlp}, "
            f"head={self.quantize_head})"
        )
