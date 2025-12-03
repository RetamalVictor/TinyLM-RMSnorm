"""Gated MLP (SwiGLU-style) implementation."""

from typing import Optional

import torch
import torch.nn as nn

from tinylm.components.activations import get_activation_fn
from tinylm.components.registry import MLP_REGISTRY
from tinylm.quant import QuantConfig, make_linear


@MLP_REGISTRY.register("gated")
class GatedMLP(nn.Module):
    """Gated MLP block (SwiGLU-style).

    Structure: output = Act(xW_gate) * (xW_up), then output @ W_down

    The gating mechanism improves gradient flow and expressiveness.

    Used by: LLaMA, Mistral
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        hidden_ratio: float = 4.0,
        activation: str = "silu",
        dropout: float = 0.0,
        bias: bool = False,
        quant_config: Optional[QuantConfig] = None,
    ):
        super().__init__()
        self.dim = dim

        # For gated MLPs, hidden dim is typically 2/3 of what it would be
        # to match parameter count with standard MLP
        if hidden_dim is None:
            hidden_dim = int(dim * hidden_ratio * 2 / 3)
            # Round to nearest multiple of 256 for efficiency
            hidden_dim = 256 * ((hidden_dim + 255) // 256)

        self._hidden_dim = hidden_dim

        # Three projections: gate, up, down
        self.w_gate = make_linear(
            dim, hidden_dim, bias=bias,
            quant_config=quant_config, layer_type="mlp"
        )
        self.w_up = make_linear(
            dim, hidden_dim, bias=bias,
            quant_config=quant_config, layer_type="mlp"
        )
        self.w_down = make_linear(
            hidden_dim, dim, bias=bias,
            quant_config=quant_config, layer_type="mlp"
        )

        self.act_fn = get_activation_fn(activation)
        self.dropout = nn.Dropout(dropout)

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    @property
    def is_gated(self) -> bool:
        return True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Gated activation: act(gate) * up
        gate = self.act_fn(self.w_gate(x))
        up = self.w_up(x)
        hidden = gate * up
        hidden = self.dropout(hidden)
        return self.w_down(hidden)
