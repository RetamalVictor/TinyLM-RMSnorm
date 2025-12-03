"""Standard MLP (feed-forward) implementation."""

from typing import Optional

import torch
import torch.nn as nn

from tinylm.components.activations import get_activation_fn
from tinylm.components.registry import MLP_REGISTRY
from tinylm.quant import QuantConfig, make_linear


@MLP_REGISTRY.register("standard")
class StandardMLP(nn.Module):
    """Standard feed-forward network.

    Structure: Linear -> Activation -> Dropout -> Linear -> Dropout

    Used by: GPT-2, Falcon
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        hidden_ratio: float = 4.0,
        activation: str = "gelu",
        dropout: float = 0.0,
        bias: bool = True,
        quant_config: Optional[QuantConfig] = None,
    ):
        super().__init__()
        self.dim = dim
        self._hidden_dim = hidden_dim or int(dim * hidden_ratio)

        self.fc_up = make_linear(
            dim, self._hidden_dim, bias=bias,
            quant_config=quant_config, layer_type="mlp"
        )
        self.fc_down = make_linear(
            self._hidden_dim, dim, bias=bias,
            quant_config=quant_config, layer_type="mlp"
        )
        self.act_fn = get_activation_fn(activation)
        self.dropout = nn.Dropout(dropout)

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    @property
    def is_gated(self) -> bool:
        return False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc_up(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc_down(x)
        x = self.dropout(x)
        return x
