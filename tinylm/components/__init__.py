"""TinyLM components - building blocks for transformer architectures."""

from tinylm.components.activations import (
    GELU,
    ReLU,
    SiLU,
    build_activation,
    get_activation_fn,
)
from tinylm.components.attention import (
    MHA,
    build_attention,
)
from tinylm.components.mlp import (
    GatedMLP,
    StandardMLP,
    build_mlp,
)
from tinylm.components.normalization import (
    LayerNorm,
    RMSNorm,
    build_norm,
)
from tinylm.components.positional import (
    LearnedPositionalEmbedding,
    PositionalContext,
    RoPE,
    build_pos_emb,
)
from tinylm.components.registry import (
    ACTIVATION_REGISTRY,
    ATTENTION_REGISTRY,
    MLP_REGISTRY,
    NORM_REGISTRY,
    POS_EMB_REGISTRY,
    ComponentRegistry,
)

__all__ = [
    # Registry
    "ComponentRegistry",
    "NORM_REGISTRY",
    "POS_EMB_REGISTRY",
    "ATTENTION_REGISTRY",
    "MLP_REGISTRY",
    "ACTIVATION_REGISTRY",
    # Normalization
    "RMSNorm",
    "LayerNorm",
    "build_norm",
    # Positional
    "PositionalContext",
    "RoPE",
    "LearnedPositionalEmbedding",
    "build_pos_emb",
    # Activations
    "SiLU",
    "GELU",
    "ReLU",
    "build_activation",
    "get_activation_fn",
    # MLP
    "StandardMLP",
    "GatedMLP",
    "build_mlp",
    # Attention
    "MHA",
    "build_attention",
]
