"""TinyLM components - building blocks for transformer architectures."""

from tinylm.components.registry import (
    ComponentRegistry,
    NORM_REGISTRY,
    POS_EMB_REGISTRY,
    ATTENTION_REGISTRY,
    MLP_REGISTRY,
    ACTIVATION_REGISTRY,
)
from tinylm.components.normalization import (
    RMSNorm,
    LayerNorm,
    build_norm,
)
from tinylm.components.positional import (
    PositionalContext,
    RoPE,
    LearnedPositionalEmbedding,
    build_pos_emb,
)
from tinylm.components.activations import (
    SiLU,
    GELU,
    ReLU,
    build_activation,
    get_activation_fn,
)
from tinylm.components.mlp import (
    StandardMLP,
    GatedMLP,
    build_mlp,
)
from tinylm.components.attention import (
    MHA,
    build_attention,
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
