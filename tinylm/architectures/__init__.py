"""Architecture presets for TinyLM."""

from typing import Dict

from tinylm.architectures.config import ArchitectureConfig

# Architecture presets
ARCHITECTURES: Dict[str, ArchitectureConfig] = {
    "llama": ArchitectureConfig(
        name="llama",
        norm_type="rmsnorm",
        norm_position="pre",
        norm_eps=1e-6,
        pos_emb_type="rope",
        rope_base=10000,
        attention_type="mha",
        activation="silu",
        mlp_type="gated",
        mlp_ratio=4.0,
        use_bias=False,
    ),
    "gpt": ArchitectureConfig(
        name="gpt",
        norm_type="layernorm",
        norm_position="post",
        norm_eps=1e-5,
        pos_emb_type="learned",
        attention_type="mha",
        activation="gelu",
        mlp_type="standard",
        mlp_ratio=4.0,
        use_bias=True,
    ),
}


def get_architecture(name: str) -> ArchitectureConfig:
    """Get architecture configuration by name.

    Args:
        name: Architecture name ("llama", "gpt", etc.)

    Returns:
        ArchitectureConfig for the requested architecture

    Raises:
        ValueError: If architecture is not found
    """
    if name not in ARCHITECTURES:
        available = list(ARCHITECTURES.keys())
        raise ValueError(
            f"Unknown architecture: '{name}'. "
            f"Available: {available}"
        )
    # Return a copy to prevent mutation
    return ArchitectureConfig(**ARCHITECTURES[name].to_dict())


def register_architecture(name: str, config: ArchitectureConfig) -> None:
    """Register a custom architecture.

    Args:
        name: Name for the architecture
        config: Architecture configuration
    """
    ARCHITECTURES[name] = config


def list_architectures() -> list:
    """List available architecture names."""
    return list(ARCHITECTURES.keys())


__all__ = [
    "ArchitectureConfig",
    "ARCHITECTURES",
    "get_architecture",
    "register_architecture",
    "list_architectures",
]
