"""Architecture configuration for TinyLM."""

from dataclasses import asdict, dataclass
from typing import Any, Dict, Literal, Optional


@dataclass
class ArchitectureConfig:
    """Configuration specifying which components an architecture uses.

    This defines the building blocks that make up a transformer architecture,
    allowing TinyLM to mimic different LLM architectures like LLaMA, GPT-2, etc.
    """

    # Architecture name
    name: str = "llama"

    # Normalization
    norm_type: Literal["rmsnorm", "layernorm"] = "rmsnorm"
    norm_position: Literal["pre", "post"] = "pre"
    norm_eps: float = 1e-6

    # Positional embeddings
    pos_emb_type: Literal["rope", "learned"] = "rope"
    rope_base: int = 10000

    # Attention
    attention_type: Literal["mha", "mqa", "gqa"] = "mha"
    attention_op: Literal["standard", "flash", "memory_efficient"] = "standard"
    n_kv_heads: Optional[int] = None  # For MQA/GQA

    # Activation
    activation: Literal["silu", "gelu", "relu"] = "silu"

    # MLP
    mlp_type: Literal["standard", "gated"] = "gated"
    mlp_ratio: float = 4.0

    # Bias settings
    use_bias: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to dictionary for checkpoints."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ArchitectureConfig":
        """Deserialize configuration from dictionary."""
        # Filter to only known fields
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known_fields}
        return cls(**filtered)

    def __repr__(self) -> str:
        return (
            f"ArchitectureConfig(name={self.name!r}, "
            f"norm={self.norm_type}/{self.norm_position}, "
            f"pos_emb={self.pos_emb_type}, "
            f"attn={self.attention_type}, "
            f"mlp={self.mlp_type}/{self.activation})"
        )
