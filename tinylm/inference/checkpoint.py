"""Checkpoint loading utilities."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from tokenizers import Tokenizer

from tinylm.model import TinyLM
from tinylm.quant import QuantConfig
from tinylm.registry import ModelRegistry


@dataclass
class LoadedModel:
    """Container for loaded model and associated data."""

    model: TinyLM
    tokenizer: Tokenizer
    config: Dict[str, Any]
    checkpoint: Dict[str, Any]

    @property
    def params(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


def load_checkpoint(
    path: str,
    device: str = "cuda",
    eval_mode: bool = True,
) -> LoadedModel:
    """Load model from checkpoint.

    Args:
        path: Path to checkpoint file
        device: Device to load model on
        eval_mode: If True, set model to eval mode

    Returns:
        LoadedModel with model, tokenizer, config, and raw checkpoint

    Raises:
        FileNotFoundError: If checkpoint doesn't exist
        ValueError: If checkpoint is missing required data
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(path, map_location="cpu")

    # Load tokenizer
    if "tokenizer" in ckpt:
        tokenizer = Tokenizer.from_str(ckpt["tokenizer"])
    elif "tok" in ckpt:
        tokenizer = Tokenizer.from_str(ckpt["tok"])
    else:
        raise ValueError("Checkpoint missing tokenizer")

    # Load config - handle nested structure from Hydra
    cfg = ckpt.get("config", {})
    if "model" in cfg:
        model_cfg = cfg["model"]
    else:
        model_cfg = cfg if cfg else {}

    # Defaults
    dim = model_cfg.get("dim", 384)
    n_layers = model_cfg.get("n_layers", 6)
    n_heads = model_cfg.get("n_heads", 6)
    vocab_size = tokenizer.get_vocab_size()

    # Load quantization config if present
    quant_config = None
    if "quant_config" in ckpt and ckpt["quant_config"] is not None:
        quant_config = QuantConfig.from_dict(ckpt["quant_config"])

    # Create model
    model = TinyLM(
        vocab_size=vocab_size,
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        dropout=0.0,
        quant_config=quant_config,
    )

    # Load weights
    state_dict = ckpt["model"]
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)

    # Move to device
    model = model.to(device)
    if eval_mode:
        model = model.eval()

    # Flatten config for easy access
    flat_config = {
        "architecture": model_cfg.get("architecture", "llama"),
        "dim": dim,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "vocab_size": vocab_size,
    }

    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        config=flat_config,
        checkpoint=ckpt,
    )


def load_from_registry(
    name: Optional[str] = None,
    tag: Optional[str] = None,
    device: str = "cuda",
    eval_mode: bool = True,
) -> LoadedModel:
    """Load model from registry by name or tag.

    Args:
        name: Model name (exact match)
        tag: Tag to filter by (returns first match)
        device: Device to load model on
        eval_mode: If True, set model to eval mode

    Returns:
        LoadedModel

    Raises:
        ValueError: If no model found or neither name nor tag provided
    """
    if not name and not tag:
        raise ValueError("Must provide either name or tag")

    registry = ModelRegistry()

    if name:
        entry = registry.get(name)
        if not entry:
            available = registry.list_names()
            raise ValueError(
                f"Model '{name}' not found. Available: {available}"
            )
    else:
        models = registry.list(tag=tag)
        if not models:
            raise ValueError(f"No models found with tag '{tag}'")
        # Return first match (most recently added)
        entry = models[0]

    return load_checkpoint(entry.checkpoint, device=device, eval_mode=eval_mode)


def list_models(tag: Optional[str] = None) -> List[str]:
    """List available model names from registry.

    Args:
        tag: Optional tag to filter by

    Returns:
        List of model names
    """
    registry = ModelRegistry()
    if tag:
        return [m.name for m in registry.list(tag=tag)]
    return registry.list_names()
