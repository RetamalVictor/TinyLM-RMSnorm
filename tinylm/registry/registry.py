"""Model registry with YAML storage."""

from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class BenchmarkResults:
    """Results from functional benchmarks."""

    passed: bool
    coherence_score: float  # 0-1, higher is better
    repetition_rate: float  # 0-1, lower is better
    prompt_completion_score: float  # 0-1, higher is better
    decode_tps: Optional[float] = None  # tokens/sec
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelEntry:
    """A registered model entry."""

    name: str
    architecture: str
    checkpoint: str
    created: str  # ISO format timestamp

    # Model info
    params: Optional[int] = None
    dim: Optional[int] = None
    n_layers: Optional[int] = None
    n_heads: Optional[int] = None
    vocab_size: Optional[int] = None

    # Training info
    dataset: Optional[str] = None
    steps: Optional[int] = None
    batch_size: Optional[int] = None
    lr: Optional[float] = None
    val_loss: Optional[float] = None

    # Benchmark results
    benchmarks: Optional[Dict[str, Any]] = None

    # Metadata
    tags: List[str] = field(default_factory=list)
    notes: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelEntry":
        """Create entry from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in asdict(self).items() if v is not None}


class ModelRegistry:
    """Registry for tracking trained models.

    Stores model metadata in a YAML file for human readability
    and git-friendliness (though the data file itself is gitignored).

    Usage:
        registry = ModelRegistry()

        # Add a model
        entry = ModelEntry(
            name="llama-13M-v1",
            architecture="llama",
            checkpoint="outputs/best.pt",
            created=datetime.now().isoformat(),
        )
        registry.add(entry)

        # List models
        for model in registry.list():
            print(model.name, model.checkpoint)

        # Get specific model
        model = registry.get("llama-13M-v1")
    """

    def __init__(self, path: Optional[Path] = None):
        """Initialize registry.

        Args:
            path: Path to registry YAML file. Defaults to models/registry.yaml
        """
        if path is None:
            path = Path(__file__).parent.parent.parent / "models" / "registry.yaml"
        self.path = Path(path)
        self._ensure_exists()

    def _ensure_exists(self) -> None:
        """Ensure registry file exists."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._save({"models": {}})

    def _load(self) -> Dict[str, Any]:
        """Load registry data."""
        with open(self.path) as f:
            data = yaml.safe_load(f) or {}
        return data if "models" in data else {"models": {}}

    def _save(self, data: Dict[str, Any]) -> None:
        """Save registry data."""
        with open(self.path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def add(self, entry: ModelEntry, overwrite: bool = False) -> None:
        """Add a model to the registry.

        Args:
            entry: Model entry to add
            overwrite: If True, overwrite existing entry with same name

        Raises:
            ValueError: If model already exists and overwrite=False
        """
        data = self._load()
        if entry.name in data["models"] and not overwrite:
            raise ValueError(
                f"Model '{entry.name}' already exists. Use overwrite=True to replace."
            )
        data["models"][entry.name] = entry.to_dict()
        self._save(data)

    def get(self, name: str) -> Optional[ModelEntry]:
        """Get a model by name.

        Args:
            name: Model name

        Returns:
            ModelEntry if found, None otherwise
        """
        data = self._load()
        if name in data["models"]:
            return ModelEntry.from_dict(data["models"][name])
        return None

    def remove(self, name: str) -> bool:
        """Remove a model from the registry.

        Args:
            name: Model name

        Returns:
            True if removed, False if not found
        """
        data = self._load()
        if name in data["models"]:
            del data["models"][name]
            self._save(data)
            return True
        return False

    def list(self, tag: Optional[str] = None) -> List[ModelEntry]:
        """List all registered models.

        Args:
            tag: Optional tag to filter by

        Returns:
            List of ModelEntry objects
        """
        data = self._load()
        entries = [ModelEntry.from_dict(m) for m in data["models"].values()]
        if tag:
            entries = [e for e in entries if tag in e.tags]
        return entries

    def list_names(self) -> List[str]:
        """List all model names."""
        return list(self._load()["models"].keys())

    def update_benchmarks(
        self, name: str, benchmarks: Dict[str, Any]
    ) -> None:
        """Update benchmark results for a model.

        Args:
            name: Model name
            benchmarks: Benchmark results dict
        """
        data = self._load()
        if name not in data["models"]:
            raise ValueError(f"Model '{name}' not found")
        data["models"][name]["benchmarks"] = benchmarks
        self._save(data)

    def add_tag(self, name: str, tag: str) -> None:
        """Add a tag to a model.

        Args:
            name: Model name
            tag: Tag to add
        """
        data = self._load()
        if name not in data["models"]:
            raise ValueError(f"Model '{name}' not found")
        tags = data["models"][name].get("tags", [])
        if tag not in tags:
            tags.append(tag)
            data["models"][name]["tags"] = tags
            self._save(data)
