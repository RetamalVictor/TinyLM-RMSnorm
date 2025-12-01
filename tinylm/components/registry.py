"""Component registry system for TinyLM building blocks."""

from typing import TypeVar, Type, Dict, Callable, List
import torch.nn as nn

T = TypeVar('T', bound=nn.Module)


class ComponentRegistry:
    """Registry for component implementations with factory pattern."""

    def __init__(self, name: str):
        self.name = name
        self._registry: Dict[str, Type[nn.Module]] = {}

    def register(self, name: str) -> Callable[[Type[T]], Type[T]]:
        """Decorator to register a component implementation.

        Usage:
            @NORM_REGISTRY.register("rmsnorm")
            class RMSNorm(nn.Module):
                ...
        """
        def decorator(cls: Type[T]) -> Type[T]:
            if name in self._registry:
                raise ValueError(f"{self.name} '{name}' already registered")
            self._registry[name] = cls
            return cls
        return decorator

    def build(self, name: str, **kwargs) -> nn.Module:
        """Build a component by name with given kwargs."""
        if name not in self._registry:
            raise ValueError(
                f"Unknown {self.name}: '{name}'. "
                f"Available: {self.available()}"
            )
        return self._registry[name](**kwargs)

    def get(self, name: str) -> Type[nn.Module]:
        """Get the class for a registered component."""
        if name not in self._registry:
            raise ValueError(
                f"Unknown {self.name}: '{name}'. "
                f"Available: {self.available()}"
            )
        return self._registry[name]

    def available(self) -> List[str]:
        """List available implementations."""
        return list(self._registry.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._registry


# Global registries for each component type
NORM_REGISTRY = ComponentRegistry("normalization")
POS_EMB_REGISTRY = ComponentRegistry("positional_embedding")
ATTENTION_REGISTRY = ComponentRegistry("attention")
MLP_REGISTRY = ComponentRegistry("mlp")
ACTIVATION_REGISTRY = ComponentRegistry("activation")
