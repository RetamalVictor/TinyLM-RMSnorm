"""Component registry system for TinyLM building blocks."""

from typing import Callable, Dict, Generic, List, Type, TypeVar

import torch.nn as nn

T = TypeVar('T')


class BaseRegistry(Generic[T]):
    """Generic registry base class for pluggable implementations.

    Provides core registration, lookup, and listing functionality.
    Subclasses can add domain-specific methods (build, create_linear, etc.).

    Type Parameters:
        T: The base type that registered classes must extend.
    """

    def __init__(self, name: str):
        """Initialize registry.

        Args:
            name: Human-readable name for error messages (e.g., "normalization").
        """
        self.name = name
        self._registry: Dict[str, Type[T]] = {}

    def register(self, name: str) -> Callable[[Type[T]], Type[T]]:
        """Decorator to register an implementation.

        Usage:
            @REGISTRY.register("myimpl")
            class MyImpl:
                ...

        Args:
            name: Unique identifier for this implementation.

        Returns:
            Decorator function.

        Raises:
            ValueError: If name is already registered.
        """
        def decorator(cls: Type[T]) -> Type[T]:
            if name in self._registry:
                raise ValueError(f"{self.name} '{name}' already registered")
            self._registry[name] = cls
            return cls
        return decorator

    def get(self, name: str) -> Type[T]:
        """Get the class for a registered implementation.

        Args:
            name: Registered name.

        Returns:
            The registered class.

        Raises:
            ValueError: If name is not registered.
        """
        if name not in self._registry:
            raise ValueError(
                f"Unknown {self.name}: '{name}'. "
                f"Available: {self.available()}"
            )
        return self._registry[name]

    def available(self) -> List[str]:
        """List registered implementation names."""
        return list(self._registry.keys())

    def __contains__(self, name: str) -> bool:
        """Check if a name is registered."""
        return name in self._registry


class ComponentRegistry(BaseRegistry[nn.Module]):
    """Registry for nn.Module components with factory pattern."""

    def build(self, name: str, **kwargs) -> nn.Module:
        """Build a component by name with given kwargs.

        Args:
            name: Registered component name.
            **kwargs: Arguments to pass to component constructor.

        Returns:
            Instantiated component.
        """
        return self.get(name)(**kwargs)


# Global registries for each component type
NORM_REGISTRY = ComponentRegistry("normalization")
POS_EMB_REGISTRY = ComponentRegistry("positional_embedding")
ATTENTION_REGISTRY = ComponentRegistry("attention")
ATTENTION_OP_REGISTRY = ComponentRegistry("attention_op")
MLP_REGISTRY = ComponentRegistry("mlp")
ACTIVATION_REGISTRY = ComponentRegistry("activation")
