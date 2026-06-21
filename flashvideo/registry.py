"""Pluggable component registry for FlashVideo.

Provides a generic ``Registry`` class used to register and look up models,
schedulers, datasets, and task pipelines by string key so they can be
instantiated from YAML configuration files without hard-coded imports.
"""

from __future__ import annotations

from typing import Any, Callable, Dict


class Registry:
    """A named registry that maps string keys to classes or factory functions.

    Usage::

        MODELS = Registry("models")

        @MODELS.register("VideoDiT")
        class VideoDiT(nn.Module):
            ...

        cls = MODELS.get("VideoDiT")
        model = MODELS.build("VideoDiT", in_channels=4)
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._registry: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def register(self, key: str) -> Callable:
        """Decorator that registers *cls_or_fn* under *key*."""

        def _wrap(cls_or_fn: Any) -> Any:
            if key in self._registry:
                raise KeyError(f"[{self.name}] '{key}' is already registered")
            self._registry[key] = cls_or_fn
            return cls_or_fn

        return _wrap

    def register_module(self, key: str, module: Any) -> None:
        """Imperatively register *module* under *key*."""
        if key in self._registry:
            raise KeyError(f"[{self.name}] '{key}' is already registered")
        self._registry[key] = module

    def get(self, key: str) -> Any:
        """Return the registered class / function for *key*."""
        if key not in self._registry:
            available = ", ".join(sorted(self._registry)) or "(empty)"
            raise KeyError(f"[{self.name}] '{key}' not found. Available: {available}")
        return self._registry[key]

    def build(self, key: str, **kwargs: Any) -> Any:
        """Instantiate the registered class for *key* with *kwargs*."""
        cls_or_fn = self.get(key)
        return cls_or_fn(**kwargs)

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def keys(self):
        return self._registry.keys()

    def items(self):
        return self._registry.items()

    def __contains__(self, key: str) -> bool:
        return key in self._registry

    def __len__(self) -> int:
        return len(self._registry)

    def __repr__(self) -> str:
        entries = ", ".join(sorted(self._registry))
        return f"Registry(name={self.name!r}, entries=[{entries}])"


# ======================================================================
# Global registries
# ======================================================================

MODELS = Registry("models")
SCHEDULERS = Registry("schedulers")
DATASETS = Registry("datasets")
TASKS = Registry("tasks")
