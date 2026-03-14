"""ModelRegistry: maps short names to model classes via decorator."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Type

if TYPE_CHECKING:
    from src.models.base.base_model import BaseTimeSeriesFoundationModel


class ModelRegistry:
    """Maps short names (e.g. "ttm", "chronos2") to model classes."""

    _registry: Dict[str, Type["BaseTimeSeriesFoundationModel"]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator: ``@ModelRegistry.register("ttm")``."""

        def decorator(model_cls: Type) -> Type:
            existing = cls._registry.get(name)
            if existing is not None and existing is not model_cls:
                raise ValueError(
                    f"'{name}' already registered to '{existing.__name__}'"
                )
            cls._registry[name] = model_cls
            return model_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> Type["BaseTimeSeriesFoundationModel"]:
        if name not in cls._registry:
            raise KeyError(
                f"Model '{name}' not registered. "
                f"Available: {sorted(cls._registry.keys())}"
            )
        return cls._registry[name]

    @classmethod
    def list_models(cls) -> list[str]:
        return sorted(cls._registry.keys())
