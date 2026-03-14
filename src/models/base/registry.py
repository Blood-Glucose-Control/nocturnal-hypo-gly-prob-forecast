"""ModelRegistry: maps short names to model classes via decorator."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Dict, List, Type

if TYPE_CHECKING:
    from src.models.base.base_model import BaseTimeSeriesFoundationModel

# Module paths for lazy import — only loaded when get() is called.
_MODEL_MODULES: Dict[str, str] = {
    "ttm": "src.models.ttm",
    "chronos2": "src.models.chronos2",
    "tide": "src.models.tide",
    "sundial": "src.models.sundial",
    "tsmixer": "src.models.tsmixer",
    "timesfm": "src.models.timesfm",
    "timegrad": "src.models.timegrad",
    "moirai": "src.models.moirai",
    "moment": "src.models.moment",
}


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
        # Lazy import: if not yet registered, try importing the module.
        if name not in cls._registry:
            mod_path = _MODEL_MODULES.get(name)
            if mod_path is not None:
                importlib.import_module(mod_path)
        if name not in cls._registry:
            raise KeyError(
                f"Model '{name}' not registered. "
                f"Available: {sorted(cls._registry.keys())}"
            )
        return cls._registry[name]

    @classmethod
    def list_models(cls) -> List[str]:
        return sorted(_MODEL_MODULES.keys())

    @classmethod
    def available_models(cls) -> List[str]:
        """Models whose dependencies are actually installed."""
        result: List[str] = []
        for name, mod_path in _MODEL_MODULES.items():
            if name in cls._registry:
                result.append(name)
                continue
            try:
                importlib.import_module(mod_path)
                if name in cls._registry:
                    result.append(name)
            except ModuleNotFoundError:
                pass
        return sorted(result)
