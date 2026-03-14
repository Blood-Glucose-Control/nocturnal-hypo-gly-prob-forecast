# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)

"""
ModelRegistry: class-based registry for time series foundation models.

Models self-register via the ``@ModelRegistry.register("name")`` decorator.
The registry allows workflows to resolve a model class by a short name
(e.g. "ttm", "chronos2") without hardcoding if/elif dispatch chains.

Design notes
------------
- Registration is a pure class-level side-effect: no singleton objects,
  no global state beyond the _registry dict.
- The registry does NOT create model instances — callers still construct
  the model-specific config and pass it to the class.  This is intentional:
  every model has a different config type (TTMConfig, Chronos2Config, …),
  and the registry should not paper over those differences.
- ``load_from_checkpoint()`` reads the ``model_type`` field written by
  ``BaseTimeSeriesFoundationModel.save()`` into ``metadata.json`` and
  resolves the class, then delegates to ``cls.load()``.  The ``model_type``
  stored there is the *registry key* (e.g. "ttm"), not the class name.
  For checkpoints saved before the registry existed the lookup falls back
  to a class-name-to-key mapping.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Type

if TYPE_CHECKING:  # pragma: no cover
    from src.models.base.base_model import BaseTimeSeriesFoundationModel


# Fallback: map class names (as stored in old metadata.json) to registry keys.
# Only needed for checkpoints saved before @ModelRegistry.register was added.
_CLASS_NAME_TO_KEY: Dict[str, str] = {
    "TTMForecaster": "ttm",
    "Chronos2Forecaster": "chronos2",
    "TiDEForecaster": "tide",
    "TimesFMForecaster": "timesfm",
    "TimeGradForecaster": "timegrad",
    "TSMixerForecaster": "tsmixer",
    "SundialForecaster": "sundial",
    "MoiraiForecaster": "moirai",
    "MomentForecaster": "moment",
}


class ModelRegistry:
    """Class-based registry mapping short names to model classes.

    Usage
    -----
    Register a model (done once, at import time)::

        @ModelRegistry.register("ttm")
        class TTMForecaster(BaseTimeSeriesFoundationModel):
            ...

    Look up the class in a workflow::

        cls = ModelRegistry.get("ttm")
        config = TTMConfig(...)
        model = cls(config)

    Load from a checkpoint (class resolved from metadata.json)::

        model = ModelRegistry.load_from_checkpoint("/path/to/checkpoint")
    """

    _registry: Dict[str, Type["BaseTimeSeriesFoundationModel"]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator that registers a model class under ``name``.

        Args:
            name: Short identifier for the model (e.g. "ttm", "chronos2").
                  Must be unique within the registry.

        Returns:
            The original class, unmodified (decorator is non-wrapping).

        Raises:
            ValueError: If ``name`` is already registered to a *different* class.
                        Re-registering the same class under the same name is a
                        no-op (safe for reload scenarios in interactive sessions).
        """

        def decorator(model_cls: Type) -> Type:
            existing = cls._registry.get(name)
            if existing is not None and existing is not model_cls:
                raise ValueError(
                    f"Cannot register '{model_cls.__name__}' as '{name}': "
                    f"already registered to '{existing.__name__}'. "
                    f"Use a unique name or unregister the existing entry first."
                )
            cls._registry[name] = model_cls
            return model_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> Type["BaseTimeSeriesFoundationModel"]:
        """Return the model class registered under ``name``.

        Args:
            name: Registry key (e.g. "ttm", "chronos2").

        Returns:
            The model class (not an instance).

        Raises:
            KeyError: If ``name`` is not in the registry.  The error message
                      lists all currently registered names to aid debugging.
        """
        if name not in cls._registry:
            raise KeyError(
                f"Model '{name}' is not registered. "
                f"Available models: {sorted(cls._registry.keys())}"
            )
        return cls._registry[name]

    @classmethod
    def list_models(cls) -> list[str]:
        """Return a sorted list of all registered model names."""
        return sorted(cls._registry.keys())

    @classmethod
    def load_from_checkpoint(
        cls, checkpoint_path: str
    ) -> "BaseTimeSeriesFoundationModel":
        """Load any model from a checkpoint directory.

        Reads ``metadata.json`` inside ``checkpoint_path`` to determine which
        model class to use, then calls ``model_cls.load(checkpoint_path)``.

        The ``metadata.json`` produced by ``BaseTimeSeriesFoundationModel.save()``
        stores ``model_type`` as the class name (e.g. ``"TTMForecaster"``).
        This method accepts either the registry key (e.g. ``"ttm"``) or the
        class name (e.g. ``"TTMForecaster"``) in that field.

        Args:
            checkpoint_path: Path to the directory containing ``metadata.json``
                             and the model checkpoint files.

        Returns:
            A loaded, ready-to-use model instance (``is_fitted`` will reflect
            the state recorded in ``metadata.json``).

        Raises:
            FileNotFoundError: If ``metadata.json`` does not exist at
                               ``checkpoint_path``.
            KeyError: If the model type recorded in ``metadata.json`` is not
                      in the registry (and is not a known class name).
        """
        metadata_path = Path(checkpoint_path) / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"No metadata.json found at '{checkpoint_path}'. "
                f"Ensure the checkpoint was saved with "
                f"BaseTimeSeriesFoundationModel.save()."
            )

        with open(metadata_path) as f:
            metadata = json.load(f)

        raw_type = metadata.get("model_type", "")

        # Try registry key first (forward-compatible), then class-name fallback
        if raw_type in cls._registry:
            model_key = raw_type
        elif raw_type in _CLASS_NAME_TO_KEY:
            model_key = _CLASS_NAME_TO_KEY[raw_type]
        else:
            raise KeyError(
                f"Cannot resolve model type '{raw_type}' from metadata.json "
                f"at '{checkpoint_path}'. "
                f"Registered models: {sorted(cls._registry.keys())}"
            )

        model_cls = cls.get(model_key)
        return model_cls.load(checkpoint_path)
