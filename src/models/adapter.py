"""ModelAdapter protocol for a stable training/inference/persistence surface.

This protocol defines the contract that pipeline code can rely on regardless of
the underlying model family implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol, TypeVar, runtime_checkable

import numpy as np
import pandas as pd

from src.models.base import ModelConfig

ConfigT = TypeVar("ConfigT", bound=ModelConfig)


@runtime_checkable
class ModelAdapter(Protocol[ConfigT]):
    """Unified model contract for fit/predict/save/load operations."""

    config: ConfigT
    is_fitted: bool

    @property
    def supports_zero_shot(self) -> bool: ...

    @property
    def supports_probabilistic_forecast(self) -> bool: ...

    def fit(
        self,
        train_data: Any,
        output_dir: str = "./output",
        **kwargs: Any,
    ) -> dict[str, Any]: ...

    def predict(
        self,
        data: pd.DataFrame,
        quantile_levels: Optional[list[float]] = None,
        **kwargs: Any,
    ) -> np.ndarray: ...

    def save(
        self,
        model_path: str,
        save_config: bool = True,
        save_metadata: bool = True,
    ) -> None: ...

    @classmethod
    def load(
        cls,
        model_path: str,
        config: Optional[ConfigT] = None,
    ) -> "ModelAdapter[ConfigT]": ...

    def get_model_info(self) -> dict[str, Any]: ...


def assert_model_adapter(adapter: ModelAdapter[ConfigT]) -> ModelAdapter[ConfigT]:
    """Type-level helper documenting adapter-conformant call sites."""
    return adapter


@dataclass(frozen=True)
class AdapterMetadata:
    """Small immutable metadata snapshot used by pipeline orchestration."""

    model_type: str
    supports_zero_shot: bool
    supports_probabilistic_forecast: bool
    is_fitted: bool

    @classmethod
    def from_adapter(cls, adapter: ModelAdapter[ModelConfig]) -> "AdapterMetadata":
        config_model_type = getattr(
            adapter.config, "model_type", adapter.__class__.__name__
        )
        return cls(
            model_type=str(config_model_type),
            supports_zero_shot=adapter.supports_zero_shot,
            supports_probabilistic_forecast=adapter.supports_probabilistic_forecast,
            is_fitted=adapter.is_fitted,
        )
