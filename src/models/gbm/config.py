# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)

"""
GBM (gradient-boosted tree) model configuration.

Wraps AutoGluon's tabular time-series forecasters (``RecursiveTabular`` /
``DirectTabular``) with a gradient-boosting engine (LightGBM / CatBoost /
XGBoost) selected via the underlying tabular ``model_name``.

Both tabular strategies support known-future covariates (e.g. insulin and meal
timing) and static features, but not past covariates -- mirroring the
known-future covariate routing used by the TiDE / DeepAR families.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..base import ModelConfig, TrainingBackend

# Sweep-facing strategy name -> AutoGluon TimeSeries model key.
_STRATEGY_TO_AG_KEY: Dict[str, str] = {
    "recursive": "RecursiveTabular",
    "direct": "DirectTabular",
}

# Sweep-facing backend name -> AutoGluon tabular ``model_name``.
_BACKEND_TO_MODEL_NAME: Dict[str, str] = {
    "lightgbm": "GBM",
    "catboost": "CAT",
    "xgboost": "XGB",
}


@dataclass
class GBMConfig(ModelConfig):
    """Configuration for gradient-boosted tree forecasters via AutoGluon.

    Inherits from :class:`ModelConfig` and adds a searchable ``strategy``
    (recursive/direct tabular reduction) and ``backend`` (LightGBM/CatBoost/
    XGBoost) plus the underlying mlforecast tabular knobs.
    """

    # Override parent defaults
    model_type: str = "gbm"
    forecast_length: int = 24  # +120 min at 5-min cadence (MetaboNet shape)
    context_length: int = 288  # ~24 h of history
    training_backend: TrainingBackend = TrainingBackend.CUSTOM
    training_mode: str = "from_scratch"

    # Tabular reduction strategy + gradient-boosting engine
    strategy: str = "recursive"  # "recursive" | "direct"
    backend: str = "lightgbm"  # "lightgbm" | "catboost" | "xgboost"

    # mlforecast tabular knobs (None -> AutoGluon defaults for the series freq)
    lags: Optional[List[int]] = None
    differences: Optional[List[int]] = None
    target_scaler: Optional[str] = None
    max_num_items: Optional[int] = None
    max_num_samples: Optional[int] = None

    # Underlying tabular model hyperparameters (passed as model_hyperparameters)
    model_hyperparameters: Dict[str, Any] = field(default_factory=dict)

    # Gap handling
    imputation_threshold_mins: int = 45
    min_segment_length: Optional[int] = None

    # Covariates
    covariate_cols: List[str] = field(default_factory=list)
    target_col: str = "bg_mM"
    patient_col: str = "patient_id"
    time_col: str = "datetime"
    interval_mins: int = 5

    # AutoGluon settings
    eval_metric: str = "WQL"
    enable_ensemble: bool = False
    time_limit: Optional[int] = None

    def __post_init__(self) -> None:
        if self.strategy not in _STRATEGY_TO_AG_KEY:
            raise ValueError(
                f"GBM strategy must be one of {sorted(_STRATEGY_TO_AG_KEY)}, "
                f"got {self.strategy!r}"
            )
        if self.backend not in _BACKEND_TO_MODEL_NAME:
            raise ValueError(
                f"GBM backend must be one of {sorted(_BACKEND_TO_MODEL_NAME)}, "
                f"got {self.backend!r}"
            )
        if self.min_segment_length is None:
            self.min_segment_length = self.context_length + self.forecast_length

    @property
    def autogluon_model_key(self) -> str:
        """AutoGluon TimeSeries model key for the selected strategy."""
        return _STRATEGY_TO_AG_KEY[self.strategy]

    def get_autogluon_hyperparameters(self) -> Dict[str, Dict[str, Any]]:
        """Build the hyperparameters dict for ``TimeSeriesPredictor.fit()``.

        Returns:
            Dict keyed by the strategy's AutoGluon model key mapping to the
            tabular ``model_name`` plus optional mlforecast knobs.
        """
        params: Dict[str, Any] = {
            "model_name": _BACKEND_TO_MODEL_NAME[self.backend],
            "model_hyperparameters": dict(self.model_hyperparameters),
        }
        if self.lags is not None:
            params["lags"] = list(self.lags)
        if self.differences is not None:
            params["differences"] = list(self.differences)
        if self.target_scaler is not None:
            params["target_scaler"] = self.target_scaler
        if self.max_num_items is not None:
            params["max_num_items"] = self.max_num_items
        if self.max_num_samples is not None:
            params["max_num_samples"] = self.max_num_samples
        return {self.autogluon_model_key: params}


def create_default_gbm_config(**overrides: Any) -> GBMConfig:
    """Create a :class:`GBMConfig` with validated defaults.

    Args:
        **overrides: Configuration parameters to override.

    Returns:
        GBMConfig instance.
    """
    defaults: dict[str, Any] = {
        "context_length": 288,
        "forecast_length": 24,
    }
    defaults.update(overrides)
    return GBMConfig(**defaults)
