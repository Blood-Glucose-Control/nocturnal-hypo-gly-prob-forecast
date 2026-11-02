# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)

"""
GBM forecaster using AutoGluon's tabular time-series backend.

Wraps AutoGluon's ``RecursiveTabular`` / ``DirectTabular`` reductions with a
gradient-boosting engine (LightGBM / CatBoost / XGBoost). The tree backend is a
strong, fast-to-train, CPU-based diversity member for the nocturnal
hypoglycemia forecasting ensemble, and it natively consumes known-future
covariates (insulin + meal timing) and static features.

All data prep, training, inference, and checkpoint logic is inherited from
:class:`AutoGluonBaseModel`; this class only selects the tabular strategy /
backend via :class:`GBMConfig` and emits a model-specific training banner.
"""

import logging

from ...utils.logging_helper import info_print
from ..autogluon_base import AutoGluonBaseModel
from ..base.registry import ModelRegistry
from .config import GBMConfig

logger = logging.getLogger(__name__)


@ModelRegistry.register("gbm")
class GBMForecaster(AutoGluonBaseModel):
    """Gradient-boosted tree time series forecaster using AutoGluon backend.

    Trains from scratch (no pre-trained weights). Inherits all data prep,
    training, inference, and checkpoint logic from AutoGluonBaseModel; the
    LightGBM/CatBoost/XGBoost backend and recursive/direct strategy are chosen
    via the config's hyperparameters.
    """

    config_class = GBMConfig
    config: GBMConfig

    _PREDICTOR_JSON_NAME = "gbm_predictor.json"

    @property
    def supports_zero_shot(self) -> bool:
        """Return whether this model supports zero-shot inference."""
        return False

    def _train_model_info_log(self) -> None:
        config = self.config
        cov_str = (
            f", covariates: {config.covariate_cols}" if config.covariate_cols else ""
        )
        info_print(
            f"Starting GBMForecaster training: "
            f"strategy={config.strategy}, backend={config.backend}, "
            f"context={config.context_length}, forecast={config.forecast_length}"
            f"{cov_str}"
        )
