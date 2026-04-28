# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)

"""
Naive baseline forecaster wrapping AutoGluon's NaiveModel / AverageModel.

These are the absolute performance floors for the nocturnal hypoglycemia
forecasting task. Both models are non-seasonal and non-parametric:

  NaiveModel   — carries the last observed BG value forward for all horizons.
  AverageModel — averages all observed BG values and predicts that constant.

AutoGluon synthesizes quantile forecasts from training residuals; the resulting
PIT histograms will show miscalibration, which is expected and reported in the
paper as a contrast against natively probabilistic models.

No GPU required; training completes in seconds.
"""

import logging

from src.models.autogluon_base import AutoGluonBaseModel
from src.models.base.registry import ModelRegistry
from src.utils.logging_helper import info_print

from .config import NaiveBaselineConfig

logger = logging.getLogger(__name__)


@ModelRegistry.register("naive_baseline")
class NaiveBaselineForecaster(AutoGluonBaseModel):
    """Naive / Average baseline forecaster using AutoGluon backend.

    Thin subclass of AutoGluonBaseModel that plugs in NaiveBaselineConfig
    and a model-specific training banner. All heavy lifting (data prep,
    fit, predict, save/load) is inherited from AutoGluonBaseModel.

    model_name="Naive"   → NaiveModel (carry-forward)
    model_name="Average" → AverageModel (historical mean)
    """

    config_class = NaiveBaselineConfig
    config: NaiveBaselineConfig

    _PREDICTOR_JSON_NAME = "naive_baseline_predictor.json"

    @property
    def supports_zero_shot(self) -> bool:
        # No pretrained weights; predictor must see some data to compute residuals
        return False

    def _train_model_info_log(self) -> None:
        info_print(
            f"Starting NaiveBaselineForecaster ({self.config.model_name}) training: "
            f"forecast={self.config.forecast_length} steps — "
            f"no learning, residuals computed for quantile synthesis"
        )
