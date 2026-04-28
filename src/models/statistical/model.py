# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)

"""
Statistical forecaster wrapping AutoGluon's AutoARIMA, Theta, and NPTS.

These classical models serve as interpretable baselines above the naive floor.
Key paper-relevant points:

  AutoARIMA — non-seasonal ARIMA with automatic order selection; optionally
    uses IOB as an exogenous regressor. Quantiles are synthesised from
    residuals (not natively probabilistic); expect miscalibrated PIT histograms.

  Theta     — double exponential smoothing with no seasonal decomposition;
    purely univariate. Quantiles also from residuals.

  NPTS      — Nonparametric Time Series model; kernel-based, natively
    probabilistic. Best-calibrated PIT among statistical baselines.

All three models run on CPU (no GPU required). AutoARIMA fitting is capped at
time_limit=7200 s per job; on timeout, AutoGluon substitutes Naive for unfit
series without discarding completed fits.
"""

import logging

from src.models.autogluon_base import AutoGluonBaseModel
from src.models.base.registry import ModelRegistry
from src.utils.logging_helper import info_print

from .config import StatisticalConfig

logger = logging.getLogger(__name__)


@ModelRegistry.register("statistical")
class StatisticalForecaster(AutoGluonBaseModel):
    """Statistical baseline forecaster (AutoARIMA / Theta / NPTS) via AutoGluon.

    Thin subclass of AutoGluonBaseModel. The model variant is selected by
    config.model_name — one class handles all three statistical models.
    """

    config_class = StatisticalConfig
    config: StatisticalConfig

    _PREDICTOR_JSON_NAME = "statistical_predictor.json"

    @property
    def supports_zero_shot(self) -> bool:
        return False

    def _train_model_info_log(self) -> None:
        config = self.config
        cov_note = (
            f", exogenous regressors: {config.covariate_cols}"
            if config.model_name == "AutoARIMA" and config.covariate_cols
            else ""
        )
        time_note = (
            f", time_limit={config.time_limit}s"
            if config.time_limit is not None
            else ""
        )
        info_print(
            f"Starting StatisticalForecaster ({config.model_name}) training: "
            f"forecast={config.forecast_length} steps, "
            f"context={config.context_length} steps"
            f"{cov_note}{time_note}"
        )
