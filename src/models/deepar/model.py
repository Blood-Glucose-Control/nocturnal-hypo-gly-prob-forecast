# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)

"""
DeepAR forecaster using AutoGluon's TimeSeriesPredictor backend.

DeepAR is an autoregressive RNN that models the conditional distribution of
future values given context. It is natively probabilistic (StudentT output head),
making it a strong baseline for our nocturnal hypoglycemia probability estimation.

Unlike Naive/Statistical models, DeepAR benefits from GPU acceleration and
covariate inputs (past IOB included as past time-varying features).

GPU memory: ~4 GB peak at batch_size=128, hidden_size=64 → 2 workers per 96 GB
Blackwell GPU is safe and recommended for the sweep.
"""

import logging

from src.models.autogluon_base import AutoGluonBaseModel
from src.models.base.registry import ModelRegistry
from src.utils.logging_helper import info_print

from .config import DeepARConfig

logger = logging.getLogger(__name__)


@ModelRegistry.register("deepar")
class DeepARForecaster(AutoGluonBaseModel):
    """DeepAR time series forecaster using AutoGluon backend.

    Natively probabilistic RNN model. Inherits all data prep, training,
    inference, and checkpoint logic from AutoGluonBaseModel.
    """

    config_class = DeepARConfig
    config: DeepARConfig

    _PREDICTOR_JSON_NAME = "deepar_predictor.json"

    @property
    def supports_zero_shot(self) -> bool:
        return False

    def _train_model_info_log(self) -> None:
        config = self.config
        cov_str = (
            f", covariates: {config.covariate_cols}" if config.covariate_cols else ""
        )
        info_print(
            f"Starting DeepARForecaster training: "
            f"context={config.context_length}, "
            f"hidden={config.hidden_size}×{config.num_layers} ({config.cell_type.upper()}), "
            f"batch={config.batch_size}, lr={config.lr}"
            f"{cov_str}"
        )
