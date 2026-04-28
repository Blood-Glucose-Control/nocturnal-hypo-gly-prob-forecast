# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)

"""
Temporal Fusion Transformer forecaster using AutoGluon's TimeSeriesPredictor.

TFT uses gated residual networks, LSTM encoder/decoders, and interpretable
multi-head attention. It supports variable-selection networks for weighting
input features, making it one of the strongest deep-learning baselines for
structured time series with covariates.

In our nocturnal CGM setting, IOB is included as a past-observed feature
(not a known future covariate — future IOB is unknowable at midnight).
Quantile regression is used for probabilistic output.

GPU memory: ~8 GB peak at the default configuration → 2 workers per 96 GB
Blackwell GPU is safe.
"""

import logging

from src.models.autogluon_base import AutoGluonBaseModel
from src.models.base.registry import ModelRegistry
from src.utils.logging_helper import info_print

from .config import TFTConfig

logger = logging.getLogger(__name__)


@ModelRegistry.register("tft")
class TFTForecaster(AutoGluonBaseModel):
    """Temporal Fusion Transformer forecaster using AutoGluon backend.

    Natively probabilistic attention-based model. Inherits all data prep,
    training, inference, and checkpoint logic from AutoGluonBaseModel.
    """

    config_class = TFTConfig
    config: TFTConfig

    _PREDICTOR_JSON_NAME = "tft_predictor.json"

    @property
    def supports_zero_shot(self) -> bool:
        return False

    def _train_model_info_log(self) -> None:
        config = self.config
        cov_str = (
            f", covariates: {config.covariate_cols}" if config.covariate_cols else ""
        )
        info_print(
            f"Starting TFTForecaster training: "
            f"context={config.context_length}, "
            f"hidden={config.hidden_dim}, heads={config.num_heads}, "
            f"batch={config.batch_size}, lr={config.lr}"
            f"{cov_str}"
        )
