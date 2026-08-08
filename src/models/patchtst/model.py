# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)

"""
PatchTST forecaster using AutoGluon's TimeSeriesPredictor backend.

PatchTST tokenizes the input time series into fixed-length patches and
processes them with a standard Transformer encoder. It supports covariate
inputs as additional patch channels (past IOB treated as a past feature).

GPU memory: ~6 GB peak at the default configuration → 2 workers per 96 GB
Blackwell GPU is safe.
"""

import logging

from src.models.autogluon_base import AutoGluonBaseModel
from src.models.base.registry import ModelRegistry
from src.utils.logging_helper import info_print

from .config import PatchTSTConfig

logger = logging.getLogger(__name__)


@ModelRegistry.register("patchtst")
class PatchTSTForecaster(AutoGluonBaseModel):
    """PatchTST time series forecaster using AutoGluon backend.

    Transformer with patch tokenization. Inherits all data prep, training,
    inference, and checkpoint logic from AutoGluonBaseModel.
    """

    config_class = PatchTSTConfig
    config: PatchTSTConfig

    _PREDICTOR_JSON_NAME = "patchtst_predictor.json"

    @property
    def supports_zero_shot(self) -> bool:
        return False

    def _train_model_info_log(self) -> None:
        config = self.config
        n_patches = config.context_length // config.stride
        cov_str = (
            f", covariates: {config.covariate_cols}" if config.covariate_cols else ""
        )
        info_print(
            f"Starting PatchTSTForecaster training: "
            f"context={config.context_length} (≈{n_patches} patches), "
            f"d_model={config.d_model}, nhead={config.nhead}, "
            f"depth={config.num_encoder_layers}, "
            f"batch={config.batch_size}, lr={config.lr}"
            f"{cov_str}"
        )
