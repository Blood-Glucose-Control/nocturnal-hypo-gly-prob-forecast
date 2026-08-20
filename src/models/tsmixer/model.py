# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)

"""
TSMixer forecaster backed by Darts' TSMixerModel.

This implementation is intentionally thin and delegates shared training/data
plumbing to DartsGlobalModelBase so additional Darts-backed models can be added
with the same integration pattern as AutoGluon-backed models.
"""

from __future__ import annotations

import logging
from typing import Any

from src.models.base.registry import ModelRegistry
from src.models.darts_base import DartsGlobalModelBase
from src.utils.logging_helper import info_print

from .config import TSMixerConfig

logger = logging.getLogger(__name__)


@ModelRegistry.register("tsmixer")
class TSMixerForecaster(DartsGlobalModelBase):
    """TSMixer forecaster using Darts' TSMixerModel runtime."""

    config_class = TSMixerConfig
    config: TSMixerConfig
    _DARTS_MODEL_FILENAME = "tsmixer_darts_model.pkl"

    @property
    def supports_zero_shot(self) -> bool:
        return False

    @property
    def supports_probabilistic_forecast(self) -> bool:
        return True

    def _create_darts_model(self) -> Any:
        try:
            from darts.models import TSMixerModel  # type: ignore[import-not-found]
            from darts.utils.likelihood_models import (  # type: ignore[import-not-found]
                QuantileRegression,
            )
        except ImportError as exc:
            raise ImportError(
                "TSMixer requires Darts dependencies. Install with: "
                "source scripts/setup_model_env.sh tsmixer"
            ) from exc

        trainer_kwargs: dict[str, Any] = {"enable_progress_bar": False}
        if self.config.use_cpu:
            trainer_kwargs.update({"accelerator": "cpu", "devices": 1})

        quantile_levels = self.config.quantile_levels or self.DEFAULT_QUANTILE_LEVELS
        return TSMixerModel(
            input_chunk_length=self.config.context_length,
            output_chunk_length=self.config.forecast_length,
            hidden_size=self.config.hidden_size,
            ff_size=self.config.ff_size,
            num_blocks=self.config.num_blocks,
            activation=self.config.activation,
            dropout=self.config.dropout,
            norm_type=self.config.norm_type,
            normalize_before=self.config.normalize_before,
            use_static_covariates=self.config.use_static_covariates,
            batch_size=self.config.batch_size,
            n_epochs=self.config.num_epochs,
            optimizer_kwargs={"lr": self.config.learning_rate},
            likelihood=QuantileRegression(quantiles=list(quantile_levels)),
            random_state=self.config.random_state,
            pl_trainer_kwargs=trainer_kwargs,
        )

    def _load_darts_model(self, model_path: str) -> Any:
        try:
            from darts.models import TSMixerModel  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "TSMixer requires Darts dependencies. Install with: "
                "source scripts/setup_model_env.sh tsmixer"
            ) from exc
        return TSMixerModel.load(model_path)

    def _train_model_info_log(self) -> None:
        cov_str = (
            f", covariates: {self.config.covariate_cols}"
            if self.config.covariate_cols
            else ""
        )
        q_str = f", quantiles: {self.config.quantile_levels}"
        info_print(
            f"Starting TSMixer training: "
            f"context={self.config.context_length}, "
            f"forecast={self.config.forecast_length}, "
            f"hidden={self.config.hidden_size}, ff={self.config.ff_size}, "
            f"blocks={self.config.num_blocks}, batch={self.config.batch_size}, "
            f"epochs={self.config.num_epochs}, lr={self.config.learning_rate}"
            f"{cov_str}{q_str}"
        )
