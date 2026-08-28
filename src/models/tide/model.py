# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)

"""
TiDE forecaster using AutoGluon's TimeSeriesPredictor backend.

TiDE delegates model loading, sliding windows, and training to AutoGluon
internally. The primary model state is self.predictor (TimeSeriesPredictor).

Two separate pipelines exist in this class:

  TRAINING:  flat_df -> patient_dict -> gap-handled segments -> TimeSeriesDataFrame
             -> AutoGluon.fit() with sliding windows over full segments

  INFERENCE: flat_df -> patient_dict -> midnight-anchored episodes -> AutoGluon.predict()
             Each episode is one clinical question: "Given 42h of context ending
             at midnight (with past covariates like iob), forecast BG for the next 6h."
"""

import logging
from typing import Any, Dict, Sequence, Tuple

import numpy as np
import pandas as pd

from ...data.preprocessing.gap_handling import segment_all_patients
from ...utils.logging_helper import info_print, prune_stale_file_handlers
from ..autogluon_data_utils import (
    build_autogluon_context_frame,
    convert_to_patient_dict,
    format_segments_for_autogluon,
)
from ..base import BaseTimeSeriesFoundationModel, TrainingBackend
from ..base.checkpoint_helpers import (
    CHECKPOINT_PATH_KEY,
    resolve_checkpoint_reference,
    write_checkpoint_reference,
)
from ..base.registry import ModelRegistry
from .config import TiDEConfig

logger = logging.getLogger(__name__)


@ModelRegistry.register("tide")
class TiDEForecaster(BaseTimeSeriesFoundationModel):
    """TiDE time series forecaster using AutoGluon backend.

    Implements the BaseTimeSeriesFoundationModel interface for TiDE,
    wrapping AutoGluon's TimeSeriesPredictor for training and inference.

    - Trains from scratch (no pre-trained weights or fine_tune flag)
    - Uses TiDE-specific hyperparameters (encoder/decoder dims, MeanScaler)
    """

    config_class = TiDEConfig
    config: TiDEConfig
    _PREDICTOR_JSON_NAME = "tide_predictor.json"

    def __init__(
        self,
        config: TiDEConfig,
    ):
        # AutoGluon predictor — set before super().__init__() which calls
        # _initialize_model() (our no-op)
        self.predictor = None
        super().__init__(config)

    @property
    def training_backend(self) -> TrainingBackend:
        """Return the training backend for this model family."""
        return TrainingBackend.CUSTOM

    @property
    def supports_zero_shot(self) -> bool:
        """Return whether this model supports zero-shot inference."""
        return False

    @property
    def supports_probabilistic_forecast(self) -> bool:
        """Return whether this model supports probabilistic forecasts."""
        return True

    def _initialize_model(self) -> None:
        """No-op: AutoGluon predictor is created lazily in _train_model
        or _load_checkpoint."""
        pass

    def _prepare_training_data(
        self,
        train_data: Any,
    ) -> Tuple[Any, None, None]:
        """Convert flat DataFrame to AutoGluon TimeSeriesDataFrame.

        Pipeline: flat_df -> patient_dict -> gap-handled segments ->
        TimeSeriesDataFrame with covariates.

        Args:
            train_data: Flat DataFrame from the registry (all patients
                concatenated with patient_col column).

        Returns:
            Tuple of (TimeSeriesDataFrame, None, None). The Nones are
            because the base class signature expects (train, val, test)
            but AutoGluon handles validation internally via sliding windows.
        """
        config = self.config

        patient_dict = convert_to_patient_dict(
            train_data, config.patient_col, config.time_col
        )
        info_print(f"Converted to {len(patient_dict)} patient dicts")

        assert config.min_segment_length is not None
        segments = segment_all_patients(
            patient_dict,
            imputation_threshold_mins=config.imputation_threshold_mins,
            min_segment_length=config.min_segment_length,
            bg_col=config.target_col,
        )
        info_print(f"Gap handling: {len(segments)} segments")

        ts_train = format_segments_for_autogluon(
            segments, config.target_col, config.covariate_cols
        )
        info_print(f"Training data: {ts_train.shape}")

        return (ts_train, None, None)

    def _build_autogluon_frequency(self) -> str:
        """Convert interval minutes to AutoGluon frequency string."""
        return f"{self.config.interval_mins}min"

    def _build_predictor_kwargs(self, output_dir: str) -> Dict[str, Any]:
        """Build TimeSeriesPredictor constructor kwargs for TiDE."""
        config = self.config
        predictor_kwargs: Dict[str, Any] = {
            "prediction_length": config.forecast_length,
            "target": "target",
            "eval_metric": config.eval_metric,
            "freq": self._build_autogluon_frequency(),
            "path": output_dir,
            "quantile_levels": config.quantile_levels or self.DEFAULT_QUANTILE_LEVELS,
        }
        return predictor_kwargs

    def _build_fit_kwargs(self, ts_train: Any) -> Dict[str, Any]:
        """Build fit kwargs for TimeSeriesPredictor.fit()."""
        config = self.config
        fit_kwargs: Dict[str, Any] = {
            "train_data": ts_train,
            "hyperparameters": config.get_autogluon_hyperparameters(),
            "enable_ensemble": config.enable_ensemble,
        }
        if config.time_limit is not None:
            fit_kwargs["time_limit"] = config.time_limit
        return fit_kwargs

    def _log_training_start(self) -> None:
        config = self.config
        info_print(
            f"Starting TiDE training: "
            f"context={config.context_length}, "
            f"hidden_dim={config.encoder_hidden_dim}, "
            f"scaling={config.scaling}"
        )

    def _train_model(
        self,
        train_data: Any,
        output_dir: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """Train TiDE from scratch via AutoGluon's TimeSeriesPredictor.

        The base class fit() passes raw train_data here (not pre-processed)
        for CUSTOM backends. We call _prepare_training_data ourselves.

        Args:
            train_data: Flat DataFrame from the registry.
            output_dir: Directory for AutoGluon to save the predictor.
            **kwargs: Passed through from fit().

        Returns:
            Dict with training metrics.
        """
        from autogluon.timeseries import (  # pyright: ignore[reportMissingImports]
            TimeSeriesPredictor,
        )

        ts_train, _, _ = self._prepare_training_data(train_data)
        predictor_kwargs = self._build_predictor_kwargs(output_dir)
        info_print(
            f"Creating TimeSeriesPredictor at {output_dir} "
            f"with freq={predictor_kwargs['freq']}"
        )
        removed_handlers = prune_stale_file_handlers("autogluon")
        if removed_handlers:
            info_print(
                f"Pruned {removed_handlers} stale AutoGluon file log handler(s) before fit."
            )
        predictor = TimeSeriesPredictor(**predictor_kwargs)
        fit_kwargs = self._build_fit_kwargs(ts_train)
        self._log_training_start()
        predictor.fit(**fit_kwargs)
        self.predictor = predictor

        info_print(f"Training complete. Predictor saved to {predictor.path}")
        return {
            "train_metrics": {
                CHECKPOINT_PATH_KEY: predictor.path,
                "status": "completed",
            }
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _predict_with_context(self, context: pd.DataFrame) -> pd.DataFrame:
        """Run predictor inference on a prebuilt AutoGluon context frame."""
        from autogluon.timeseries import (  # pyright: ignore[reportMissingImports]
            TimeSeriesDataFrame,
        )

        if self.predictor is None:
            raise ValueError("Model must be fitted or loaded before prediction")

        ts_data = TimeSeriesDataFrame(context)
        return self.predictor.predict(ts_data)

    def _collect_batch_predictions(
        self,
        ag_predictions: pd.DataFrame,
        episode_ids: Sequence[str],
        quantile_levels: Sequence[float] | None = None,
    ) -> Dict[str, np.ndarray]:
        """Collect per-episode outputs from AutoGluon predictions."""
        available_items = set(ag_predictions.index.get_level_values(0))
        results: Dict[str, np.ndarray] = {}
        for item_id in episode_ids:
            if item_id not in available_items:
                continue
            ep_preds = self._episode_predictions_frame(ag_predictions, item_id)
            if quantile_levels is not None:
                results[item_id] = self._extract_quantile_predictions(
                    ep_preds, quantile_levels
                )
            else:
                results[item_id] = np.asarray(ep_preds["mean"].to_numpy(), dtype=float)
        return results

    @staticmethod
    def _episode_predictions_frame(
        ag_predictions: pd.DataFrame,
        item_id: str,
    ) -> pd.DataFrame:
        """Return per-item prediction payload as a DataFrame."""
        episode_predictions = ag_predictions.loc[item_id]
        if isinstance(episode_predictions, pd.Series):
            return episode_predictions.to_frame().T
        return episode_predictions

    def _predict(
        self,
        data: pd.DataFrame,
        quantile_levels: Sequence[float] | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Make predictions for a single episode using the fitted predictor.

        Args:
            data: Single-episode DataFrame with target_col (bg_mM) and
                optional covariate columns (e.g. iob). Covariates are
                past-only — included in context, not as future known values.
            quantile_levels: When set, return quantile forecasts as shape
                (len(quantile_levels), forecast_length). Must be a subset of
                the quantile levels the predictor was trained with.
            **kwargs: Unused.

        Returns:
            1D numpy array of predicted BG values for the forecast horizon,
            or shape (len(quantile_levels), forecast_length) when quantile_levels
            is set.
        """
        context = self._build_prediction_context(data, item_id_column=None)
        ag_predictions = self._predict_with_context(context)
        ep_preds = self._episode_predictions_frame(ag_predictions, "ep_0")

        if quantile_levels is not None:
            return self._extract_quantile_predictions(ep_preds, quantile_levels)

        return np.asarray(ep_preds["mean"].to_numpy(), dtype=float)

    def _predict_batch(
        self,
        data: pd.DataFrame,
        episode_col: str,
        quantile_levels: Sequence[float] | None = None,
    ) -> Dict[str, np.ndarray]:
        """Native batch prediction using a single AutoGluon predictor call.

        Packs all episodes into one TimeSeriesDataFrame and calls
        self.predictor.predict() once, which fans out across AutoGluon's
        internal DataLoader.

        Args:
            data: Panel DataFrame containing episode_col with episode IDs,
                target_col (bg_mM), and optional covariate columns.
            episode_col: Column name identifying episodes.

        Returns:
            Dict mapping episode ID (as str) to either:
            - mean forecasts as a 1-D array with shape (forecast_length,), or
            - quantile forecasts as a 2-D array with shape
              (n_quantiles, forecast_length) when quantile_levels is provided.
        """
        context = self._build_prediction_context(data, item_id_column=episode_col)
        ag_predictions = self._predict_with_context(context)
        episode_ids = data[episode_col].astype(str).unique().tolist()
        return self._collect_batch_predictions(
            ag_predictions=ag_predictions,
            episode_ids=episode_ids,
            quantile_levels=quantile_levels,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_checkpoint(self, output_dir: str) -> None:
        """Save predictor path reference.

        AutoGluon auto-saves the full predictor during fit(). This method
        writes a small JSON reference file so _load_checkpoint can locate
        the predictor directory later.
        """
        if self.predictor is not None:
            ref_path = write_checkpoint_reference(
                output_dir=output_dir,
                reference_filename=self._PREDICTOR_JSON_NAME,
                target_path=str(self.predictor.path),
            )
            self.logger.info("Predictor reference saved to %s", ref_path)

    def _load_checkpoint(self, model_dir: str) -> None:
        """Load AutoGluon predictor from directory.

        Checks for a tide_predictor.json reference file first (written
        by _save_checkpoint). Falls back to loading model_dir directly as
        an AutoGluon predictor path.
        """
        from autogluon.timeseries import (  # pyright: ignore[reportMissingImports]
            TimeSeriesPredictor,
        )

        predictor_path = resolve_checkpoint_reference(
            model_dir=model_dir,
            reference_filename=self._PREDICTOR_JSON_NAME,
            required_file="predictor.pkl",
            logger=self.logger,
        )
        self.predictor = TimeSeriesPredictor.load(predictor_path)
        self.is_fitted = True
        self.logger.info("Predictor loaded from %s", predictor_path)

    def _build_prediction_context(
        self,
        data: pd.DataFrame,
        item_id_column: str | None,
    ) -> pd.DataFrame:
        """Build AutoGluon context frame with item/timestamp index."""
        config = self.config
        return build_autogluon_context_frame(
            data,
            target_col=config.target_col,
            time_col=config.time_col,
            item_id_column=item_id_column,
            covariate_cols=config.covariate_cols,
            fill_missing_covariates=True,
            covariate_fill_value=0.0,
        )

    def _extract_quantile_predictions(
        self,
        episode_predictions: pd.DataFrame,
        quantile_levels: Sequence[float],
    ) -> np.ndarray:
        """Return quantile predictions and fail if requested levels are unavailable."""
        available = [float(col) for col in episode_predictions.columns if col != "mean"]
        available_rounded = [round(level, 8) for level in available]
        missing = [
            quantile
            for quantile in quantile_levels
            if round(float(quantile), 8) not in available_rounded
        ]
        if missing:
            raise ValueError(
                f"Quantile levels {missing} not in TiDE predictor "
                f"(available: {sorted(available)}). Retrain with "
                f"DEFAULT_QUANTILE_LEVELS to get all 9 levels."
            )
        return np.stack(
            [
                np.asarray(episode_predictions[str(quantile)].to_numpy(), dtype=float)
                for quantile in quantile_levels
            ],
            axis=0,
        )
