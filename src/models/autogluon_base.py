# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)

"""
Shared AutoGluon base class for all AutoGluon TimeSeriesPredictor-backed models.

This module provides AutoGluonBaseModel, which lifts the shared data-preparation,
training, checkpoint, and inference logic out of individual AutoGluon-backed models
(Naive, Statistical, DeepAR, PatchTST, TFT, etc.) so each subclass only needs to
implement get_autogluon_hyperparameters() and a few properties.

NOT used by Chronos2Forecaster or TiDEForecaster, which pre-date this base class and
carry model-specific logic (LoRA shadow checkpoints, zero-shot pipeline, architecture
constraints). Those classes may migrate to this base in a future cleanup PR.

Data pipeline (training):
    flat_df -> patient_dict -> gap-handled segments -> TimeSeriesDataFrame
    -> TimeSeriesPredictor.fit()

Data pipeline (inference):
    flat_df (episode context) -> TimeSeriesDataFrame -> predictor.predict()
    -> np.ndarray (mean or quantile forecasts)
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.data.preprocessing.gap_handling import segment_all_patients
from src.models.base import BaseTimeSeriesFoundationModel, TrainingBackend
from src.models.chronos2.utils import (
    convert_to_patient_dict,
    format_segments_for_autogluon,
)
from src.utils.logging_helper import info_print

logger = logging.getLogger(__name__)


class AutoGluonBaseModel(BaseTimeSeriesFoundationModel):
    """Shared base for AutoGluon TimeSeriesPredictor-backed forecasting models.

    Subclasses must implement:
        - training_backend  (should return TrainingBackend.CUSTOM)
        - supports_lora     (return False; AutoGluon manages this internally)
        - supports_zero_shot
        - config_class      (the Config dataclass for this model)

    Subclasses may override:
        - supports_probabilistic_forecast (default True here)
        - _train_model_info_log()  to emit a model-specific training banner

    The config object must expose:
        - forecast_length, context_length, interval_mins, eval_metric
        - target_col, patient_col, time_col, covariate_cols
        - quantile_levels (Optional[List[float]])
        - enable_ensemble (bool)
        - time_limit (Optional[int])
        - min_segment_length (Optional[int])  — set in __post_init__
        - imputation_threshold_mins (int)
        - get_autogluon_hyperparameters() -> Dict
    """

    # Subclasses set this to their predictor JSON file name so save/load works
    # without further overrides. E.g. "deepar_predictor.json".
    _PREDICTOR_JSON_NAME: str = "ag_predictor.json"

    def __init__(self, config, lora_config=None, distributed_config=None):
        # AutoGluon predictor — must be set before super().__init__() which
        # calls _initialize_model() (our no-op).
        self.predictor: Optional[Any] = None
        super().__init__(config, lora_config, distributed_config)

    # ------------------------------------------------------------------
    # Abstract properties — subclasses must implement
    # ------------------------------------------------------------------

    @property
    def training_backend(self) -> TrainingBackend:
        return TrainingBackend.CUSTOM

    @property
    def supports_lora(self) -> bool:
        return False

    @property
    def supports_zero_shot(self) -> bool:
        return False

    @property
    def supports_probabilistic_forecast(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _initialize_model(self) -> None:
        """No-op: AutoGluon predictor is created lazily in _train_model."""
        pass

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

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
            Tuple of (TimeSeriesDataFrame, None, None). The Nones satisfy the
            base-class signature (train, val, test); AutoGluon handles
            validation internally via sliding windows.
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

    def _train_model_info_log(self) -> None:
        """Override to emit a model-specific training-start banner."""
        config = self.config
        info_print(
            f"Starting {self.__class__.__name__} training: "
            f"context={config.context_length}, "
            f"forecast={config.forecast_length}"
        )

    def _train_model(
        self,
        train_data: Any,
        output_dir: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """Train via AutoGluon's TimeSeriesPredictor.

        The base class fit() passes raw train_data here (not pre-processed)
        for CUSTOM backends. We call _prepare_training_data ourselves.

        Args:
            train_data: Flat DataFrame from the registry.
            output_dir: Directory for AutoGluon to save the predictor.
            **kwargs: Passed through from fit().

        Returns:
            Dict with training metrics.
        """
        from autogluon.timeseries import TimeSeriesPredictor  # type: ignore[import-not-found]

        config = self.config
        ts_train, _, _ = self._prepare_training_data(train_data)

        freq = f"{config.interval_mins}min"
        info_print(f"Creating TimeSeriesPredictor at {output_dir} (freq={freq})")

        predictor_kwargs: Dict[str, Any] = dict(
            prediction_length=config.forecast_length,
            target="target",
            # Covariates are past-only context (not known future values).
            # Setting them as known_covariates_names would require providing
            # future IOB/COB at inference time, which constitutes data leakage.
            freq=freq,
            eval_metric=config.eval_metric,
            path=output_dir,
        )
        if config.quantile_levels is not None:
            predictor_kwargs["quantile_levels"] = config.quantile_levels

        predictor = TimeSeriesPredictor(**predictor_kwargs)

        fit_kwargs: Dict[str, Any] = {
            "train_data": ts_train,
            "hyperparameters": config.get_autogluon_hyperparameters(),
            "enable_ensemble": config.enable_ensemble,
        }
        if config.time_limit is not None:
            fit_kwargs["time_limit"] = config.time_limit

        self._train_model_info_log()
        predictor.fit(**fit_kwargs)
        self.predictor = predictor

        info_print(f"Training complete. Predictor saved to {predictor.path}")
        return {
            "train_metrics": {"status": "completed", "predictor_path": predictor.path}
        }

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def _build_ts_frame(self, data: pd.DataFrame, item_id: str = "ep_0") -> Any:
        """Format a single-episode DataFrame into a TimeSeriesDataFrame.

        Args:
            data: Single-episode DataFrame with target_col and optional
                covariate columns.
            item_id: AutoGluon item_id label to assign.

        Returns:
            TimeSeriesDataFrame ready for predictor.predict().
        """
        from autogluon.timeseries import TimeSeriesDataFrame  # type: ignore[import-not-found]

        config = self.config
        context = data.copy()
        context["item_id"] = item_id

        if config.time_col in context.columns:
            context["timestamp"] = pd.to_datetime(context[config.time_col])
        else:
            context["timestamp"] = context.index

        context = context.rename(columns={config.target_col: "target"})

        for cov_col in config.covariate_cols:
            if cov_col not in context.columns:
                logger.warning(
                    "Covariate '%s' missing from input; filling with zeros", cov_col
                )
                context[cov_col] = 0.0

        keep_cols = ["item_id", "timestamp", "target"] + list(config.covariate_cols)
        keep_cols = [c for c in keep_cols if c in context.columns]
        context = context[keep_cols].set_index(["item_id", "timestamp"])
        return TimeSeriesDataFrame(context)

    def _extract_predictions(
        self,
        ag_predictions: Any,
        item_id: str,
        quantile_levels: Optional[List[float]],
    ) -> np.ndarray:
        """Extract mean or quantile forecasts for one item from AG output.

        Args:
            ag_predictions: DataFrame returned by predictor.predict().
            item_id: The item_id to extract.
            quantile_levels: If None, return mean; otherwise stack quantiles.

        Returns:
            1-D array (forecast_length,) for mean, or 2-D array
            (n_quantiles, forecast_length) for quantiles.
        """
        ep_preds = ag_predictions.loc[item_id]

        if quantile_levels is None:
            return ep_preds["mean"].values

        available = [float(c) for c in ep_preds.columns if c != "mean"]
        missing = [
            q
            for q in quantile_levels
            if round(q, 8) not in [round(a, 8) for a in available]
        ]
        if missing:
            raise ValueError(
                f"Quantile levels {missing} not available in predictor "
                f"(available: {sorted(available)}). "
                f"Retrain with DEFAULT_QUANTILE_LEVELS to include all 9 levels."
            )
        return np.stack([ep_preds[str(q)].values for q in quantile_levels], axis=0)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _predict(
        self,
        data: pd.DataFrame,
        quantile_levels: Optional[List[float]] = None,
        **kwargs,
    ) -> np.ndarray:
        """Make predictions for a single episode.

        Args:
            data: Single-episode DataFrame with target_col and optional
                covariate columns.
            quantile_levels: When set, return quantile forecasts of shape
                (len(quantile_levels), forecast_length).

        Returns:
            1-D mean forecast array, or 2-D quantile array.
        """
        if self.predictor is None:
            raise ValueError("Model must be fitted or loaded before prediction.")

        ts_data = self._build_ts_frame(data, item_id="ep_0")
        ag_predictions = self.predictor.predict(ts_data)
        return self._extract_predictions(ag_predictions, "ep_0", quantile_levels)

    def _predict_batch(
        self,
        data: pd.DataFrame,
        episode_col: str,
        quantile_levels: Optional[List[float]] = None,
    ) -> Dict[str, np.ndarray]:
        """Native batch prediction via a single AutoGluon predictor call.

        Packs all episodes into one TimeSeriesDataFrame and calls
        predictor.predict() once, which fans out across AutoGluon's internal
        DataLoader for efficient GPU/CPU utilization.

        Args:
            data: Panel DataFrame with episode_col, target_col, and optional
                covariate columns.
            episode_col: Column identifying episodes.
            quantile_levels: When set, each result is a 2-D quantile array.

        Returns:
            Dict mapping episode_id (str) to 1-D mean array or 2-D quantile
            array per episode.
        """
        from autogluon.timeseries import TimeSeriesDataFrame  # type: ignore[import-not-found]

        if self.predictor is None:
            raise ValueError("Model must be fitted or loaded before prediction.")

        config = self.config
        context = data.copy()
        context["item_id"] = context[episode_col].astype(str)

        if config.time_col in context.columns:
            context["timestamp"] = pd.to_datetime(context[config.time_col])
        else:
            context["timestamp"] = context.index

        context = context.rename(columns={config.target_col: "target"})

        for cov_col in config.covariate_cols:
            if cov_col not in context.columns:
                logger.warning(
                    "Covariate '%s' missing from input; filling with zeros", cov_col
                )
                context[cov_col] = 0.0

        keep_cols = ["item_id", "timestamp", "target"] + list(config.covariate_cols)
        keep_cols = [c for c in keep_cols if c in context.columns]
        context = context[keep_cols].set_index(["item_id", "timestamp"])
        ts_data = TimeSeriesDataFrame(context)

        ag_predictions = self.predictor.predict(ts_data)

        episode_ids = data[episode_col].astype(str).unique().tolist()
        results: Dict[str, np.ndarray] = {}
        for item_id in episode_ids:
            if item_id not in ag_predictions.index.get_level_values(0):
                continue
            results[item_id] = self._extract_predictions(
                ag_predictions, item_id, quantile_levels
            )
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_checkpoint(self, output_dir: str) -> None:
        """Write a JSON reference pointing at the AutoGluon predictor path.

        AutoGluon auto-saves the full predictor during fit(). This method
        writes a small JSON reference file so _load_checkpoint can locate
        the predictor directory even if the artifact tree is moved.
        """
        if self.predictor is not None:
            ref_path = os.path.join(output_dir, self._PREDICTOR_JSON_NAME)
            os.makedirs(output_dir, exist_ok=True)
            with open(ref_path, "w") as f:
                json.dump({"predictor_path": str(self.predictor.path)}, f, indent=2)
            self.logger.info("Predictor reference saved to %s", ref_path)

    def _load_checkpoint(self, model_dir: str) -> None:
        """Load AutoGluon predictor from directory.

        Checks for the JSON reference file first (written by _save_checkpoint).
        Falls back to loading model_dir directly as an AutoGluon predictor.
        """
        from autogluon.timeseries import TimeSeriesPredictor  # type: ignore[import-not-found]

        ref_path = os.path.join(model_dir, self._PREDICTOR_JSON_NAME)
        if os.path.exists(ref_path):
            with open(ref_path) as f:
                predictor_path = json.load(f)["predictor_path"]
            if not os.path.exists(os.path.join(predictor_path, "predictor.pkl")):
                self.logger.warning(
                    "Predictor not found at %s, falling back to %s",
                    predictor_path,
                    model_dir,
                )
                predictor_path = model_dir
            else:
                self.logger.info("Loading predictor from reference: %s", predictor_path)
        else:
            predictor_path = model_dir

        self.predictor = TimeSeriesPredictor.load(predictor_path)
        self.is_fitted = True
        self.logger.info("Predictor loaded from %s", predictor_path)

    @classmethod
    def load(cls, model_path: str, config=None):
        """Load a saved AutoGluon-backed model from *model_path*.

        A default config is used when none is provided because the predictor
        state is stored entirely inside the AutoGluon predictor directory; the
        config values are not needed to reconstruct inference capability.
        """
        if config is None:
            config = cls.config_class()
        instance = cls(config)
        instance._load_checkpoint(model_path)
        return instance
