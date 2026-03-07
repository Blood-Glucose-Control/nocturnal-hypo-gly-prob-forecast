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

import json
import logging
import os
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from src.data.preprocessing.gap_handling import segment_all_patients
from src.models.base import BaseTimeSeriesFoundationModel, TrainingBackend
from src.utils.logging_helper import info_print

from .config import TiDEConfig

from .utils import (
    convert_to_patient_dict,
    format_segments_for_autogluon,
)

logger = logging.getLogger(__name__)


class TiDEForecaster(BaseTimeSeriesFoundationModel):
    """TiDE time series forecaster using AutoGluon backend.

    Implements the BaseTimeSeriesFoundationModel interface for TiDE,
    wrapping AutoGluon's TimeSeriesPredictor for training and inference.

    - Trains from scratch (no pre-trained weights or fine_tune flag)
    - Uses TiDE-specific hyperparameters (encoder/decoder dims, MeanScaler)
    """

    def __init__(
        self,
        config: TiDEConfig,
        lora_config=None,
        distributed_config=None,
    ):
        # AutoGluon predictor — set before super().__init__() which calls
        # _initialize_model() (our no-op)
        self.predictor = None
        super().__init__(config, lora_config, distributed_config)

    @property
    def training_backend(self) -> TrainingBackend:
        return TrainingBackend.CUSTOM

    @property
    def supports_lora(self) -> bool:
        return False

    @property
    def supports_zero_shot(self) -> bool:
        return False

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
        from autogluon.timeseries import TimeSeriesPredictor

        config = self.config
        ts_train, _, _ = self._prepare_training_data(train_data)

        # Convert interval_mins to frequency string for AutoGluon
        # e.g., 5 -> "5min", 15 -> "15min"
        freq = f"{config.interval_mins}min"

        info_print(f"Creating TimeSeriesPredictor at {output_dir} with freq={freq}")
        predictor = TimeSeriesPredictor(
            prediction_length=config.forecast_length,
            target="target",
            eval_metric=config.eval_metric,
            freq=freq,
            path=output_dir,
        )

        fit_kwargs = {
            "train_data": ts_train,
            "hyperparameters": config.get_autogluon_hyperparameters(),
            "enable_ensemble": config.enable_ensemble,
        }
        if config.time_limit is not None:
            fit_kwargs["time_limit"] = config.time_limit

        info_print(
            f"Starting TiDE training: "
            f"context={config.context_length}, "
            f"hidden_dim={config.encoder_hidden_dim}, "
            f"scaling={config.scaling}"
        )
        predictor.fit(**fit_kwargs)
        self.predictor = predictor

        info_print(f"Training complete. Predictor saved to {predictor.path}")
        return {
            "train_metrics": {"status": "completed", "predictor_path": predictor.path}
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _predict(
        self,
        data: pd.DataFrame,
        **kwargs,
    ) -> np.ndarray:
        """Make predictions using the fitted AutoGluon predictor.

        Accepts either:
        - A panel DataFrame with 'episode_id' column (multiple episodes)
        - A plain context DataFrame (single context window)

        If no 'episode_id' column is present, the entire DataFrame is treated
        as a single episode (synthetic episode_id assigned automatically).

        Args:
            data: DataFrame with target_col (bg_mM) and optionally episode_id,
                datetime, and covariate columns.
            **kwargs: Unused. Covariates (e.g. iob) are past-only — they
                are included in the context data, not as future known values.

        Returns:
            1D numpy array of predicted BG values for the forecast horizon.
        """
        from autogluon.timeseries import TimeSeriesDataFrame

        if self.predictor is None:
            raise ValueError("Model must be fitted or loaded before prediction")

        config = self.config
        context = data.copy()

        # If no episode_id, treat entire DataFrame as a single episode
        if "episode_id" not in context.columns:
            context["episode_id"] = "ep_0"

        # Convert episode_id -> item_id for AutoGluon
        context["item_id"] = context["episode_id"]
        if config.time_col in context.columns:
            context["timestamp"] = pd.to_datetime(context[config.time_col])
        else:
            context["timestamp"] = context.index
        context = context.rename(columns={config.target_col: "target"})

        # Ensure required covariates are present; AutoGluon's saved
        # predictor will expect the same set used during training, even if
        # they are all zero.  If the evaluation dataset doesn't include a
        # covariate (e.g., iob), add a zero-filled column so the predictor
        # doesn't error out.
        for cov_col in config.covariate_cols:
            if cov_col not in context.columns:
                logger.warning(
                    "Covariate '%s' missing from input data; filling with zeros",
                    cov_col,
                )
                context[cov_col] = 0.0

        ag_cols = ["item_id", "timestamp", "target"] + config.covariate_cols
        context = context[ag_cols].set_index(["item_id", "timestamp"])
        ts_data = TimeSeriesDataFrame(context)

        ag_predictions = self.predictor.predict(ts_data)

        # AutoGluon returns MultiIndex (item_id, timestamp) with "mean" column.
        episode_ids = (
            data["episode_id"].unique() if "episode_id" in data.columns else ["ep_0"]
        )
        result_arrays = []
        for episode_id in episode_ids:
            item_id = episode_id
            if item_id in ag_predictions.index.get_level_values(0):
                pred_values = ag_predictions.loc[item_id]["mean"].values
                result_arrays.append(pred_values)

        if not result_arrays:
            return np.array([])

        return np.concatenate(result_arrays)

    def _predict_batch(
        self,
        data: pd.DataFrame,
        episode_col: str,
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
            Dict mapping episode ID (as str) to 1-D numpy forecast array.
        """
        from autogluon.timeseries import TimeSeriesDataFrame

        if self.predictor is None:
            raise ValueError("Model must be fitted or loaded before prediction")

        config = self.config
        context = data.copy()

        context["item_id"] = context[episode_col].astype(str)
        if config.time_col in context.columns:
            context["timestamp"] = pd.to_datetime(context[config.time_col])
        else:
            context["timestamp"] = context.index
        context = context.rename(columns={config.target_col: "target"})

        # Zero-fill missing covariates (predictor expects same columns as training)
        for cov_col in config.covariate_cols:
            if cov_col not in context.columns:
                logger.warning(
                    "Covariate '%s' missing from input data; filling with zeros",
                    cov_col,
                )
                context[cov_col] = 0.0

        ag_cols = ["item_id", "timestamp", "target"] + config.covariate_cols
        ag_cols = [c for c in ag_cols if c in context.columns]
        context = context[ag_cols].set_index(["item_id", "timestamp"])
        ts_data = TimeSeriesDataFrame(context)

        ag_predictions = self.predictor.predict(ts_data)

        episode_ids = [str(eid) for eid in data[episode_col].unique()]
        results: Dict[str, np.ndarray] = {}
        for item_id in episode_ids:
            if item_id in ag_predictions.index.get_level_values(0):
                results[item_id] = ag_predictions.loc[item_id]["mean"].values
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, model_path: str, config=None) -> "TiDEForecaster":
        """Load a saved TiDE model.

        Overrides base class to deserialize config as TiDEConfig
        (not ModelConfig), preserving TiDE-specific fields like
        encoder_hidden_dim, scaling, covariate_cols, etc.
        """
        if config is None:
            config_path = os.path.join(model_path, "config.json")
            if os.path.exists(config_path):
                with open(config_path) as f:
                    config_dict = json.load(f)
                if "training_backend" in config_dict:
                    config_dict["training_backend"] = TrainingBackend(
                        config_dict["training_backend"]
                    )
                config = TiDEConfig(**config_dict)
            else:
                raise ValueError(f"No config found at {config_path}")
        return super().load(model_path, config=config)

    def _save_checkpoint(self, output_dir: str) -> None:
        """Save predictor path reference.

        AutoGluon auto-saves the full predictor during fit(). This method
        writes a small JSON reference file so _load_checkpoint can locate
        the predictor directory later.
        """
        if self.predictor is not None:
            ref_path = os.path.join(output_dir, "tide_predictor.json")
            os.makedirs(output_dir, exist_ok=True)
            with open(ref_path, "w") as f:
                json.dump({"predictor_path": str(self.predictor.path)}, f, indent=2)
            self.logger.info("Predictor reference saved to %s", ref_path)

    def _load_checkpoint(self, model_dir: str) -> None:
        """Load AutoGluon predictor from directory.

        Checks for a tide_predictor.json reference file first (written
        by _save_checkpoint). Falls back to loading model_dir directly as
        an AutoGluon predictor path.
        """
        from autogluon.timeseries import TimeSeriesPredictor

        ref_path = os.path.join(model_dir, "tide_predictor.json")
        if os.path.exists(ref_path):
            with open(ref_path) as f:
                predictor_path = json.load(f)["predictor_path"]
            pkl_file = os.path.join(predictor_path, "predictor.pkl")
            if not os.path.exists(pkl_file):
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
