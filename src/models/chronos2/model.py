# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)
# For commercial licensing, contact: cjrisi/christopher AT uwaterloo/gluroo DOT ca/com

"""
Chronos-2 forecaster using AutoGluon's TimeSeriesPredictor backend.

Unlike TTM (HuggingFace Trainer + PyTorch DataLoaders), Chronos-2 delegates
model loading, LoRA, sliding windows, and training to AutoGluon internally.
The primary model state is self.predictor (TimeSeriesPredictor), not
self.model (torch.nn.Module) which stays None.

Two separate pipelines exist in this class:

  TRAINING:  flat_df → patient_dict → gap-handled segments → TimeSeriesDataFrame
             → AutoGluon.fit() with sliding windows over full segments

  INFERENCE: flat_df → patient_dict → midnight-anchored episodes → AutoGluon.predict()
             Each episode is one clinical question: "Given 42h of context ending
             at midnight + known future covariates, forecast BG for the next 6h."

Extracted from validated experiment script (1.890 RMSE, -26% vs zero-shot)
and notebook 4.17-ss-chronos2-pipeline-validation.ipynb.
"""

import json
import logging
import os
from typing import Any, Dict, Tuple

import pandas as pd

from src.data.preprocessing.gap_handling import segment_all_patients
from src.models.base import BaseTimeSeriesFoundationModel, TrainingBackend
from src.utils.logging_helper import info_print

from .config import Chronos2Config
from .utils import (
    convert_to_patient_dict,
    format_segments_for_autogluon,
)

logger = logging.getLogger(__name__)


class Chronos2Forecaster(BaseTimeSeriesFoundationModel):
    """Chronos-2 time series forecaster using AutoGluon backend.

    Implements the BaseTimeSeriesFoundationModel interface for Chronos-2,
    wrapping AutoGluon's TimeSeriesPredictor for training and inference.

    Key differences from TTM/TimesFM:
    - training_backend = CUSTOM (AutoGluon manages training internally)
    - self.model stays None; self.predictor holds the AutoGluon predictor
    - _prepare_training_data returns TimeSeriesDataFrame, not DataLoaders
    - evaluate() is overridden for midnight-anchored nocturnal evaluation
    """

    def __init__(
        self,
        config: Chronos2Config,
        lora_config=None,
        distributed_config=None,
    ):
        # AutoGluon predictor — set before super().__init__() which calls
        # _initialize_model() (our no-op)
        self.predictor = None
        # lora_config and distributed_config are accepted for base class
        # compatibility but unused — AutoGluon handles LoRA internally
        super().__init__(config, lora_config, distributed_config)

    @property
    def training_backend(self) -> TrainingBackend:
        return TrainingBackend.CUSTOM

    @property
    def supports_lora(self) -> bool:
        # AutoGluon handles LoRA internally; base class LoRA mechanism
        # is unused since self.model stays None
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
            but Chronos-2 does not split — AutoGluon handles validation
            internally via sliding windows.
        """
        config = self.config

        # flat df -> per-patient dict
        patient_dict = convert_to_patient_dict(
            train_data, config.patient_col, config.time_col
        )
        info_print(f"Converted to {len(patient_dict)} patient dicts")

        # gap handling: interpolate small gaps, segment at large gaps
        segments = segment_all_patients(
            patient_dict,
            imputation_threshold_mins=config.imputation_threshold_mins,
            min_segment_length=config.min_segment_length,
            bg_col=config.target_col,
        )
        info_print(f"Gap handling: {len(segments)} segments")

        # format for AutoGluon with covariates
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
        """Fine-tune Chronos-2 via AutoGluon's TimeSeriesPredictor.

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

        info_print(f"Creating TimeSeriesPredictor at {output_dir}")
        predictor = TimeSeriesPredictor(
            prediction_length=config.forecast_length,
            # "target" is the column name after format_segments_for_autogluon
            # renames config.target_col (e.g. "bg_mM") -> "target"
            target="target",
            known_covariates_names=config.covariate_cols,
            eval_metric=config.eval_metric,
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
            f"Starting Chronos-2 fine-tuning: "
            f"{config.fine_tune_steps} steps, lr={config.fine_tune_lr}"
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

    def predict(
        self,
        data: pd.DataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        """Make predictions on panel DataFrame with episode_id.

        Accepts a panel DataFrame (multiple episodes stacked, identified by
        episode_id column) and returns predictions in the same format.
        Converts episode_id -> item_id for AutoGluon internally.

        Args:
            data: Panel DataFrame with episode_id column and DatetimeIndex.
                Each episode has context_length rows of BG + covariates.
            **kwargs: Optional known_covariates (panel DataFrame with
                episode_id and covariate columns for the forecast horizon).

        Returns:
            DataFrame with episode_id and target_col columns containing
            the predicted BG values for each episode's forecast horizon.
        """
        from autogluon.timeseries import TimeSeriesDataFrame

        if self.predictor is None:
            raise ValueError("Model must be fitted or loaded before prediction")

        config = self.config

        if "episode_id" not in data.columns:
            raise ValueError(
                "predict() expects a panel DataFrame with 'episode_id' column. "
                "Use evaluate_nocturnal_forecasting() to build episodes."
            )

        # Convert episode_id -> item_id for AutoGluon
        context = data.copy()
        context["item_id"] = context["episode_id"]
        context["timestamp"] = context.index
        context = context.rename(columns={config.target_col: "target"})

        # Select columns AutoGluon expects
        ag_cols = ["item_id", "timestamp", "target"] + config.covariate_cols
        ag_cols = [c for c in ag_cols if c in context.columns]
        context = context[ag_cols].set_index(["item_id", "timestamp"])
        ts_data = TimeSeriesDataFrame(context)

        # Build known covariates for AutoGluon if provided
        known_cov = None
        if "known_covariates" in kwargs and kwargs["known_covariates"] is not None:
            kcov = kwargs["known_covariates"].copy()
            kcov["item_id"] = kcov["episode_id"]
            kcov["timestamp"] = kcov.index
            cov_cols = ["item_id", "timestamp"] + [
                c for c in config.covariate_cols if c in kcov.columns
            ]
            kcov = kcov[cov_cols].set_index(["item_id", "timestamp"])
            known_cov = TimeSeriesDataFrame(kcov)

        # Call AutoGluon predictor
        ag_predictions = self.predictor.predict(ts_data, known_covariates=known_cov)

        # Convert AutoGluon output (MultiIndex item_id/timestamp, "mean" column)
        # back to panel DataFrame with episode_id and target_col
        result_rows = []
        for episode_id in data["episode_id"].unique():
            item_id = episode_id  # same mapping
            if item_id in ag_predictions.index.get_level_values(0):
                pred_series = ag_predictions.loc[item_id]["mean"]
                ep_df = pd.DataFrame(
                    {
                        config.target_col: pred_series.values,
                        "episode_id": episode_id,
                    },
                    index=pred_series.index,
                )
                result_rows.append(ep_df)

        if not result_rows:
            return pd.DataFrame(columns=[config.target_col, "episode_id"])

        return pd.concat(result_rows)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    # Why override load(): The base class load() at base_model.py:553
    # hardcodes `ModelConfig.from_dict(config_dict)`, which crashes when
    # the saved config.json contains Chronos-2-specific fields (covariate_cols,
    # fine_tune_steps, etc.) that ModelConfig doesn't accept. We override to
    # deserialize as Chronos2Config instead, then delegate to super().load()
    # with the pre-built config so it skips the ModelConfig.from_dict() path.
    @classmethod
    def load(cls, model_path: str, config=None) -> "Chronos2Forecaster":
        """Load a saved Chronos-2 model.

        Overrides base class to deserialize config as Chronos2Config
        (not ModelConfig), preserving Chronos-2-specific fields like
        covariate_cols, fine_tune_steps, etc.
        """
        if config is None:
            config_path = os.path.join(model_path, "config.json")
            if os.path.exists(config_path):
                with open(config_path) as f:
                    config_dict = json.load(f)
                # Convert serialized enum strings back to enum values.
                # ModelConfig.to_dict() saves enums as strings (e.g.
                # TrainingBackend.CUSTOM -> "custom"), but dataclass
                # constructors don't auto-coerce strings back to enums.
                if "training_backend" in config_dict:
                    config_dict["training_backend"] = TrainingBackend(
                        config_dict["training_backend"]
                    )
                config = Chronos2Config(**config_dict)
            else:
                raise ValueError(f"No config found at {config_path}")
        # Pass pre-deserialized config to parent — skips ModelConfig.from_dict()
        return super().load(model_path, config=config)

    def _save_checkpoint(self, output_dir: str) -> None:
        """Save predictor path reference.

        AutoGluon auto-saves the full predictor during fit(). This method
        writes a small JSON reference file so _load_checkpoint can locate
        the predictor directory later.
        """
        if self.predictor is not None:
            ref_path = os.path.join(output_dir, "chronos2_predictor.json")
            os.makedirs(output_dir, exist_ok=True)
            with open(ref_path, "w") as f:
                json.dump({"predictor_path": str(self.predictor.path)}, f, indent=2)
            self.logger.info("Predictor reference saved to %s", ref_path)

    def _load_checkpoint(self, model_dir: str) -> None:
        """Load AutoGluon predictor from directory.

        Checks for a chronos2_predictor.json reference file first (written
        by _save_checkpoint). Falls back to loading model_dir directly as
        an AutoGluon predictor path.
        """
        from autogluon.timeseries import TimeSeriesPredictor

        ref_path = os.path.join(model_dir, "chronos2_predictor.json")
        if os.path.exists(ref_path):
            with open(ref_path) as f:
                predictor_path = json.load(f)["predictor_path"]
            # Fall back to model_dir if the referenced path no longer exists
            # (e.g. the model directory was relocated after training)
            if not os.path.exists(predictor_path):
                self.logger.warning(
                    "Predictor path %s not found, falling back to %s",
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
