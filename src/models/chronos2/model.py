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

import numpy as np
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
    - Midnight-anchored nocturnal evaluation is a separate concern handled
      by the standalone evaluate_with_covariates() utility in chronos2/utils.py,
      which takes (predictor, test_data, known_covariates, episodes) directly.
      It is not wired into this class because known_covariates and episode
      metadata are external pipeline concerns, not model-internal state.
    """

    config_class = Chronos2Config

    def __init__(
        self,
        config: Chronos2Config,
        lora_config=None,
        distributed_config=None,
    ):
        # AutoGluon predictor — set before super().__init__() which calls
        # _initialize_model() (our no-op)
        self.predictor = None
        # Chronos2Pipeline for zero-shot inference (lazily initialized)
        self._zs_pipeline = None
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

    @property
    def supports_zero_shot(self) -> bool:
        return True

    @property
    def supports_probabilistic_forecast(self) -> bool:
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
        predictor_kwargs = dict(
            prediction_length=config.forecast_length,
            # "target" is the column name after format_segments_for_autogluon
            # renames config.target_col (e.g. "bg_mM") -> "target"
            target="target",
            # known_covariates_names intentionally NOT set — covariates (IOB,
            # COB) are included as past-only context columns. Setting them as
            # "known" would require providing future values at inference time,
            # which constitutes data leakage (post-midnight IOB/COB are
            # reactive to future BG and unknowable at the prediction origin).
            freq=f"{config.interval_mins}min",
            eval_metric=config.eval_metric,
            path=output_dir,
        )
        if config.quantile_levels is not None:
            predictor_kwargs["quantile_levels"] = config.quantile_levels
        predictor = TimeSeriesPredictor(**predictor_kwargs)

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
    # Inference helpers
    # ------------------------------------------------------------------

    def _ensure_zs_pipeline(self):
        """Lazily initialise the zero-shot Chronos2Pipeline."""
        if self._zs_pipeline is None:
            import torch
            from chronos import Chronos2Pipeline

            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._zs_pipeline = Chronos2Pipeline.from_pretrained(
                self.config.model_path,
                device_map=device,
                torch_dtype=torch.float32,
            )

    def _prepare_zero_shot_context(self, data: pd.DataFrame):
        """Validate target column and return a Chronos-2 context tensor.

        Returns:
            torch.Tensor of shape (1, 1, context_length).
        """
        import torch

        config = self.config
        if config.target_col not in data.columns:
            raise ValueError(
                f"Expected target column '{config.target_col}' not found in input "
                f"DataFrame. Available columns: {list(data.columns)}"
            )
        bg_values = data[config.target_col].values.astype(np.float32)
        bg_values = bg_values[-config.context_length :]
        return torch.tensor(bg_values).reshape(1, 1, -1)

    def _prepare_autogluon_data(self, data: pd.DataFrame):
        """Format a raw inference DataFrame into a TimeSeriesDataFrame.

        Handles episode_id assignment, timestamp conversion, target column
        renaming, and covariate column selection — the shared boilerplate
        required before calling ``self.predictor.predict()``.

        Returns:
            TimeSeriesDataFrame ready for AutoGluon prediction.
        """
        from autogluon.timeseries import TimeSeriesDataFrame

        config = self.config
        context = data.copy()

        if "episode_id" not in context.columns:
            context["episode_id"] = "ep_0"

        context["item_id"] = context["episode_id"]
        if config.time_col in context.columns:
            context["timestamp"] = pd.to_datetime(context[config.time_col])
        else:
            context["timestamp"] = context.index
        context = context.rename(columns={config.target_col: "target"})

        ag_cols = ["item_id", "timestamp", "target"] + config.covariate_cols
        ag_cols = [c for c in ag_cols if c in context.columns]
        context = context[ag_cols].set_index(["item_id", "timestamp"])
        return TimeSeriesDataFrame(context)

    @staticmethod
    def _episode_ids_from(data: pd.DataFrame) -> np.ndarray:
        """Return the array of episode IDs present in *data*."""
        if "episode_id" in data.columns:
            return data["episode_id"].unique()
        return np.array(["ep_0"])

    def _zero_shot_forecast(self, data: pd.DataFrame, quantile_levels=None):
        """Run zero-shot inference via Chronos2Pipeline.

        Returns:
            Tuple of (quantiles_np, mean_np) where:
              - quantiles_np: shape (n_quantile_levels, forecast_length)
              - mean_np: shape (forecast_length,)
        """
        self._ensure_zs_pipeline()
        context = self._prepare_zero_shot_context(data)
        kwargs = dict(prediction_length=self.config.forecast_length)
        if quantile_levels is not None:
            kwargs["quantile_levels"] = quantile_levels
        quantiles, mean = self._zs_pipeline.predict_quantiles(context, **kwargs)
        return (
            quantiles[0].detach().cpu().numpy(),
            mean[0].squeeze().detach().cpu().numpy(),
        )

    def _autogluon_extract(self, data: pd.DataFrame, columns: list) -> np.ndarray:
        """Run fine-tuned AutoGluon inference and extract specified columns.

        Args:
            data: Input DataFrame (same format as _predict).
            columns: Column names to extract from AutoGluon predictions,
                e.g. ["mean"] for point forecast or ["0.1", "0.2", ...] for
                quantiles. Multiple columns are stacked along axis 0.

        Returns:
            np.ndarray. For a single column: shape (n_episodes * forecast_length,).
            For multiple columns: shape (len(columns), n_episodes * forecast_length).
        """
        ts_data = self._prepare_autogluon_data(data)
        ag_predictions = self.predictor.predict(ts_data)
        multi = len(columns) > 1

        result_arrays = []
        for episode_id in self._episode_ids_from(data):
            if episode_id not in ag_predictions.index.get_level_values(0):
                continue
            ep_preds = ag_predictions.loc[episode_id]
            if multi:
                result_arrays.append(np.stack([ep_preds[c].values for c in columns]))
            else:
                result_arrays.append(ep_preds[columns[0]].values)

        if not result_arrays:
            return np.empty((len(columns), 0)) if multi else np.array([])
        return np.concatenate(result_arrays, axis=-1)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _predict(
        self, data: pd.DataFrame, quantile_levels=None, **kwargs
    ) -> np.ndarray:
        """Generate forecasts using Chronos-2.

        Two inference paths:
        - Zero-shot (self.predictor is None): Uses Chronos2Pipeline directly.
        - Fine-tuned (self.predictor exists): Uses AutoGluon predictor.

        Args:
            data: Single-episode DataFrame with target_col (bg_mM).
                For multi-episode panels, use predict_batch() instead.
            quantile_levels: When None, returns point forecast as shape
                (forecast_length,). When set, returns quantile forecasts as
                shape (len(quantile_levels), forecast_length).

        Returns:
            np.ndarray — point forecast or quantile forecasts depending on
            quantile_levels parameter.
        """
        if quantile_levels is not None:
            return self._predict_quantiles_impl(data, quantile_levels)

        if self.predictor is None:
            _, mean = self._zero_shot_forecast(data)
            return mean
        return self._autogluon_extract(data, columns=["mean"])

    def _predict_quantiles_impl(
        self, data: pd.DataFrame, quantile_levels: list
    ) -> np.ndarray:
        """Internal quantile forecast logic shared by _predict and _predict_batch."""
        if self.predictor is None:
            quantiles, _ = self._zero_shot_forecast(data, quantile_levels)
            return quantiles

        # Validate requested levels against training-time registration.
        available = set(
            round(q, 8)
            for q in (self.config.quantile_levels or self.DEFAULT_QUANTILE_LEVELS)
        )
        missing = [q for q in quantile_levels if round(q, 8) not in available]
        if missing:
            raise ValueError(
                f"Quantile levels {missing} were not registered at training time. "
                f"Available: {sorted(available)}. Re-train with these levels set in "
                f"config.quantile_levels, or request a subset of the available levels."
            )
        return self._autogluon_extract(data, columns=[str(q) for q in quantile_levels])

    def _predict_batch(
        self,
        data: pd.DataFrame,
        episode_col: str,
        quantile_levels=None,
    ) -> Dict[str, np.ndarray]:
        """Native batch prediction for multiple episodes.

        - Fine-tuned path: packs all episodes into one TimeSeriesDataFrame
          and calls self.predictor.predict() once.
        - Zero-shot path: batches via Chronos2Pipeline.predict_quantiles()
          with an (N, 1, L) tensor for N series in one forward pass.

        Args:
            data: Panel DataFrame containing episode_col with episode IDs,
                target_col (bg_mM), and optional covariate columns.
            episode_col: Column name identifying episodes.
            quantile_levels: When None, extract "mean" column (point forecasts).
                When set, extract quantile columns. For fine-tuned models,
                levels must be a subset of those registered at training time.

        Returns:
            Dict mapping episode ID (as str) to numpy forecast array.
            Point forecasts: shape (forecast_length,) per episode.
            Quantile forecasts: shape (len(quantile_levels), forecast_length).
        """
        import torch

        config = self.config
        episode_ids = data[episode_col].astype(str).unique().tolist()
        if not episode_ids:
            return {}

        if self.is_fitted:
            # Fine-tuned path: single AutoGluon predict call with all episodes
            if self.predictor is None:
                raise ValueError(
                    "Model is marked as fitted but predictor is None. "
                    "The checkpoint may not have loaded correctly."
                )

            if quantile_levels is not None:
                # Validate requested levels against training-time registration.
                available = set(
                    round(q, 8)
                    for q in (
                        self.config.quantile_levels or self.DEFAULT_QUANTILE_LEVELS
                    )
                )
                missing = [q for q in quantile_levels if round(q, 8) not in available]
                if missing:
                    raise ValueError(
                        f"Quantile levels {missing} were not registered at training time. "
                        f"Available: {sorted(available)}. Re-train with these levels set in "
                        f"config.quantile_levels, or request a subset of the available levels."
                    )

            from autogluon.timeseries import TimeSeriesDataFrame

            context = data.copy()
            context["item_id"] = context[episode_col].astype(str)
            if config.time_col in context.columns:
                context["timestamp"] = pd.to_datetime(context[config.time_col])
            else:
                context["timestamp"] = context.index
            context = context.rename(columns={config.target_col: "target"})

            ag_cols = ["item_id", "timestamp", "target"] + config.covariate_cols
            ag_cols = [c for c in ag_cols if c in context.columns]
            context = context[ag_cols].set_index(["item_id", "timestamp"])
            ts_data = TimeSeriesDataFrame(context)

            ag_predictions = self.predictor.predict(ts_data)

            # Choose which columns to extract
            if quantile_levels is not None:
                columns = [str(q) for q in quantile_levels]
            else:
                columns = ["mean"]
            multi = len(columns) > 1

            results: Dict[str, np.ndarray] = {}
            for item_id in episode_ids:
                if item_id in ag_predictions.index.get_level_values(0):
                    ep_preds = ag_predictions.loc[item_id]
                    if multi:
                        results[item_id] = np.stack(
                            [ep_preds[c].values for c in columns]
                        )
                    else:
                        results[item_id] = ep_preds[columns[0]].values
            return results

        # Zero-shot path: batch via Chronos2Pipeline
        self._ensure_zs_pipeline()

        if config.target_col not in data.columns:
            raise ValueError(
                f"Target column '{config.target_col}' not found in data. "
                f"Available columns: {list(data.columns)}"
            )

        # Build (N, 1, L) tensor — one series per episode
        grouped = data.groupby(data[episode_col].astype(str))
        series_list = []
        for ep_id in episode_ids:
            ep_data = grouped.get_group(ep_id)
            bg = ep_data[config.target_col].values.astype(np.float32)
            bg = bg[-config.context_length :]
            series_list.append(torch.tensor(bg))

        # Pad to same length for stacking
        max_len = max(len(s) for s in series_list)
        padded = torch.stack(
            [
                torch.nn.functional.pad(s, (max_len - len(s), 0), value=float("nan"))
                for s in series_list
            ]
        )  # (N, L)
        context_tensor = padded.unsqueeze(1)  # (N, 1, L)

        zs_kwargs = dict(prediction_length=config.forecast_length)
        if quantile_levels is not None:
            zs_kwargs["quantile_levels"] = quantile_levels

        quantiles, mean = self._zs_pipeline.predict_quantiles(
            context_tensor, **zs_kwargs
        )

        results = {}
        for i, ep_id in enumerate(episode_ids):
            if quantile_levels is not None:
                results[ep_id] = quantiles[i].detach().cpu().numpy()
            else:
                results[ep_id] = mean[i].squeeze().detach().cpu().numpy()
        return results

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
