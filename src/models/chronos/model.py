# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)

"""
Chronos-2 Forecaster Model.

Ground truth: scripts/chronos2_experiment.py
Ported piece by piece from working script.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from src.models.base import BaseTimeSeriesFoundationModel, TrainingBackend
from src.models.chronos.config import Chronos2Config
from src.data.models import ColumnNames
from src.utils.logging_helper import info_print


class Chronos2Forecaster(BaseTimeSeriesFoundationModel):
    """Chronos-2 forecaster - minimal skeleton.

    Key difference from other models: uses self.predictor (AutoGluon), not self.model.
    """

    def __init__(
        self, config: Chronos2Config, lora_config=None, distributed_config=None
    ):
        # Chronos uses predictor, not model
        self.predictor = None
        super().__init__(config, lora_config, distributed_config)
        self.config: Chronos2Config = self.config

    # Required properties
    @property
    def training_backend(self) -> TrainingBackend:
        return TrainingBackend.CUSTOM

    @property
    def supports_lora(self) -> bool:
        return False  # AutoGluon handles LoRA internally

    # Required abstract methods - stubs for now
    def _initialize_model(self) -> None:
        info_print("Chronos2Forecaster initialized (predictor created during fit)")

    def _prepare_training_data(
        self, train_data: Any
    ) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
        # TODO: Chronos doesn't use DataLoaders - framework mismatch
        raise NotImplementedError("Use _train_model directly for Chronos")

    def _train_model(
        self, train_data: Any, output_dir: str, **kwargs
    ) -> Dict[str, Any]:
        """
        Train Chronos-2 model via AutoGluon TimeSeriesPredictor.

        Ported from scripts/chronos2_experiment.py:train_model()

        Args:
            train_data: Dict of patient_id -> DataFrame (from data loader)
            output_dir: Directory to save model
            **kwargs:
                val_data: Optional validation data dict

        Returns:
            Dict with training results including val_rmse
        """
        from autogluon.timeseries import TimeSeriesPredictor

        val_data = kwargs.get("val_data", None)
        covariate_names = self.config.known_covariates_names or []

        info_print(f"Training Chronos-2 (covariates={covariate_names})")
        info_print(
            f"Config: steps={self.config.fine_tune_steps}, lr={self.config.fine_tune_lr}"
        )

        # Build training episodes
        info_print("Building training episodes...")
        train_episodes = self._build_episodes(
            train_data, covariate_names=covariate_names
        )
        info_print(f"Total training episodes: {len(train_episodes)}")

        # Sample if needed
        if len(train_episodes) > self.config.max_train_episodes:
            info_print(f"Sampling {self.config.max_train_episodes} episodes...")
            np.random.seed(42)
            indices = np.random.choice(
                len(train_episodes), self.config.max_train_episodes, replace=False
            )
            train_episodes = [train_episodes[i] for i in sorted(indices)]

        # Build validation episodes if provided
        val_episodes = None
        if val_data is not None:
            info_print("Building validation episodes...")
            val_episodes = self._build_episodes(
                val_data, covariate_names=covariate_names
            )
            info_print(f"Total validation episodes: {len(val_episodes)}")

            if len(val_episodes) > self.config.max_val_episodes:
                info_print(f"Sampling {self.config.max_val_episodes} val episodes...")
                np.random.seed(42)
                indices = np.random.choice(
                    len(val_episodes), self.config.max_val_episodes, replace=False
                )
                val_episodes = [val_episodes[i] for i in sorted(indices)]

        # Format for AutoGluon
        info_print("Formatting data for AutoGluon...")
        ts_train, known_cov_train = self._format_for_autogluon(
            train_episodes, covariate_names=covariate_names
        )
        info_print(f"Training data shape: {ts_train.shape}")

        ts_val = None
        if val_episodes:
            ts_val, _ = self._format_for_autogluon(
                val_episodes, covariate_names=covariate_names
            )
            info_print(f"Validation data shape: {ts_val.shape}")

        # Create predictor
        predictor_kwargs = {
            "prediction_length": self.config.forecast_length,
            "target": "target",
            "eval_metric": "RMSE",
            "path": output_dir,
        }
        if covariate_names:
            predictor_kwargs["known_covariates_names"] = covariate_names

        self.predictor = TimeSeriesPredictor(**predictor_kwargs)

        # Determine if fine-tuning or zero-shot
        fine_tune = self.config.training_mode != "zero_shot"

        # Train
        info_print(
            f"Starting {'fine-tuning' if fine_tune else 'zero-shot'} training..."
        )

        fit_kwargs = {
            "train_data": ts_train,
            "hyperparameters": {
                "Chronos2": {
                    "model_path": self.config.model_path,
                    "fine_tune": fine_tune,
                    "fine_tune_steps": self.config.fine_tune_steps,
                    "fine_tune_lr": self.config.fine_tune_lr,
                    "context_length": self.config.context_length,
                }
            },
            "time_limit": self.config.time_limit,
            "enable_ensemble": self.config.enable_ensemble,
        }

        if ts_val is not None:
            fit_kwargs["tuning_data"] = ts_val

        self.predictor.fit(**fit_kwargs)

        # Get results
        results = {
            "train_episodes": len(train_episodes),
            "val_episodes": len(val_episodes) if val_episodes else 0,
            "output_dir": output_dir,
        }

        info_print(f"Model saved to: {output_dir}")

        # Store episodes for later use in prediction
        self._train_episodes = train_episodes
        self._val_episodes = val_episodes

        return results

    def predict(
        self,
        data: Any,
        batch_size: Optional[int] = None,
        known_covariates: Any = None,
    ) -> np.ndarray:
        """
        Generate predictions using the trained Chronos-2 model.

        Args:
            data: Either:
                - Dict of patient_id -> DataFrame (will build episodes)
                - TimeSeriesDataFrame (use directly)
            batch_size: Ignored (AutoGluon handles batching)
            known_covariates: Pre-built known covariates TimeSeriesDataFrame

        Returns:
            numpy array of predictions (mean values)
        """
        from autogluon.timeseries import TimeSeriesDataFrame

        if self.predictor is None:
            raise RuntimeError("Model not trained. Call fit() or load() first.")

        covariate_names = self.config.known_covariates_names or []

        # If data is already TimeSeriesDataFrame, use directly
        if isinstance(data, TimeSeriesDataFrame):
            ts_data = data
            known_cov = known_covariates
        else:
            # Build episodes and format
            episodes = self._build_episodes(data, covariate_names=covariate_names)
            ts_data, known_cov = self._format_for_autogluon(
                episodes, covariate_names=covariate_names
            )

        # Predict
        if known_cov is not None:
            predictions = self.predictor.predict(ts_data, known_covariates=known_cov)
        else:
            predictions = self.predictor.predict(ts_data)

        # Extract mean predictions as numpy array
        # AutoGluon returns DataFrame with columns like 'mean', '0.1', '0.5', '0.9'
        if "mean" in predictions.columns:
            return predictions["mean"].values
        else:
            # Fallback to first numeric column
            return predictions.iloc[:, 0].values

    def _save_checkpoint(self, output_dir: str) -> None:
        """Save config for later loading. AutoGluon handles model weights."""
        import json
        import os

        # Save config.json so load() can reconstruct the model
        config_path = os.path.join(output_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        info_print(f"Chronos-2 model saved to: {output_dir}")

    def _load_checkpoint(self, model_dir: str) -> None:
        """Load a trained Chronos-2 model from disk."""
        from autogluon.timeseries import TimeSeriesPredictor

        info_print(f"Loading Chronos-2 model from: {model_dir}")
        self.predictor = TimeSeriesPredictor.load(model_dir)

    # =========================================================================
    # EPISODE BUILDING - Ported from scripts/chronos2_experiment.py
    # =========================================================================

    def _build_episodes(
        self,
        patient_data: Dict[str, pd.DataFrame],
        covariate_names: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Build midnight-anchored episodes from patient data.

        Ported from scripts/chronos2_experiment.py:build_episodes()

        Args:
            patient_data: Dict of patient_id -> DataFrame with DatetimeIndex
            covariate_names: List of covariate column names to include (e.g., ["iob", "cob"])

        Returns:
            List of episode dicts with keys:
                - anchor: midnight timestamp
                - context_df: DataFrame with context window
                - target_bg: numpy array of forecast targets
                - future_covariates: dict of covariate_name -> numpy array (if covariates provided)
        """
        TARGET_COL = ColumnNames.BG.value
        covariate_names = covariate_names or []

        interval_mins = self.config.interval_mins
        context_len = self.config.context_length
        horizon = self.config.forecast_length

        episodes = []

        for pid, pdf in patient_data.items():
            df = pdf.sort_index()
            df = df[~df.index.duplicated(keep="last")]

            # Check which covariates are available in this patient's data
            available_covariates = [
                cov
                for cov in covariate_names
                if cov in df.columns and df[cov].notna().any()
            ]

            # Resample to regular grid
            freq = f"{interval_mins}min"
            grid = pd.date_range(
                df.index.min().floor(freq), df.index.max().floor(freq), freq=freq
            )
            df = df.reindex(grid)

            dt = pd.Timedelta(minutes=interval_mins)
            earliest = df.index.min() + context_len * dt
            latest = df.index.max() - (horizon - 1) * dt

            first_midnight = earliest.normalize()
            if first_midnight < earliest:
                first_midnight += pd.Timedelta(days=1)

            last_midnight = latest.normalize()
            if last_midnight < first_midnight:
                continue

            for anchor in pd.date_range(first_midnight, last_midnight, freq="D"):
                window_start = anchor - context_len * dt
                window_end = anchor + horizon * dt
                window_index = pd.date_range(
                    window_start, window_end, freq=freq, inclusive="left"
                )

                # Get columns: target + available covariates
                cols_to_get = [TARGET_COL] + available_covariates
                window_df = df.reindex(window_index)[cols_to_get]

                # Skip if any BG missing
                if window_df[TARGET_COL].isna().any():
                    continue

                context_df = window_df.iloc[:context_len].copy()
                forecast_df = window_df.iloc[context_len:].copy()

                target_bg = forecast_df[TARGET_COL].to_numpy()

                episode = {
                    "anchor": anchor,
                    "context_df": context_df,
                    "target_bg": target_bg,
                }

                # Add covariates if available
                if available_covariates:
                    # Check if any covariate is mostly missing
                    skip_episode = False
                    for cov in available_covariates:
                        if context_df[cov].isna().mean() > 0.5:
                            skip_episode = True
                            break
                        # Forward fill and fill remaining NaN with 0
                        context_df[cov] = context_df[cov].ffill().fillna(0)

                    if skip_episode:
                        continue

                    # Store future covariate values
                    episode["future_covariates"] = {
                        cov: forecast_df[cov].ffill().fillna(0).to_numpy()
                        for cov in available_covariates
                    }
                    episode["context_df"] = context_df

                episodes.append(episode)

        return episodes

    def _format_for_autogluon(
        self,
        episodes: List[Dict[str, Any]],
        covariate_names: Optional[List[str]] = None,
    ) -> Tuple[Any, Optional[Any]]:
        """
        Convert episodes to AutoGluon TimeSeriesDataFrame format.

        Ported from scripts/chronos2_experiment.py:format_for_autogluon()

        Args:
            episodes: List of episode dicts from _build_episodes()
            covariate_names: List of covariate column names to include

        Returns:
            Tuple of (train_data, known_covariates)
            - train_data: TimeSeriesDataFrame
            - known_covariates: TimeSeriesDataFrame or None
        """
        from autogluon.timeseries import TimeSeriesDataFrame

        TARGET_COL = ColumnNames.BG.value
        covariate_names = covariate_names or []

        train_data_list = []
        known_cov_list = []

        for i, ep in enumerate(episodes):
            item_id = f"ep_{i:04d}"

            df = ep["context_df"].copy()
            df["item_id"] = item_id
            df["timestamp"] = df.index
            df["target"] = df[TARGET_COL]

            # Check if this episode has covariates
            has_covariates = "future_covariates" in ep and ep["future_covariates"]

            if has_covariates:
                # Add covariate columns to training data
                cols = ["item_id", "timestamp", "target"]
                for cov in covariate_names:
                    if cov in df.columns:
                        cols.append(cov)
                train_data_list.append(df[cols])

                # Build known covariates for future
                future_timestamps = pd.date_range(
                    ep["anchor"],
                    periods=self.config.forecast_length,
                    freq=f"{self.config.interval_mins}min",
                )
                future_dict = {
                    "item_id": item_id,
                    "timestamp": future_timestamps,
                }
                for cov, values in ep["future_covariates"].items():
                    future_dict[cov] = values
                known_cov_list.append(pd.DataFrame(future_dict))
            else:
                train_data_list.append(df[["item_id", "timestamp", "target"]])

        train_combined = pd.concat(train_data_list, ignore_index=True)
        train_combined = train_combined.set_index(["item_id", "timestamp"])
        train_data = TimeSeriesDataFrame(train_combined)

        known_covariates = None
        if covariate_names and known_cov_list:
            known_combined = pd.concat(known_cov_list, ignore_index=True)
            known_combined = known_combined.set_index(["item_id", "timestamp"])
            known_covariates = TimeSeriesDataFrame(known_combined)

        return train_data, known_covariates
