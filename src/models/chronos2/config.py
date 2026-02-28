# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)
# For commercial licensing, contact: cjrisi/christopher AT uwaterloo/gluroo DOT ca/com

"""
Chronos-2 configuration classes.

Extends the base ModelConfig with Chronos-2 / AutoGluon-specific parameters.
Chronos-2 uses AutoGluon's TimeSeriesPredictor backend (not HuggingFace Trainer),
so training parameters map to AutoGluon's API rather than transformers.Trainer.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.models.base import ModelConfig, TrainingBackend


@dataclass
class Chronos2Config(ModelConfig):
    """Configuration for Chronos-2 model using AutoGluon's TimeSeriesPredictor.

    Inherits from ModelConfig and adds Chronos-2-specific attributes for
    AutoGluon training, gap handling, and covariate configuration.

    Attributes:
        fine_tune_steps: Number of fine-tuning gradient steps.
        fine_tune_lr: Learning rate for fine-tuning.
        time_limit: AutoGluon time limit in seconds (None = unlimited).
        imputation_threshold_mins: Gaps up to this duration are interpolated.
        min_segment_length: Minimum rows for a gap-handled segment to be kept.
            Auto-computed as context_length + forecast_length if None.
        covariate_cols: Column names for past-only context covariates (e.g.,
            ["iob"] or ["iob", "cob"]). Included in training data and context
            windows but NOT provided for the forecast horizon.
        target_col: Source column name for the target variable.
        patient_col: Column name for patient identifiers in flat DataFrames.
        time_col: Column name for timestamps in flat DataFrames.
        interval_mins: Data sampling interval in minutes.
        eval_metric: AutoGluon evaluation metric.
        enable_ensemble: Whether to enable AutoGluon ensembling.
        min_past: Minimum past context for AutoGluon sliding windows.
    """

    # Override parent defaults
    model_type: str = "chronos2"
    model_path: str = "autogluon/chronos-2"
    forecast_length: int = 72  # 6 hours at 5-min intervals
    training_backend: TrainingBackend = TrainingBackend.CUSTOM

    # Chronos-2 / AutoGluon specific training
    fine_tune_steps: int = 15000
    fine_tune_lr: float = 1e-5
    time_limit: Optional[int] = None

    # Gap handling (used in _prepare_training_data)
    imputation_threshold_mins: int = 45
    min_segment_length: Optional[int] = None

    # Covariates — column names to include as past-only context features.
    # These columns appear in training data and inference context windows but
    # are NOT provided for the forecast horizon (avoiding data leakage from
    # post-midnight reactive events). Defaults to ["iob"].
    covariate_cols: List[str] = field(default_factory=lambda: ["iob"])
    target_col: str = "bg_mM"
    patient_col: str = "p_num"
    time_col: str = "datetime"

    # Data grid
    interval_mins: int = 5  # CGM sampling interval (5 min for all datasets)

    # AutoGluon training settings
    eval_metric: str = "RMSE"
    enable_ensemble: bool = False
    # min_past=1 is AutoGluon's default. It controls the minimum number of past
    # steps required when AutoGluon creates sliding windows from each segment.
    # With gap-handled segments (each >= context_length + forecast_length rows),
    # most windows naturally get full context regardless of this setting.
    min_past: int = 1

    def __post_init__(self):
        if self.min_segment_length is None:
            self.min_segment_length = self.context_length + self.forecast_length

    def get_autogluon_hyperparameters(self) -> Dict:
        """Build hyperparameters dict for TimeSeriesPredictor.fit().

        Returns:
            Dict with "Chronos2" key mapping to AutoGluon hyperparameters.
        """
        hp = {
            "Chronos2": {
                "model_path": self.model_path,
                "fine_tune": self.training_mode == "fine_tune",
                "fine_tune_steps": self.fine_tune_steps,
                "fine_tune_lr": self.fine_tune_lr,
                "context_length": self.context_length,
            }
        }
        if self.min_past != 1:
            hp["Chronos2"]["min_past"] = self.min_past
        return hp
