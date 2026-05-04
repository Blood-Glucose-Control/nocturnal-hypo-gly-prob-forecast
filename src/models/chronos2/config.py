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
        fine_tune_batch_size: Per-device training batch size passed to the
            HuggingFace Trainer via AutoGluon. Defaults to AutoGluon's
            default of 32 when None. Reducing this helps with GPU OOM;
            increasing it may improve gradient stability.
        batch_size: Inference batch size for AutoGluon predict(). Defaults
            to AutoGluon's default of 256 when None. AutoGluon will
            automatically halve this on OOM during prediction.
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
    model_path: Optional[str] = "autogluon/chronos-2"
    forecast_length: int = 96  # 8 hours at 5-min intervals
    training_backend: TrainingBackend = TrainingBackend.CUSTOM

    # Chronos-2 / AutoGluon specific training
    fine_tune_steps: int = 15000
    fine_tune_lr: float = 1e-5
    # None = use AutoGluon defaults (fine_tune_batch_size=32, batch_size=256)
    fine_tune_batch_size: Optional[int] = None
    batch_size: Optional[int] = None  # type: ignore[assignment]
    time_limit: Optional[int] = None
    # How often (in gradient steps) the HuggingFace Trainer logs loss/lr.
    # None = use HF Trainer default (500). Set explicitly to override.
    fine_tune_logging_steps: Optional[int] = None

    # Gap handling (used in _prepare_training_data)
    imputation_threshold_mins: int = 45
    min_segment_length: Optional[int] = None

    # Covariates — column names to include as past-only context features.
    # These columns appear in training data and inference context windows but
    # are NOT provided for the forecast horizon (avoiding data leakage from
    # post-midnight reactive events). Defaults to ["iob"].
    covariate_cols: List[str] = field(default_factory=lambda: ["iob"])
    target_col: str = "bg_mM"
    # Joint co-target mode: when joint_target_cols has >1 entry, each column becomes
    # a separate item in the AutoGluon panel (long-format stacking). The model
    # trains jointly on all targets via shared weights. At inference, only
    # target_col (primary target) predictions are returned.
    # Empty list = single-target mode (backward compatible, uses covariates).
    # covariate_cols are ignored in multi-target mode.
    joint_target_cols: List[str] = field(default_factory=list)
    patient_col: str = "p_num"
    time_col: str = "datetime"

    # Data grid
    interval_mins: int = 5  # CGM sampling interval (5 min for all datasets)

    # AutoGluon training settings
    eval_metric: str = "WQL"
    enable_ensemble: bool = False
    # min_past=1 is AutoGluon's default. It controls the minimum number of past
    # steps required when AutoGluon creates sliding windows from each segment.
    # With gap-handled segments (each >= context_length + forecast_length rows),
    # most windows naturally get full context regardless of this setting.
    min_past: int = 1

    # Periodic checkpointing during fine-tuning.
    # When set, the HuggingFace Trainer saves a checkpoint every N steps via
    # fine_tune_trainer_kwargs. After training, _train_model materialises each
    # checkpoint-N/ dir into a standalone eval-ready snapshot alongside the
    # main output_dir so it can be passed directly to --checkpoint.
    checkpoint_save_steps: Optional[int] = None

    @property
    def is_multitarget(self) -> bool:
        """True when multiple target columns are configured (joint forecasting)."""
        return len(self.joint_target_cols) > 1

    def __post_init__(self):
        if self.min_segment_length is None:
            self.min_segment_length = self.context_length + self.forecast_length

        # Validate multi-target config
        if self.is_multitarget and self.target_col not in self.joint_target_cols:
            raise ValueError(
                f"target_col '{self.target_col}' must be in joint_target_cols "
                f"{self.joint_target_cols} (it is the primary prediction target)"
            )
        if len(self.joint_target_cols) == 1:
            raise ValueError(
                f"joint_target_cols has 1 entry {self.joint_target_cols}; use target_col "
                f"for single-target mode, or add more columns for joint mode"
            )

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
                # Disable cross_learning so each time series is predicted
                # independently.  Our episodes are unrelated patient-nights;
                # joint prediction across items is wrong.
                "cross_learning": False,
            }
        }
        if self.fine_tune_batch_size is not None:
            hp["Chronos2"]["fine_tune_batch_size"] = self.fine_tune_batch_size
        if self.batch_size is not None:
            hp["Chronos2"]["batch_size"] = self.batch_size
        if self.min_past != 1:
            hp["Chronos2"]["min_past"] = self.min_past
        trainer_kwargs: dict = {}
        if self.fine_tune_logging_steps is not None:
            trainer_kwargs.update(
                {
                    # Log loss/lr every N steps so training progress is visible
                    # in the log without waiting for a full checkpoint interval.
                    "logging_strategy": "steps",
                    "logging_steps": self.fine_tune_logging_steps,
                }
            )
        if self.checkpoint_save_steps is not None:
            trainer_kwargs.update(
                {
                    "save_strategy": "steps",
                    "save_steps": self.checkpoint_save_steps,
                    # Don't save optimizer/scheduler state — we only need weights.
                    "save_only_model": True,
                    # Override Chronos2 pipeline's hardcoded save_total_limit=1
                    # so intermediate checkpoints are not deleted.
                    "save_total_limit": None,
                }
            )
        if trainer_kwargs:
            hp["Chronos2"]["fine_tune_trainer_kwargs"] = trainer_kwargs
        return hp
