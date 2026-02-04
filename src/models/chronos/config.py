# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)

"""
Chronos-2 configuration class.

Chronos-2 uses AutoGluon's TimeSeriesPredictor, which has different parameters
than HuggingFace-based models. Key difference: uses `fine_tune_steps` not epochs.

Reference:
    https://auto.gluon.ai/stable/_modules/autogluon/timeseries/models/chronos/chronos2.html
"""

import warnings
from typing import Any, Dict, List, Optional

from src.models.base import ModelConfig, TrainingBackend


class Chronos2Config(ModelConfig):
    """Configuration for Chronos-2 fine-tuning via AutoGluon.

    Important: Chronos-2 uses `fine_tune_steps`, not `num_epochs`.
    The base class `num_epochs` parameter is ignored.

    Attributes:
        # From base (used)
        model_path: HuggingFace Hub model ID (default: "autogluon/chronos-2").
        context_length: Input sequence length (default: 512).
        forecast_length: Prediction horizon (default: 72).
        training_mode: "zero_shot" or "fine_tune".

        # Chronos-2 specific
        fine_tune_steps: Number of gradient steps (default: 5000).
        fine_tune_lr: Learning rate (default: 1e-4).

        # Covariates
        known_covariates_names: Columns known at prediction time (e.g., ["iob", "cob"]).
            These must match columns in your data. Empty list = no covariates.

        # Episode building
        interval_mins: CGM sampling interval (default: 5).
        max_train_episodes: Max training episodes (default: 5000).
        max_val_episodes: Max validation episodes (default: 500).

        # AutoGluon
        time_limit: Training time limit in seconds (None = unlimited).
        enable_ensemble: Enable AutoGluon ensemble (default: False).
    """

    def __init__(self, **kwargs):
        # Warn if user passes num_epochs (common mistake)
        if "num_epochs" in kwargs:
            warnings.warn(
                "Chronos2Config: 'num_epochs' is ignored. "
                "Use 'fine_tune_steps' instead (Chronos-2 has no epochs concept).",
                UserWarning,
                stacklevel=2,
            )

        # Chronos-specific params to extract
        chronos_params = {
            "fine_tune_steps",
            "fine_tune_lr",
            "known_covariates_names",
            "interval_mins",
            "max_train_episodes",
            "max_val_episodes",
            "quantiles",
            "time_limit",
            "enable_ensemble",
        }

        # Filter for parent class
        base_kwargs = {k: v for k, v in kwargs.items() if k not in chronos_params}

        # Set defaults
        base_kwargs.setdefault("model_path", "autogluon/chronos-2")
        base_kwargs.setdefault("context_length", 512)
        base_kwargs.setdefault("forecast_length", 72)
        base_kwargs.setdefault("training_mode", "fine_tune")

        super().__init__(**base_kwargs)

        # Override identity
        self.model_type = "chronos2"
        self.training_backend = TrainingBackend.CUSTOM

        # Chronos-2 fine-tuning
        self.fine_tune_steps: int = kwargs.get("fine_tune_steps", 5000)
        self.fine_tune_lr: float = kwargs.get("fine_tune_lr", 1e-4)

        # Covariates
        self.known_covariates_names: List[str] = kwargs.get(
            "known_covariates_names", []
        )

        # Episode building
        self.interval_mins: int = kwargs.get("interval_mins", 5)
        self.max_train_episodes: int = kwargs.get("max_train_episodes", 5000)
        self.max_val_episodes: int = kwargs.get("max_val_episodes", 500)

        # Probabilistic
        self.quantiles: List[float] = kwargs.get("quantiles", [0.1, 0.5, 0.9])

        # AutoGluon
        self.time_limit: Optional[int] = kwargs.get("time_limit", None)
        self.enable_ensemble: bool = kwargs.get("enable_ensemble", False)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        d = super().to_dict()
        d.update(
            {
                "fine_tune_steps": self.fine_tune_steps,
                "fine_tune_lr": self.fine_tune_lr,
                "known_covariates_names": self.known_covariates_names,
                "interval_mins": self.interval_mins,
                "max_train_episodes": self.max_train_episodes,
                "max_val_episodes": self.max_val_episodes,
                "quantiles": self.quantiles,
                "time_limit": self.time_limit,
                "enable_ensemble": self.enable_ensemble,
            }
        )
        return d

    def get_autogluon_hyperparameters(self) -> Dict[str, Any]:
        """Get hyperparameters for AutoGluon's TimeSeriesPredictor.fit().

        Returns:
            Dict to pass as hyperparameters={"Chronos2": ...}
        """
        return {
            "model_path": self.model_path,
            "context_length": self.context_length,
            "fine_tune": self.training_mode == "fine_tune",
            "fine_tune_steps": self.fine_tune_steps,
            "fine_tune_lr": self.fine_tune_lr,
        }


# Factory functions


def create_chronos2_zero_shot_config(**kwargs) -> Chronos2Config:
    """Config for zero-shot inference (no fine-tuning)."""
    kwargs.setdefault("training_mode", "zero_shot")
    kwargs.setdefault("fine_tune_steps", 0)
    return Chronos2Config(**kwargs)


def create_chronos2_fine_tune_config(
    fine_tune_steps: int = 5000,
    fine_tune_lr: float = 1e-4,
    **kwargs,
) -> Chronos2Config:
    """Config for fine-tuning without covariates (target signal only)."""
    kwargs.update(
        {
            "training_mode": "fine_tune",
            "fine_tune_steps": fine_tune_steps,
            "fine_tune_lr": fine_tune_lr,
        }
    )
    return Chronos2Config(**kwargs)


def create_chronos2_covariate_config(
    covariates: List[str],
    fine_tune_steps: int = 5000,
    fine_tune_lr: float = 1e-4,
    **kwargs,
) -> Chronos2Config:
    """Config for fine-tuning with known covariates.

    Args:
        covariates: List of covariate column names that exist in your data.
            Examples: ["iob"] for Brown, ["cob"] for Aleppo, ["iob", "cob"] for both.
        fine_tune_steps: Number of gradient steps.
        fine_tune_lr: Learning rate.
        **kwargs: Additional config options.

    Returns:
        Configured Chronos2Config with covariates enabled.
    """
    if not covariates:
        raise ValueError("covariates must be a non-empty list of column names")
    kwargs.update(
        {
            "training_mode": "fine_tune",
            "known_covariates_names": covariates,
            "fine_tune_steps": fine_tune_steps,
            "fine_tune_lr": fine_tune_lr,
        }
    )
    return Chronos2Config(**kwargs)
