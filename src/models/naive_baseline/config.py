# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)

"""
Naive baseline forecasting model configuration.

Wraps AutoGluon's NaiveModel (last-value carry-forward) and AverageModel
(historical mean) as simple performance floors for comparing against TSFMs.

Both models are purely statistical — no learning occurs, training is
instantaneous, and no GPU is required. The predictor still needs to be "fit"
on training data so AutoGluon can compute residuals for quantile synthesis.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.models.base import ModelConfig, TrainingBackend

_VALID_MODEL_NAMES = frozenset({"Naive", "Average"})


@dataclass
class NaiveBaselineConfig(ModelConfig):
    """Configuration for NaiveModel / AverageModel via AutoGluon.

    Attributes:
        model_name: "Naive" (last-value carry-forward) or "Average"
            (historical mean). Controls the AutoGluon hyperparameters key.
        covariate_cols: Kept for pipeline compatibility; neither Naive nor
            Average uses covariates (AutoGluon ignores them for these models).
    """

    model_type: str = "naive_baseline"
    training_mode: str = "from_scratch"
    training_backend: TrainingBackend = TrainingBackend.CUSTOM

    # Which AutoGluon naive model to use
    model_name: str = "Naive"  # or "Average"

    # Minimal segment length to keep after gap handling
    min_segment_length: Optional[int] = None

    # Gap handling
    imputation_threshold_mins: int = 45

    # Covariate config (ignored by Naive/Average but needed for pipeline compat)
    covariate_cols: List[str] = field(default_factory=list)
    target_col: str = "bg_mM"
    patient_col: str = "p_num"
    time_col: str = "datetime"
    interval_mins: int = 5

    # AutoGluon settings
    eval_metric: str = "WQL"
    enable_ensemble: bool = False
    time_limit: Optional[int] = None

    # Quantile levels for probabilistic output
    quantile_levels: Optional[List[float]] = None

    def __post_init__(self):
        if self.model_name not in _VALID_MODEL_NAMES:
            raise ValueError(
                f"model_name must be one of {sorted(_VALID_MODEL_NAMES)}, "
                f"got '{self.model_name}'."
            )
        if self.min_segment_length is None:
            # Naive/Average need very little context; use forecast_length as floor
            self.min_segment_length = self.forecast_length

    def get_autogluon_hyperparameters(self) -> Dict:
        """Build hyperparameters dict for TimeSeriesPredictor.fit().

        AutoGluon selects the model via the top-level key name. Both Naive and
        Average accept an empty dict (no tunable hyperparameters).
        """
        return {self.model_name: {}}
