# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)

"""Search space for the DeepAR x Optuna sweep.

Tuned for the **annual competition (IOBP2 extension) holdout** reality:

- DeepAR supports known-future covariates and static features, but not past
  covariates, so selected insulin channels are routed as known-future features.
- Only CGM + insulin are available at holdout forecast time (no carbs, wearables,
  or meal labels), so the known covariate candidates are **insulin-only** (raw
  insulin + IOB/insulin-activity kinetics) plus optional deterministic hour
  features.
- Runs are expected to train on IOBP2 patients only (see ``data.source_filter``),
  for which device/modality/algorithm/source statics are constant, so the default
  static candidates drop those and keep only the IOBP2-varying demographics +
  insulin type.
- ``eval_metric`` is fixed to MAPE because it is low-impact for this externally
  scored objective and MAPE is equivalent to MARD.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import optuna

from ..optuna_chronos2.data import (
    PIECEWISE_STATIC_CANDIDATES,  # noqa: F401  (kept for reference/YAML overrides)
    STATIC_COVARIATE_CANDIDATES,  # noqa: F401
)
from ..optuna_chronos2.features import (
    INSULIN_OBSERVED_COL,
    KNOWN_COVARIATE_CANDIDATES,
    PAST_COVARIATE_CANDIDATES,
    PAST_COVARIATE_MASKS,
)

# Insulin-derived known-future covariates only (carb-derived features are
# unavailable in the holdout and are excluded). Preserves
# PAST_COVARIATE_CANDIDATES order.
INSULIN_KNOWN_CANDIDATES: tuple[str, ...] = tuple(
    c
    for c in PAST_COVARIATE_CANDIDATES
    if PAST_COVARIATE_MASKS.get(c) == INSULIN_OBSERVED_COL
)

# Static covariates that vary within IOBP2 AND are populated in the annual holdout.
# (ethnicity + age_of_diagnosis are 0% populated in the holdout; is_pregnant is
# all-False there — all three are excluded to keep train/serve parity.)
IOBP2_STATIC_CANDIDATES: tuple[str, ...] = (
    "age",
    "gender",
    "height",
    "weight",
    "insulin_type_basal",
    "insulin_type_bolus",
)

# distr_output choices -> constructed to gluonts Output instances in the objective.
DISTR_OUTPUT_CHOICES: tuple[str, ...] = (
    "studentt",
    "normal",
    "laplace",
)


@dataclass
class DeepARSearchPoint:
    """A fully-sampled DeepAR trial configuration."""

    hyperparameters: Dict[str, Any]  # DeepAR model hyperparameters
    context_length: int
    forecast_length: int
    interval_mins: int
    eval_metric: str
    target_scaler: Optional[str]
    distr_output: str
    known_cols: List[str]
    static_cols: List[str]


@dataclass
class DeepARSearchSpaceBounds:
    """Tunable bounds for the DeepAR search space (overridable from YAML/config)."""

    forecast_length: int = 24
    interval_mins: int = 5
    context_min: int = 216
    context_max: int = 288
    context_step: int = 12
    num_layers_choices: List[int] = field(default_factory=lambda: [1, 2, 3, 4])
    hidden_size_choices: List[int] = field(
        default_factory=lambda: [20, 40, 80, 160, 320]
    )
    dropout_rate_min: float = 0.0
    dropout_rate_max: float = 0.3
    lr_min: float = 1e-4
    lr_max: float = 1e-2
    weight_decay_min: float = 1e-8
    weight_decay_max: float = 1e-2
    max_epochs_choices: List[int] = field(default_factory=lambda: [50, 100])
    batch_size_choices: List[int] = field(default_factory=lambda: [64, 128, 256])
    num_batches_per_epoch_choices: List[int] = field(
        default_factory=lambda: [50, 100, 200]
    )
    num_parallel_samples_choices: List[int] = field(default_factory=lambda: [100, 200])
    target_scaler_choices: List[Optional[str]] = field(
        default_factory=lambda: [None, "standard", "robust", "mean_abs"]
    )
    distr_output_choices: List[str] = field(
        default_factory=lambda: list(DISTR_OUTPUT_CHOICES)
    )
    eval_metric: str = "MAPE"
    known_candidates: List[str] = field(
        default_factory=lambda: list(INSULIN_KNOWN_CANDIDATES)
    )
    static_candidates: List[str] = field(
        default_factory=lambda: list(IOBP2_STATIC_CANDIDATES)
    )
    insulin_as_known: bool = True


def suggest_deepar_point(
    trial: optuna.Trial, bounds: DeepARSearchSpaceBounds
) -> DeepARSearchPoint:
    """Sample one DeepAR configuration + covariate selection for a trial."""
    context_length = trial.suggest_int(
        "context_length",
        bounds.context_min,
        bounds.context_max,
        step=bounds.context_step,
    )
    num_layers = trial.suggest_categorical("num_layers", bounds.num_layers_choices)
    hidden_size = trial.suggest_categorical("hidden_size", bounds.hidden_size_choices)
    dropout_rate = trial.suggest_float(
        "dropout_rate", bounds.dropout_rate_min, bounds.dropout_rate_max
    )
    lr = trial.suggest_float("lr", bounds.lr_min, bounds.lr_max, log=True)
    weight_decay = trial.suggest_float(
        "weight_decay", bounds.weight_decay_min, bounds.weight_decay_max, log=True
    )
    max_epochs = trial.suggest_categorical("max_epochs", bounds.max_epochs_choices)
    batch_size = trial.suggest_categorical("batch_size", bounds.batch_size_choices)
    num_batches_per_epoch = trial.suggest_categorical(
        "num_batches_per_epoch", bounds.num_batches_per_epoch_choices
    )
    num_parallel_samples = trial.suggest_categorical(
        "num_parallel_samples", bounds.num_parallel_samples_choices
    )
    target_scaler = trial.suggest_categorical(
        "target_scaler", bounds.target_scaler_choices
    )
    distr_output = trial.suggest_categorical(
        "distr_output", bounds.distr_output_choices
    )

    use_known_time = trial.suggest_categorical("use_known_time", [False, True])
    known_cols = list(KNOWN_COVARIATE_CANDIDATES) if use_known_time else []
    if bounds.insulin_as_known:
        insulin_known = [
            col
            for col in bounds.known_candidates
            if trial.suggest_categorical(f"use_known__{col}", [False, True])
        ]
        known_cols = list(dict.fromkeys([*known_cols, *insulin_known]))

    static_cols = [
        col
        for col in bounds.static_candidates
        if trial.suggest_categorical(f"use_static__{col}", [False, True])
    ]

    hyperparameters: Dict[str, Any] = {
        "context_length": context_length,
        "num_layers": num_layers,
        "hidden_size": hidden_size,
        "dropout_rate": dropout_rate,
        "lr": lr,
        "weight_decay": weight_decay,
        "max_epochs": max_epochs,
        "batch_size": batch_size,
        "num_batches_per_epoch": num_batches_per_epoch,
        "num_parallel_samples": num_parallel_samples,
    }
    if target_scaler is not None:
        hyperparameters["target_scaler"] = target_scaler

    return DeepARSearchPoint(
        hyperparameters=hyperparameters,
        context_length=context_length,
        forecast_length=bounds.forecast_length,
        interval_mins=bounds.interval_mins,
        eval_metric=bounds.eval_metric,
        target_scaler=target_scaler,
        distr_output=distr_output,
        known_cols=known_cols,
        static_cols=static_cols,
    )
