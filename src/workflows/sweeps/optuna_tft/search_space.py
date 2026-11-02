# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)

"""Search space for the Temporal Fusion Transformer (TFT) x Optuna sweep.

Tuned for the **annual competition (IOBP2 extension) holdout** reality:

- Only CGM + insulin are available at holdout forecast time (no carbs, wearables,
  or meal labels), so the default past-covariate candidates are **insulin-only**
  (raw insulin + IOB/insulin-activity kinetics) and carb-derived features are
  excluded to avoid train/serve skew.
- Runs are expected to train on IOBP2 patients only (see ``data.source_filter``),
  for which device/modality/algorithm/source statics are constant, so the default
  static candidates drop those and keep only the IOBP2-varying demographics +
  insulin type.
- Architecture/training bounds are narrowed to the regions the first ~290-trial
  ``so1`` study favoured, and the saturated ranges (batch size, batches/epoch,
  hidden/variable dim, context) are pushed higher.
- ``distr_output`` is now searchable (glucose is positive / right-skewed, so
  Student-t / Laplace / Gamma heads may beat the default quantile head).

``insulin_as_known`` routes the selected insulin covariates to *known* future
covariates instead of *past* ones (the parallel ``so3`` study), reflecting the
competition's allowance of future insulin as a known input.
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
    PAST_COVARIATE_CANDIDATES,
    PAST_COVARIATE_MASKS,
)

# Insulin-derived past covariates only (carb-derived features are unavailable in
# the holdout and are excluded). Preserves PAST_COVARIATE_CANDIDATES order.
INSULIN_PAST_CANDIDATES: tuple[str, ...] = tuple(
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
# (Gamma is excluded — fails to fit under mean_abs + imputed context. Normal is
# excluded — worst performer in so2 and the only head that produced NaN-loss fit
# failures. Laplace ~ L1 loss suits MARD and was the so2 best.)
DISTR_OUTPUT_CHOICES: tuple[str, ...] = (
    "quantile",  # default QuantileOutput
    "studentt",
    "laplace",
)


@dataclass
class TFTSearchPoint:
    """A fully-sampled TFT trial configuration."""

    hyperparameters: Dict[str, Any]  # TFT model hyperparameters
    context_length: int
    forecast_length: int
    interval_mins: int
    eval_metric: str
    target_scaler: Optional[str]
    distr_output: str
    past_cols: List[str]
    known_cols: List[str]
    static_cols: List[str]


@dataclass
class TFTSearchSpaceBounds:
    """Tunable bounds for the TFT search space (overridable from YAML/config)."""

    forecast_length: int = 24
    interval_mins: int = 5
    # Longer context helped in so1 (top-decile median 276); keep it high.
    context_min: int = 216
    context_max: int = 288
    context_step: int = 12
    # hidden_dim must be divisible by num_heads; every choice below divides evenly
    # by every num_heads option, so no invalid combination is sampled. Pushed
    # higher than so1 (which topped out at 128 / 64).
    hidden_dim_choices: List[int] = field(default_factory=lambda: [64, 128, 256])
    # Low variable_dim acts as an information bottleneck on continuous-feature
    # embeddings, which curbs VSN softmax collapse (important for so3's known
    # covariates); low values are included by default.
    variable_dim_choices: List[int] = field(default_factory=lambda: [4, 8, 16, 32, 64])
    num_heads_choices: List[int] = field(default_factory=lambda: [2, 4, 8])
    dropout_max: float = 0.5
    lr_min: float = 3e-4
    lr_max: float = 3e-3
    # L2 regularization on the optimizer. Strong weight decay (~1e-2) is a key
    # lever against TFT Variable-Selection-Network softmax collapse when a feature
    # is routed to the decoder as a known covariate (see so3). Searched log-uniform.
    weight_decay_min: float = 1e-8
    weight_decay_max: float = 1e-2
    max_epochs_choices: List[int] = field(default_factory=lambda: [50, 100])
    batch_size_choices: List[int] = field(default_factory=lambda: [256, 384])
    num_batches_per_epoch_choices: List[int] = field(
        default_factory=lambda: [200, 300, 400]
    )
    early_stopping_patience: int = 10
    # so1 favoured mean_abs under the default quantile head; with distributional
    # heads (Normal/StudentT/Laplace) a centered scaler (standard/robust) may pair
    # better, so the scaler is searched jointly with distr_output.
    target_scaler_choices: List[Optional[str]] = field(
        default_factory=lambda: ["mean_abs", "standard", "robust"]
    )
    # RMSE (best in so1) + MAPE (~ MARD, the objective). WQL/MASE dropped.
    eval_metric_choices: List[str] = field(default_factory=lambda: ["RMSE", "MAPE"])
    distr_output_choices: List[str] = field(
        default_factory=lambda: list(DISTR_OUTPUT_CHOICES)
    )
    # Covariate candidate pools (holdout-parity defaults).
    past_candidates: List[str] = field(
        default_factory=lambda: list(INSULIN_PAST_CANDIDATES)
    )
    static_candidates: List[str] = field(
        default_factory=lambda: list(IOBP2_STATIC_CANDIDATES)
    )
    # so3: route selected insulin covariates to KNOWN future covariates instead
    # of past covariates (competition allows future insulin as a known input).
    insulin_as_known: bool = False


def suggest_tft_point(
    trial: optuna.Trial, bounds: TFTSearchSpaceBounds
) -> TFTSearchPoint:
    """Sample one TFT configuration + covariate selection for a trial."""
    context_length = trial.suggest_int(
        "context_length",
        bounds.context_min,
        bounds.context_max,
        step=bounds.context_step,
    )
    hidden_dim = trial.suggest_categorical("hidden_dim", bounds.hidden_dim_choices)
    variable_dim = trial.suggest_categorical(
        "variable_dim", bounds.variable_dim_choices
    )
    num_heads = trial.suggest_categorical("num_heads", bounds.num_heads_choices)
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, bounds.dropout_max)
    lr = trial.suggest_float("lr", bounds.lr_min, bounds.lr_max, log=True)
    weight_decay = trial.suggest_float(
        "weight_decay", bounds.weight_decay_min, bounds.weight_decay_max, log=True
    )
    max_epochs = trial.suggest_categorical("max_epochs", bounds.max_epochs_choices)
    batch_size = trial.suggest_categorical("batch_size", bounds.batch_size_choices)
    num_batches_per_epoch = trial.suggest_categorical(
        "num_batches_per_epoch", bounds.num_batches_per_epoch_choices
    )
    target_scaler = trial.suggest_categorical(
        "target_scaler", bounds.target_scaler_choices
    )
    eval_metric = trial.suggest_categorical("eval_metric", bounds.eval_metric_choices)
    distr_output = trial.suggest_categorical(
        "distr_output", bounds.distr_output_choices
    )

    # Covariate selection. Each candidate is an independent on/off switch; the
    # ``insulin_as_known`` flag decides whether selected insulin covariates enter
    # as known-future or past covariates.
    selected = [
        col
        for col in bounds.past_candidates
        if trial.suggest_categorical(f"use_cov__{col}", [False, True])
    ]
    if bounds.insulin_as_known:
        known_cols, past_cols = selected, []
    else:
        past_cols, known_cols = selected, []
    static_cols = [
        col
        for col in bounds.static_candidates
        if trial.suggest_categorical(f"use_static__{col}", [False, True])
    ]

    hyperparameters: Dict[str, Any] = {
        "context_length": context_length,
        "hidden_dim": hidden_dim,
        "variable_dim": variable_dim,
        "num_heads": num_heads,
        "dropout_rate": dropout_rate,
        "lr": lr,
        "weight_decay": weight_decay,
        "max_epochs": max_epochs,
        "batch_size": batch_size,
        "num_batches_per_epoch": num_batches_per_epoch,
        "early_stopping_patience": bounds.early_stopping_patience,
    }
    if target_scaler is not None:
        hyperparameters["target_scaler"] = target_scaler

    return TFTSearchPoint(
        hyperparameters=hyperparameters,
        context_length=context_length,
        forecast_length=bounds.forecast_length,
        interval_mins=bounds.interval_mins,
        eval_metric=eval_metric,
        target_scaler=target_scaler,
        distr_output=distr_output,
        past_cols=past_cols,
        known_cols=known_cols,
        static_cols=static_cols,
    )
