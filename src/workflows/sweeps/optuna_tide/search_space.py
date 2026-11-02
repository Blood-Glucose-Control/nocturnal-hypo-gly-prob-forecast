# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)

"""Search space for the TiDE x Optuna sweep.

Tuned for the **annual competition (IOBP2 extension) holdout** reality:

- TiDE supports known-future covariates and static features, but not past
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
    "quantile",  # default TiDE distribution head (left unset in objective)
    "studentt",
    "laplace",
)


@dataclass
class TiDESearchPoint:
    """A fully-sampled TiDE trial configuration."""

    hyperparameters: Dict[str, Any]  # TiDE model hyperparameters
    context_length: int
    forecast_length: int
    interval_mins: int
    eval_metric: str
    target_scaler: Optional[str]
    distr_output: str
    known_cols: List[str]
    static_cols: List[str]


@dataclass
class TiDESearchSpaceBounds:
    """Tunable bounds for the TiDE search space (overridable from YAML/config)."""

    forecast_length: int = 24
    interval_mins: int = 5
    context_min: int = 216
    context_max: int = 288
    context_step: int = 12
    encoder_hidden_dim_choices: List[int] = field(
        default_factory=lambda: [32, 64, 128, 256]
    )
    temporal_hidden_dim_choices: List[int] = field(
        default_factory=lambda: [32, 64, 128, 256]
    )
    distr_hidden_dim_choices: List[int] = field(
        default_factory=lambda: [32, 64, 128, 256]
    )
    feat_proj_hidden_dim_choices: List[int] = field(
        default_factory=lambda: [16, 32, 64]
    )
    num_layers_encoder_choices: List[int] = field(default_factory=lambda: [1, 2, 3])
    num_layers_decoder_choices: List[int] = field(default_factory=lambda: [1, 2, 3])
    decoder_output_dim_choices: List[int] = field(default_factory=lambda: [8, 16, 32])
    dropout_rate_min: float = 0.0
    dropout_rate_max: float = 0.5
    layer_norm_choices: List[bool] = field(default_factory=lambda: [True, False])
    lr_min: float = 1e-4
    lr_max: float = 1e-2
    weight_decay_min: float = 1e-8
    weight_decay_max: float = 1e-2
    max_epochs_choices: List[int] = field(default_factory=lambda: [50, 100])
    batch_size_choices: List[int] = field(default_factory=lambda: [128, 256, 512])
    num_batches_per_epoch_choices: List[int] = field(
        default_factory=lambda: [50, 100, 200]
    )
    target_scaler_choices: List[Optional[str]] = field(
        default_factory=lambda: [None, "standard", "robust", "mean_abs"]
    )
    distr_output_choices: List[str] = field(
        default_factory=lambda: list(DISTR_OUTPUT_CHOICES)
    )
    # Width of the FeatureProjection ResBlock that compresses the dynamic
    # features (calendar + known-future insulin) before the encoder. Default 2
    # is under-powered for insulin; scales the encoder input size.
    num_feat_dynamic_proj_choices: List[int] = field(
        default_factory=lambda: [2, 4, 8, 16]
    )
    # Window-level scaler applied inside the model (distinct from the
    # series-level target_scaler). None maps to gluonts NOPScaler.
    scaling_choices: List[Optional[str]] = field(
        default_factory=lambda: ["mean", "std", None]
    )
    eval_metric: str = "MAPE"
    known_candidates: List[str] = field(
        default_factory=lambda: list(INSULIN_KNOWN_CANDIDATES)
    )
    static_candidates: List[str] = field(
        default_factory=lambda: list(IOBP2_STATIC_CANDIDATES)
    )
    insulin_as_known: bool = True


def suggest_tide_point(
    trial: optuna.Trial, bounds: TiDESearchSpaceBounds
) -> TiDESearchPoint:
    """Sample one TiDE configuration + covariate selection for a trial."""
    context_length = trial.suggest_int(
        "context_length",
        bounds.context_min,
        bounds.context_max,
        step=bounds.context_step,
    )
    encoder_hidden_dim = trial.suggest_categorical(
        "encoder_hidden_dim", bounds.encoder_hidden_dim_choices
    )
    # TiDE feeds the dense-encoder output straight into the dense decoder, so the
    # encoder and decoder must share the same hidden dim (gluonts DenseEncoder ->
    # DenseDecoder). Tie the decoder dim to the encoder dim.
    decoder_hidden_dim = encoder_hidden_dim
    temporal_hidden_dim = trial.suggest_categorical(
        "temporal_hidden_dim", bounds.temporal_hidden_dim_choices
    )
    distr_hidden_dim = trial.suggest_categorical(
        "distr_hidden_dim", bounds.distr_hidden_dim_choices
    )
    feat_proj_hidden_dim = trial.suggest_categorical(
        "feat_proj_hidden_dim", bounds.feat_proj_hidden_dim_choices
    )
    num_layers_encoder = trial.suggest_categorical(
        "num_layers_encoder", bounds.num_layers_encoder_choices
    )
    num_layers_decoder = trial.suggest_categorical(
        "num_layers_decoder", bounds.num_layers_decoder_choices
    )
    decoder_output_dim = trial.suggest_categorical(
        "decoder_output_dim", bounds.decoder_output_dim_choices
    )
    dropout_rate = trial.suggest_float(
        "dropout_rate", bounds.dropout_rate_min, bounds.dropout_rate_max
    )
    layer_norm = trial.suggest_categorical("layer_norm", bounds.layer_norm_choices)
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
    distr_output = trial.suggest_categorical(
        "distr_output", bounds.distr_output_choices
    )
    num_feat_dynamic_proj = trial.suggest_categorical(
        "num_feat_dynamic_proj", bounds.num_feat_dynamic_proj_choices
    )
    scaling = trial.suggest_categorical("scaling", bounds.scaling_choices)

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
        "encoder_hidden_dim": encoder_hidden_dim,
        "decoder_hidden_dim": decoder_hidden_dim,
        "temporal_hidden_dim": temporal_hidden_dim,
        "distr_hidden_dim": distr_hidden_dim,
        "feat_proj_hidden_dim": feat_proj_hidden_dim,
        "num_layers_encoder": num_layers_encoder,
        "num_layers_decoder": num_layers_decoder,
        "decoder_output_dim": decoder_output_dim,
        "dropout_rate": dropout_rate,
        "layer_norm": layer_norm,
        "lr": lr,
        "weight_decay": weight_decay,
        "max_epochs": max_epochs,
        "batch_size": batch_size,
        "num_batches_per_epoch": num_batches_per_epoch,
        "num_feat_dynamic_proj": num_feat_dynamic_proj,
        "scaling": scaling,
    }
    if target_scaler is not None:
        hyperparameters["target_scaler"] = target_scaler

    return TiDESearchPoint(
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
