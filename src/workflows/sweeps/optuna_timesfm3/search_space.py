# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)

"""Search space for the TimesFM 3.0 (LoRA fine-tune) x Optuna sweep.

Tuned for the **annual competition (IOBP2 extension) holdout** reality:

- Each trial fully LoRA-fine-tunes TimesFM 3.0 (expensive: minutes on GPU),
  so the space favours a small number of high-signal knobs.
- TimesFM 3.0's grad-enabled ``decode`` accepts covariates spanning the context
  **and** the forecast horizon, so selected insulin channels are routed as
  KNOWN-FUTURE features (the competition permits future insulin). This differs
  from the committed ``src/models/timesfm3`` adapter, which is past-only.
- Only CGM + insulin are available at holdout forecast time (no carbs,
  wearables, or meal labels), so the known covariate candidates are
  **insulin-only** (raw insulin + IOB/insulin-activity kinetics) plus optional
  deterministic hour features.
- ``eval_metric`` is fixed to ``mard_avg`` (the optimized objective); it is not
  a searched knob — the objective computes it directly from the temporal
  validation windows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import optuna

from ..optuna_chronos2.features import (
    INSULIN_OBSERVED_COL,
    KNOWN_COVARIATE_CANDIDATES,
    MEAL_EVENT_COL,
    PAST_COVARIATE_CANDIDATES,
    PAST_COVARIATE_MASKS,
    SEX_KNOWN_COL,
)

# Insulin-derived known-future covariate candidates only (carb-derived features
# are unavailable in the holdout and are excluded). Preserves
# PAST_COVARIATE_CANDIDATES order:
#   total_insulin, basal_insulin, bolus_insulin, iob_exp,
#   insulin_activity_exp, iob_hov, insulin_activity_hov,
#   iob_wil, insulin_activity_wil
INSULIN_KNOWN_CANDIDATES: tuple[str, ...] = tuple(
    c
    for c in PAST_COVARIATE_CANDIDATES
    if PAST_COVARIATE_MASKS.get(c) == INSULIN_OBSERVED_COL
)


@dataclass
class TimesFM3SearchPoint:
    """A fully-sampled TimesFM 3.0 fine-tune trial configuration."""

    hyperparameters: Dict[str, Any]  # LoRA + optimizer fine-tune hyperparameters
    context_length: int
    forecast_length: int
    interval_mins: int
    eval_metric: str
    known_cols: List[str]


@dataclass
class TimesFM3SearchSpaceBounds:
    """Tunable bounds for the TimesFM 3.0 search space (overridable from YAML)."""

    forecast_length: int = 24
    interval_mins: int = 5
    # Context window. 288 steps = 24h is the CONTEST MAXIMUM; top trials all pin
    # here, so we keep a narrow band near the cap (224..288) rather than a hard fix.
    context_min: int = 224
    context_max: int = 288
    context_step: int = 32
    # LoRA adapter capacity.
    lora_r_choices: List[int] = field(default_factory=lambda: [4, 8, 16, 32, 64])
    # v3: added 512 — top v2 trials pinned lora_alpha at the 256 ceiling.
    lora_alpha_choices: List[int] = field(
        default_factory=lambda: [8, 16, 32, 64, 128, 256, 512]
    )
    lora_dropout_min: float = 0.0
    # v3: raised 0.3 -> 0.5 — top v2 trials pinned dropout at the 0.3 ceiling.
    lora_dropout_max: float = 0.5
    # LoRA target modules: "all-linear" (every nn.Linear), "attention" (q/k/v/o
    # projections), or "attn-qv" (query+value projections only).
    lora_target_modules_choices: List[str] = field(
        default_factory=lambda: ["all-linear", "attention", "attn-qv"]
    )
    # Advanced LoRA variants.
    use_dora_choices: List[bool] = field(default_factory=lambda: [False, True])
    # v3: rsLoRA pruned (never selected by any top v2 trial); kept as a 1-choice
    # knob so seed configs and the objective's ``use_rslora`` read still resolve.
    use_rslora_choices: List[bool] = field(default_factory=lambda: [False])
    # Optimizer.
    lr_min: float = 1e-5
    # v3: raised 1e-3 -> 3e-3 — top v2 trials pinned lr near the 1e-3 ceiling.
    lr_max: float = 3e-3
    weight_decay_min: float = 1e-8
    weight_decay_max: float = 1e-2
    # LR schedule. v3: pruned to cosine-only (every top v2 trial used cosine);
    # kept as a 1-choice knob for seed/objective compatibility.
    lr_schedule_choices: List[str] = field(default_factory=lambda: ["cosine"])
    warmup_ratio_min: float = 0.0
    warmup_ratio_max: float = 0.15
    # Fine-tune schedule. Val best is typically ~epoch 1 (immediate overfit), so
    # allow short runs with a little upside; early stopping + pruning cap it.
    max_epochs_choices: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 6])
    # v3: added 8 — top v2 trials pinned batch_size at the 16 floor.
    batch_size_choices: List[int] = field(default_factory=lambda: [8, 16, 32, 64])
    # Inference: z-normalize the context before forecasting.
    use_znorm_choices: List[bool] = field(default_factory=lambda: [False, True])
    # Fixed objective metric (computed directly; never sampled).
    eval_metric: str = "mard_avg"
    # Known-future insulin channel candidates (per-channel on/off).
    known_candidates: List[str] = field(
        default_factory=lambda: list(INSULIN_KNOWN_CANDIDATES)
    )
    insulin_as_known: bool = True
    # Deterministic hour-of-day known features (hour_sin/hour_cos, on/off
    # together). v5 drops them (``use_known_time`` pinned False across v3/v4
    # top trials); default True preserves v3/v4 behaviour.
    include_known_time: bool = True
    # Meal-announcement flag as an optional known-future covariate (v3). Carries
    # signal orthogonal to insulin/bolus (meal-bolus rises vs correction-bolus
    # falls); trainable on IOBP2-train, applicable to the holdout.
    include_meal: bool = True
    meal_known_col: str = MEAL_EVENT_COL
    # Known-future channels added on EVERY trial (not toggled). v5 promotes the
    # channels that every top-v4/v3 trial selected (basal/bolus/iob_wil/meal) to
    # always-on so the sampler spends its budget on genuinely uncertain knobs.
    # Default empty preserves v3/v4 behaviour (every channel toggled).
    always_known_cols: List[str] = field(default_factory=list)
    # Per-patient sex as an always-on known-future covariate (the Chronos-2 A/B
    # showed a large holdout gain from sex-as-known). Default off for back-compat.
    include_sex: bool = False
    sex_known_col: str = SEX_KNOWN_COL
    # Remaining demographics (age/height/weight/BMI/age-of-diagnosis/Hispanic)
    # added together via a single ``use_demographics`` bundle toggle. Default
    # empty keeps them out (v3/v4). v5 sets DEMOGRAPHIC_KNOWN_COLS.
    demographic_bundle: List[str] = field(default_factory=list)


def suggest_timesfm3_point(
    trial: optuna.Trial, bounds: TimesFM3SearchSpaceBounds
) -> TimesFM3SearchPoint:
    """Sample one TimesFM 3.0 fine-tune configuration + known-future covariate set.

    All selected insulin channels (and optionally hour_sin/hour_cos) are routed
    as KNOWN-FUTURE covariates spanning context + horizon; there is no past-only
    covariate path in this sweep.
    """
    context_length = trial.suggest_int(
        "context_length",
        bounds.context_min,
        bounds.context_max,
        step=bounds.context_step,
    )
    lora_r = trial.suggest_categorical("lora_r", bounds.lora_r_choices)
    lora_alpha = trial.suggest_categorical("lora_alpha", bounds.lora_alpha_choices)
    lora_dropout = trial.suggest_float(
        "lora_dropout", bounds.lora_dropout_min, bounds.lora_dropout_max
    )
    lora_target_modules = trial.suggest_categorical(
        "lora_target_modules", bounds.lora_target_modules_choices
    )
    use_dora = trial.suggest_categorical("use_dora", bounds.use_dora_choices)
    use_rslora = trial.suggest_categorical("use_rslora", bounds.use_rslora_choices)
    use_znorm = trial.suggest_categorical("use_znorm", bounds.use_znorm_choices)
    lr = trial.suggest_float("lr", bounds.lr_min, bounds.lr_max, log=True)
    weight_decay = trial.suggest_float(
        "weight_decay", bounds.weight_decay_min, bounds.weight_decay_max, log=True
    )
    lr_schedule = trial.suggest_categorical("lr_schedule", bounds.lr_schedule_choices)
    warmup_ratio = (
        trial.suggest_float(
            "warmup_ratio", bounds.warmup_ratio_min, bounds.warmup_ratio_max
        )
        if lr_schedule != "constant"
        else 0.0
    )
    max_epochs = trial.suggest_categorical("max_epochs", bounds.max_epochs_choices)
    batch_size = trial.suggest_categorical("batch_size", bounds.batch_size_choices)

    # Known-future covariate selection: optional deterministic hour features
    # (on/off together), per-channel insulin toggles, and an optional meal flag.
    use_known_time = bounds.include_known_time and trial.suggest_categorical(
        "use_known_time", [False, True]
    )
    known_cols = list(KNOWN_COVARIATE_CANDIDATES) if use_known_time else []
    if bounds.insulin_as_known:
        insulin_known = [
            col
            for col in bounds.known_candidates
            if trial.suggest_categorical(f"use_known__{col}", [False, True])
        ]
        known_cols = list(dict.fromkeys([*known_cols, *insulin_known]))
    if bounds.include_meal and trial.suggest_categorical(
        f"use_known__{bounds.meal_known_col}", [False, True]
    ):
        known_cols = list(dict.fromkeys([*known_cols, bounds.meal_known_col]))

    # Always-on known-future channels (v5): added on every trial, not toggled.
    if bounds.always_known_cols:
        known_cols = list(dict.fromkeys([*known_cols, *bounds.always_known_cols]))
    # Per-patient demographics as known-future covariates (v5): sex always-on,
    # the remaining demographics added together via a single bundle toggle.
    if bounds.include_sex:
        known_cols = list(dict.fromkeys([*known_cols, bounds.sex_known_col]))
    if bounds.demographic_bundle and trial.suggest_categorical(
        "use_demographics", [False, True]
    ):
        known_cols = list(dict.fromkeys([*known_cols, *bounds.demographic_bundle]))

    hyperparameters: Dict[str, Any] = {
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "lora_target_modules": lora_target_modules,
        "use_dora": use_dora,
        "use_rslora": use_rslora,
        "use_znorm": use_znorm,
        "lr": lr,
        "weight_decay": weight_decay,
        "lr_schedule": lr_schedule,
        "warmup_ratio": warmup_ratio,
        "max_epochs": max_epochs,
        "batch_size": batch_size,
    }

    return TimesFM3SearchPoint(
        hyperparameters=hyperparameters,
        context_length=context_length,
        forecast_length=bounds.forecast_length,
        interval_mins=bounds.interval_mins,
        eval_metric=bounds.eval_metric,
        known_cols=known_cols,
    )
