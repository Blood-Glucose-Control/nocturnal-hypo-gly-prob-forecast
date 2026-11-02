# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)

"""Search space for the Moirai-1.x (native fine-tune) x Optuna sweep.

Tuned for the **annual competition (IOBP2 extension) holdout** reality:

- Each trial fully fine-tunes a Moirai-1.x checkpoint (cheap: seconds-minutes on
  GPU), so the space favours a small number of high-signal knobs.
- uni2ts ``MoiraiForecast`` accepts covariates spanning the context **and** the
  forecast horizon (``feat_dynamic_real``), so insulin is routed as KNOWN-FUTURE
  (the competition permits future insulin). ``total_insulin`` is always known;
  ``basal_insulin`` / ``bolus_insulin`` are optional known channels.
- Only CGM + insulin are available at holdout forecast time (no carbs,
  wearables, or meal labels), so covariate candidates are **insulin-only**.
- Optional past-only insulin-kinetics channels (IOB / insulin-activity) may be
  added on the context side.
- ``mard_avg`` is the optimized objective; the objective computes it directly
  from the temporal validation windows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import optuna

from ..optuna_chronos2.features import SEX_KNOWN_COL

# Moirai-1.x checkpoints that expose a uni2ts fine-tune module. Moirai-2.0 / MoE
# are inference-only in uni2ts 2.0.0 and are excluded from this fine-tune sweep.
MOIRAI_FINETUNE_MODELS: tuple[str, ...] = (
    "Salesforce/moirai-1.0-R-small",
    "Salesforce/moirai-1.0-R-base",
    "Salesforce/moirai-1.0-R-large",
    "Salesforce/moirai-1.1-R-small",
    "Salesforce/moirai-1.1-R-base",
    "Salesforce/moirai-1.1-R-large",
)

# v2 default model set: the v1 sweep's top configs all pinned moirai-1.1-R-base
# (the largest checkpoint searched) and never selected 1.0-* or 1.1-small, so v2
# restricts to the 1.1-R base/large contenders and adds the previously-missing
# -large checkpoint.
MOIRAI_V2_MODELS: tuple[str, ...] = (
    "Salesforce/moirai-1.1-R-base",
    "Salesforce/moirai-1.1-R-large",
)

# Known-future insulin channels fed on every trial (v1 top configs all used all
# three; the toggles are removed in v2). Their future values are competition-
# permitted and span the context + forecast horizon as feat_dynamic_real.
ALWAYS_KNOWN_INSULIN: tuple[str, ...] = (
    "total_insulin",
    "basal_insulin",
    "bolus_insulin",
)


@dataclass
class MoiraiSearchPoint:
    """A fully-sampled Moirai-1.x fine-tune trial configuration."""

    config_kwargs: Dict[str, Any]  # kwargs to build a MoiraiConfig
    context_length: int
    forecast_length: int
    interval_mins: int
    past_cols: List[str]  # past-only covariates (context side)
    known_cols: List[str]  # known-future covariates (context + horizon)


@dataclass
class MoiraiSearchSpaceBounds:
    """Tunable bounds for the Moirai fine-tune search space (YAML-overridable).

    Defaults are the **v2** space: expanded where v1 pinned a ceiling (add
    -large; epochs up to 16), restricted where v1 showed clear weakness (context
    fixed at the 288 contest max; no "auto" patch; no head_only; narrower lr; all
    three insulin channels always known-future), and augmented with four knobs v1
    left out (AdamW beta2, masked-reconstruction max ratio, warmup, grad-clip).
    """

    forecast_length: int = 24
    interval_mins: int = 5
    num_samples: int = 100  # inference Monte-Carlo samples (not searched)
    # v2: 1.1-R base/large only (see MOIRAI_V2_MODELS).
    model_path_choices: List[str] = field(
        default_factory=lambda: list(MOIRAI_V2_MODELS)
    )
    # Context window (steps). v2 fixes 288 = 24h (the CONTEST MAXIMUM); every v1
    # top trial pinned here.
    context_min: int = 288
    context_max: int = 288
    context_step: int = 96
    # Patch size. v2 drops "auto" (never selected by a top v1 trial).
    patch_size_choices: List[Any] = field(default_factory=lambda: [16, 32])
    # Optimizer. v2 narrows lr to the top-v1 cluster.
    lr_min: float = 2e-5
    lr_max: float = 2e-4
    weight_decay_min: float = 1e-8
    weight_decay_max: float = 1e-2
    # AdamW betas (v2 new knob: beta2 searched; beta1 fixed).
    beta1: float = 0.9
    beta2_choices: List[float] = field(default_factory=lambda: [0.95, 0.98, 0.999])
    # Masked-reconstruction augmentation (v2 new knob). min fixed; max searched.
    min_mask_ratio: float = 0.15
    max_mask_ratio_choices: List[float] = field(default_factory=lambda: [0.3, 0.5])
    # LR warmup + grad clipping (v2 new knobs; were fixed in v1).
    warmup_steps_choices: List[int] = field(default_factory=lambda: [0, 500, 1000])
    gradient_clip_val_choices: List[float] = field(
        default_factory=lambda: [0.5, 1.0, 2.0]
    )
    # Which parameters to fine-tune. v2 drops head_only (absent from v1 top).
    finetune_pattern_choices: List[str] = field(
        default_factory=lambda: ["full", "freeze_ffn"]
    )
    # Fine-tune schedule. v2 lifts the ceiling (v1 top all pinned 8 epochs).
    num_epochs_choices: List[int] = field(default_factory=lambda: [8, 12, 16])
    batch_size_choices: List[int] = field(default_factory=lambda: [32, 64])
    # Known-future insulin channels (all always-on in v2; see ALWAYS_KNOWN_INSULIN).
    always_known_candidates: List[str] = field(
        default_factory=lambda: list(ALWAYS_KNOWN_INSULIN)
    )
    # Optional past-only insulin-kinetics channels (context side; low importance,
    # kept as toggles for member diversity).
    past_candidates: List[str] = field(
        default_factory=lambda: ["iob_exp", "insulin_activity_exp"]
    )
    # v3: insulin-kinetics channels promoted to KNOWN-FUTURE (deterministic from
    # the known insulin schedule, so competition-legal ahead of time). Default
    # empty preserves v2 (kinetics stay past-only via ``past_candidates``); v3
    # sets always-on ``iob_exp`` + a toggled ``insulin_activity_exp``.
    always_known_kinetics: List[str] = field(default_factory=list)
    known_kinetics_candidates: List[str] = field(default_factory=list)
    # v3: per-patient demographics as known-future covariates. Sex is always-on
    # (the Chronos-2 sex A/B gave a large holdout gain); the remaining
    # demographics are added together via a single ``use_demographics`` bundle
    # toggle. Defaults keep them out (v2).
    include_sex_known: bool = False
    sex_known_col: str = SEX_KNOWN_COL
    demographic_bundle: List[str] = field(default_factory=list)
    # Fixed objective metric (computed directly; never sampled).
    eval_metric: str = "mard_avg"


def suggest_moirai_point(
    trial: optuna.Trial, bounds: MoiraiSearchSpaceBounds
) -> MoiraiSearchPoint:
    """Sample one Moirai-1.x fine-tune configuration + covariate selection (v2).

    All three insulin channels are always known-future; optional insulin-kinetics
    channels (past-only) are added via per-channel toggles. AdamW beta2, the
    masked-reconstruction max ratio, warmup, and grad-clip are searched.
    """
    model_path = trial.suggest_categorical("model_path", bounds.model_path_choices)
    context_length = trial.suggest_int(
        "context_length",
        bounds.context_min,
        bounds.context_max,
        step=bounds.context_step,
    )
    patch_size = trial.suggest_categorical("patch_size", bounds.patch_size_choices)
    lr = trial.suggest_float("lr", bounds.lr_min, bounds.lr_max, log=True)
    weight_decay = trial.suggest_float(
        "weight_decay", bounds.weight_decay_min, bounds.weight_decay_max, log=True
    )
    beta2 = trial.suggest_categorical("beta2", bounds.beta2_choices)
    max_mask_ratio = trial.suggest_categorical(
        "max_mask_ratio", bounds.max_mask_ratio_choices
    )
    warmup_steps = trial.suggest_categorical(
        "warmup_steps", bounds.warmup_steps_choices
    )
    gradient_clip_val = trial.suggest_categorical(
        "gradient_clip_val", bounds.gradient_clip_val_choices
    )
    finetune_pattern = trial.suggest_categorical(
        "finetune_pattern", bounds.finetune_pattern_choices
    )
    num_epochs = trial.suggest_categorical("num_epochs", bounds.num_epochs_choices)
    batch_size = trial.suggest_categorical("batch_size", bounds.batch_size_choices)

    # Known-future insulin: all three channels, always on.
    known_cols: List[str] = list(dict.fromkeys(bounds.always_known_candidates))

    # Past-only insulin-kinetics channels.
    past_cols: List[str] = [
        col
        for col in bounds.past_candidates
        if trial.suggest_categorical(f"use_past__{col}", [False, True])
    ]

    # v3: insulin-kinetics promoted to known-future (always-on + toggled).
    if bounds.always_known_kinetics:
        known_cols = list(dict.fromkeys([*known_cols, *bounds.always_known_kinetics]))
    for col in bounds.known_kinetics_candidates:
        if trial.suggest_categorical(f"use_known__{col}", [False, True]):
            known_cols = list(dict.fromkeys([*known_cols, col]))

    # v3: per-patient demographics as known-future covariates (sex always-on;
    # the rest via a single bundle toggle).
    if bounds.include_sex_known:
        known_cols = list(dict.fromkeys([*known_cols, bounds.sex_known_col]))
    if bounds.demographic_bundle and trial.suggest_categorical(
        "use_demographics", [False, True]
    ):
        known_cols = list(dict.fromkeys([*known_cols, *bounds.demographic_bundle]))

    config_kwargs: Dict[str, Any] = {
        "model_path": model_path,
        "context_length": context_length,
        "forecast_length": bounds.forecast_length,
        "training_mode": "fine_tune",
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "beta1": bounds.beta1,
        "beta2": beta2,
        "min_mask_ratio": bounds.min_mask_ratio,
        "max_mask_ratio": max_mask_ratio,
        "warmup_steps": warmup_steps,
        "gradient_clip_val": gradient_clip_val,
        "finetune_pattern": finetune_pattern,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "patch_size": patch_size,
        "num_samples": bounds.num_samples,
        "past_covariate_dim": len(past_cols),
        "covariate_cols": list(past_cols),
        "known_covariate_dim": len(known_cols),
        "known_covariate_cols": list(known_cols),
        "target_col": "bg_mM",
        "interval_mins": bounds.interval_mins,
        "fp16": True,
        "use_cpu": False,
    }

    return MoiraiSearchPoint(
        config_kwargs=config_kwargs,
        context_length=context_length,
        forecast_length=bounds.forecast_length,
        interval_mins=bounds.interval_mins,
        past_cols=list(past_cols),
        known_cols=list(known_cols),
    )
