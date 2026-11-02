# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)

"""Optuna search space for Chronos-2 (LoRA) on the MetaboNet challenge shape.

``suggest_search_point`` samples a full configuration for one trial covering:

* Model context length (24-288 steps).
* LoRA adapter hyperparameters (rank, alpha, dropout).
* Optimizer / trainer hyperparameters (learning rate, steps, batch size, weight
  decay, warmup ratio).
* AutoGluon fine-tuning selection metric.
* Covariate selection: which past covariates, whether to use known-future
  time features, which static covariates to attach, and (when
  ``insulin_as_known``) which insulin channels to route as known-future
  covariates via dedicated ``use_known__*`` toggles.

The result is a :class:`SearchPoint` carrying the ``Chronos2Config`` keyword
arguments plus the resolved covariate column lists, so the objective can build
both the config and the data panels from one sampled point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import optuna

from .data import PIECEWISE_STATIC_CANDIDATES, STATIC_COVARIATE_CANDIDATES
from .features import DEMOGRAPHIC_KNOWN_COLS as _DEMOGRAPHIC_KNOWN_COLS
from .features import (
    INSULIN_OBSERVED_COL,
    KNOWN_COVARIATE_CANDIDATES,
    MEAL_EVENT_COL,
    MINS_TO_MEAL_COL,
    PAST_COVARIATE_CANDIDATES,
    PAST_COVARIATE_MASKS,
    SEX_KNOWN_COL,
)


@dataclass
class SearchPoint:
    """A fully-sampled trial configuration."""

    config_kwargs: Dict[str, Any]
    past_cols: List[str]
    known_cols: List[str]
    static_cols: List[str]
    eval_metric: str


# Named LoRA target-module presets for Chronos-2 (T5-based). "auto" lets PEFT
# infer target modules; the others use verified Chronos-2 module paths.
LORA_TARGET_MODULE_PRESETS: Dict[str, list[str] | None] = {
    "auto": None,
    "attn_qv": ["self_attention.q", "self_attention.v"],
    "attn_qkvo": [
        "self_attention.q",
        "self_attention.k",
        "self_attention.v",
        "self_attention.o",
    ],
    "full": [
        "self_attention.q",
        "self_attention.k",
        "self_attention.v",
        "self_attention.o",
        "output_patch_embedding.output_layer",
    ],
    # Broadest LoRA set: all attention projections + the feed-forward (MLP)
    # layers (layer.2.mlp.wi/wo) + the forecast head. Still LoRA (base frozen),
    # not full-weight fine-tuning.
    "full_mlp": [
        "self_attention.q",
        "self_attention.k",
        "self_attention.v",
        "self_attention.o",
        "mlp.wi",
        "mlp.wo",
        "output_patch_embedding.output_layer",
    ],
}


@dataclass
class SearchSpaceBounds:
    """Tunable bounds for the search space (overridable from YAML/config)."""

    context_min: int = 24
    context_max: int = 288
    context_step: int = 12
    lr_min: float = 1e-6
    lr_max: float = 1e-3
    steps_choices: List[int] = field(default_factory=lambda: [500, 1000, 2000, 4000])
    batch_size_choices: List[int] = field(default_factory=lambda: [64, 128, 256, 512])
    lora_r_choices: List[int] = field(default_factory=lambda: [4, 8, 16, 32, 64])
    lora_alpha_choices: List[int] = field(default_factory=lambda: [8, 16, 32, 64, 128])
    lora_dropout_max: float = 0.2
    lora_target_modules_choices: List[str] = field(
        default_factory=lambda: ["auto", "attn_qv", "attn_qkvo", "full"]
    )
    weight_decay_max: float = 0.1
    warmup_ratio_max: float = 0.2
    lr_scheduler_choices: List[str] = field(
        default_factory=lambda: ["constant_with_warmup", "linear", "cosine"]
    )
    grad_accum_choices: List[int] = field(default_factory=lambda: [1, 2, 4])
    cross_learning_choices: List[bool] = field(default_factory=lambda: [False, True])
    target_scaler_choices: List[Optional[str]] = field(
        default_factory=lambda: [None, "standard", "robust", "mean_abs"]
    )
    eval_metric_choices: List[str] = field(
        default_factory=lambda: ["RMSE", "WQL", "MASE", "MAPE"]
    )
    # When set, fixes the AutoGluon eval_metric instead of searching
    # eval_metric_choices. eval_metric only drives AutoGluon's predictor-level
    # validation scoring; fine-tune checkpointing uses eval_loss and trials are
    # scored on mard_avg by us, so it is effectively inert -- pin it to drop a
    # dead search dimension. None keeps the legacy searched behavior.
    eval_metric: Optional[str] = None
    # Fixed pass-throughs (not searched). None leaves AutoGluon defaults.
    fine_tune_eval_max_items: int | None = None
    # HF Trainer eval/checkpoint interval during fine-tuning. The chronos
    # pipeline hardcodes 100 but lets extra trainer kwargs override it; at large
    # step budgets ~1000-2000 keeps eval overhead low while giving the pruner
    # enough checkpoints. None keeps the pipeline default (100).
    eval_steps: int | None = 1000
    # Restrict selectable past-covariate pool. Defaults to all engineered
    # candidates but can be narrowed from YAML for cohort-specific runs.
    past_candidates: List[str] = field(
        default_factory=lambda: list(PAST_COVARIATE_CANDIDATES)
    )
    # Route selected insulin covariates as known-future covariates instead of
    # past covariates (so3-style setting).
    insulin_as_known: bool = False
    # Offer the known-future meal covariates (announcement flag + minutes-to-next
    # -meal) as optional known covariates; mins_since_meal is a past covariate.
    include_meal: bool = False
    meal_known_cols: List[str] = field(
        default_factory=lambda: [MEAL_EVENT_COL, MINS_TO_MEAL_COL]
    )
    # Route the broadcast per-patient sex column as a (constant) known covariate.
    # Chronos-2 ignores static_features, so this is how patient sex reaches it.
    include_sex_known: bool = False
    # Route the broadcast per-patient demographics (age, height, weight, BMI,
    # age-of-diagnosis, Hispanic/Latino) as (constant) known covariates, each via
    # its own use_known__* toggle so TPE finds the best subset.
    include_demographics_known: bool = False
    demographic_known_cols: List[str] = field(
        default_factory=lambda: list(_DEMOGRAPHIC_KNOWN_COLS)
    )
    # --- Optional knobs (searched only when set; None keeps library defaults) ---
    # Gradient clipping (HF TrainingArguments max_grad_norm), searched
    # log-uniform over [min, max] when both are provided. Stabilizes high-LR /
    # high-rank LoRA fine-tunes.
    max_grad_norm_min: float | None = None
    max_grad_norm_max: float | None = None
    # HF optimizer choices, e.g. ["adamw_torch_fused", "adafactor"]. Adafactor is
    # the T5-native optimizer and is memory-efficient for high-rank LoRA.
    optim_choices: List[str] | None = None
    # Minimum past-context steps AutoGluon requires when building training
    # windows; filters very short series. None keeps the pipeline default (1).
    min_past_choices: List[int] | None = None
    # AutoGluon covariate scaler ("local"/"global"/None) applied to covariate
    # columns before the model. Helps when insulin doses vary across patients.
    covariate_scaler_choices: List[Optional[str]] | None = None
    # Static covariate candidates to EXCLUDE from the search. Use to drop
    # covariates that are absent or category-mismatched in the deployment/holdout
    # set (e.g. for the IOBP2 annual holdout: source_file, treatment_group,
    # ethnicity, age_of_diagnosis) so trials cannot condition on a signal that
    # collapses to unknown/NaN at inference.
    static_exclude: List[str] = field(default_factory=list)


def suggest_search_point(
    trial: optuna.Trial, bounds: SearchSpaceBounds | None = None
) -> SearchPoint:
    """Sample one :class:`SearchPoint` from ``trial``."""
    bounds = bounds or SearchSpaceBounds()

    context_length = trial.suggest_int(
        "context_length",
        bounds.context_min,
        bounds.context_max,
        step=bounds.context_step,
    )
    fine_tune_lr = trial.suggest_float(
        "fine_tune_lr", bounds.lr_min, bounds.lr_max, log=True
    )
    fine_tune_steps = trial.suggest_categorical("fine_tune_steps", bounds.steps_choices)
    fine_tune_batch_size = trial.suggest_categorical(
        "fine_tune_batch_size", bounds.batch_size_choices
    )

    lora_r = trial.suggest_categorical("lora_r", bounds.lora_r_choices)
    lora_alpha = trial.suggest_categorical("lora_alpha", bounds.lora_alpha_choices)
    lora_dropout = trial.suggest_float("lora_dropout", 0.0, bounds.lora_dropout_max)
    lora_target_preset = trial.suggest_categorical(
        "lora_target_modules", bounds.lora_target_modules_choices
    )

    weight_decay = trial.suggest_float("weight_decay", 0.0, bounds.weight_decay_max)
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, bounds.warmup_ratio_max)
    lr_scheduler_type = trial.suggest_categorical(
        "lr_scheduler_type", bounds.lr_scheduler_choices
    )
    grad_accum = trial.suggest_categorical(
        "gradient_accumulation_steps", bounds.grad_accum_choices
    )
    cross_learning = trial.suggest_categorical(
        "cross_learning", bounds.cross_learning_choices
    )
    target_scaler = trial.suggest_categorical(
        "target_scaler", bounds.target_scaler_choices
    )
    if bounds.eval_metric is not None:
        eval_metric = bounds.eval_metric
    else:
        eval_metric = trial.suggest_categorical(
            "eval_metric", bounds.eval_metric_choices
        )

    # Covariate selection ------------------------------------------------
    # Insulin channels are routed either as past-only covariates (default) or as
    # future-known covariates (``insulin_as_known``). When routed to known, each
    # insulin channel gets its own ``use_known__{col}`` toggle so the sampled
    # parameters read as known-covariate switches, and only the remaining
    # (non-insulin) candidates are offered as ``use_past__*``.
    insulin_candidates = [
        col
        for col in bounds.past_candidates
        if PAST_COVARIATE_MASKS.get(col) == INSULIN_OBSERVED_COL
    ]
    if bounds.insulin_as_known:
        past_candidates = [
            col for col in bounds.past_candidates if col not in set(insulin_candidates)
        ]
    else:
        past_candidates = list(bounds.past_candidates)

    past_cols = [
        col
        for col in past_candidates
        if trial.suggest_categorical(f"use_past__{col}", [False, True])
    ]

    use_known_time = trial.suggest_categorical("use_known_time", [False, True])
    known_cols = list(KNOWN_COVARIATE_CANDIDATES) if use_known_time else []
    if bounds.insulin_as_known:
        insulin_known = [
            col
            for col in insulin_candidates
            if trial.suggest_categorical(f"use_known__{col}", [False, True])
        ]
        known_cols = list(dict.fromkeys([*known_cols, *insulin_known]))
    if bounds.include_meal:
        meal_known = [
            col
            for col in bounds.meal_known_cols
            if trial.suggest_categorical(f"use_known__{col}", [False, True])
        ]
        known_cols = list(dict.fromkeys([*known_cols, *meal_known]))
    if bounds.include_sex_known:
        known_cols = list(dict.fromkeys([*known_cols, SEX_KNOWN_COL]))
    if bounds.include_demographics_known:
        demo_known = [
            col
            for col in bounds.demographic_known_cols
            if trial.suggest_categorical(f"use_known__{col}", [False, True])
        ]
        known_cols = list(dict.fromkeys([*known_cols, *demo_known]))

    static_cols = [
        col
        for col in (*STATIC_COVARIATE_CANDIDATES, *PIECEWISE_STATIC_CANDIDATES)
        if col not in bounds.static_exclude
        and trial.suggest_categorical(f"use_static__{col}", [False, True])
    ]

    fine_tune_trainer_kwargs: Dict[str, Any] = {
        "weight_decay": weight_decay,
        "warmup_ratio": warmup_ratio,
        "lr_scheduler_type": lr_scheduler_type,
        "gradient_accumulation_steps": grad_accum,
    }
    if bounds.eval_steps is not None:
        # Override the pipeline's hardcoded 100; keep save_steps aligned so
        # load_best_model_at_end stays valid.
        fine_tune_trainer_kwargs["eval_steps"] = bounds.eval_steps
        fine_tune_trainer_kwargs["save_steps"] = bounds.eval_steps
    if bounds.max_grad_norm_min is not None and bounds.max_grad_norm_max is not None:
        fine_tune_trainer_kwargs["max_grad_norm"] = trial.suggest_float(
            "max_grad_norm",
            bounds.max_grad_norm_min,
            bounds.max_grad_norm_max,
            log=True,
        )
    if bounds.optim_choices:
        fine_tune_trainer_kwargs["optim"] = trial.suggest_categorical(
            "optim", bounds.optim_choices
        )

    lora_config: Dict[str, Any] = {
        "r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
    }
    target_modules = LORA_TARGET_MODULE_PRESETS.get(lora_target_preset)
    if target_modules is not None:
        lora_config["target_modules"] = target_modules

    config_kwargs: Dict[str, Any] = {
        "context_length": context_length,
        "forecast_length": 24,
        "training_mode": "fine_tune",
        "fine_tune_mode": "lora",
        "fine_tune_lr": fine_tune_lr,
        "fine_tune_steps": fine_tune_steps,
        "fine_tune_batch_size": fine_tune_batch_size,
        "fine_tune_context_length": context_length,
        "fine_tune_lora_config": lora_config,
        "extra_fine_tune_trainer_kwargs": fine_tune_trainer_kwargs,
        "cross_learning": cross_learning,
        "target_scaler": target_scaler,
        "fine_tune_eval_max_items": bounds.fine_tune_eval_max_items,
        "eval_metric": eval_metric,
        "enable_ensemble": False,
        "eval_during_fine_tune": True,
    }

    if bounds.min_past_choices:
        config_kwargs["min_past"] = trial.suggest_categorical(
            "min_past", bounds.min_past_choices
        )
    if bounds.covariate_scaler_choices:
        config_kwargs["covariate_scaler"] = trial.suggest_categorical(
            "covariate_scaler", bounds.covariate_scaler_choices
        )

    return SearchPoint(
        config_kwargs=config_kwargs,
        past_cols=past_cols,
        known_cols=known_cols,
        static_cols=static_cols,
        eval_metric=eval_metric,
    )
