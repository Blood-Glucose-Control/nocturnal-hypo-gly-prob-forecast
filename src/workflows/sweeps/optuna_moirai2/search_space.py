# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)

"""Search space for the Moirai-2.0 / Moirai-MoE PEFT-LoRA x Optuna sweep.

Neither model has a native uni2ts fine-tune module, so each trial LoRA-fine-tunes
the model via :class:`~src.models.moirai2.model.Moirai2LoraForecaster`. Only
insulin-derived, time-varying known-future covariates are used (raw dose + IOB /
insulin-activity kinetics). Constant per-patient demographics are intentionally
excluded: Moirai normalizes covariate channels per window, so a constant channel
is washed out (unlike Chronos-2 / TimesFM-3) — see the scoping run-book.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import optuna

from ..optuna_moirai.search_space import MoiraiSearchPoint

# HF checkpoints per variant.
MOIRAI2_MODELS: tuple[str, ...] = ("Salesforce/moirai-2.0-R-small",)
MOIRAI_MOE_MODELS: tuple[str, ...] = (
    "Salesforce/moirai-moe-1.0-R-small",
    "Salesforce/moirai-moe-1.0-R-base",
)

# Named LoRA target presets (categorical choice -> module list).
LORA_TARGET_PRESETS: Dict[str, List[str]] = {
    "attn": ["q_proj", "k_proj", "v_proj", "out_proj"],
    "attn_qv": ["q_proj", "v_proj"],
}

ALWAYS_KNOWN_INSULIN: tuple[str, ...] = (
    "total_insulin",
    "basal_insulin",
    "bolus_insulin",
)


@dataclass
class Moirai2SearchSpaceBounds:
    """Tunable bounds for the Moirai-2/MoE LoRA search space (YAML-overridable)."""

    variant: str = "moirai2"  # "moirai2" | "moirai_moe"
    forecast_length: int = 24
    interval_mins: int = 5
    num_samples: int = 100
    model_path_choices: List[str] = field(default_factory=lambda: list(MOIRAI2_MODELS))
    # Context fixed at the 288 (24h) contest maximum.
    context_min: int = 288
    context_max: int = 288
    context_step: int = 96
    # Both R-small checkpoints use a fixed patch of 16.
    patch_size: int = 16
    # LoRA knobs.
    lora_r_choices: List[int] = field(default_factory=lambda: [8, 16, 32, 64])
    lora_alpha_choices: List[int] = field(default_factory=lambda: [16, 32, 64, 128])
    lora_dropout_min: float = 0.0
    lora_dropout_max: float = 0.2
    lora_target_choices: List[str] = field(default_factory=lambda: ["attn", "attn_qv"])
    # Optimizer.
    lr_min: float = 1e-5
    lr_max: float = 1e-3
    weight_decay_min: float = 1e-8
    weight_decay_max: float = 1e-2
    beta1: float = 0.9
    beta2_choices: List[float] = field(default_factory=lambda: [0.95, 0.999])
    gradient_clip_val_choices: List[float] = field(default_factory=lambda: [1.0, 2.0])
    # Fine-tune schedule.
    num_epochs_choices: List[int] = field(default_factory=lambda: [2, 4, 8])
    batch_size_choices: List[int] = field(default_factory=lambda: [16, 32])
    # Time-varying known-future covariates (demographics excluded by design).
    always_known_candidates: List[str] = field(
        default_factory=lambda: list(ALWAYS_KNOWN_INSULIN)
    )
    known_kinetics_candidates: List[str] = field(
        default_factory=lambda: ["iob_exp", "insulin_activity_exp"]
    )
    eval_metric: str = "mard_avg"


def suggest_moirai2_point(
    trial: optuna.Trial, bounds: Moirai2SearchSpaceBounds
) -> MoiraiSearchPoint:
    """Sample one Moirai-2/MoE LoRA fine-tune configuration."""
    model_path = trial.suggest_categorical("model_path", bounds.model_path_choices)
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
    lora_target = trial.suggest_categorical("lora_target", bounds.lora_target_choices)
    lr = trial.suggest_float("lr", bounds.lr_min, bounds.lr_max, log=True)
    weight_decay = trial.suggest_float(
        "weight_decay", bounds.weight_decay_min, bounds.weight_decay_max, log=True
    )
    beta2 = trial.suggest_categorical("beta2", bounds.beta2_choices)
    gradient_clip_val = trial.suggest_categorical(
        "gradient_clip_val", bounds.gradient_clip_val_choices
    )
    num_epochs = trial.suggest_categorical("num_epochs", bounds.num_epochs_choices)
    batch_size = trial.suggest_categorical("batch_size", bounds.batch_size_choices)

    known_cols: List[str] = list(dict.fromkeys(bounds.always_known_candidates))
    for col in bounds.known_kinetics_candidates:
        if trial.suggest_categorical(f"use_known__{col}", [False, True]):
            known_cols = list(dict.fromkeys([*known_cols, col]))

    config_kwargs: Dict[str, Any] = {
        "variant": bounds.variant,
        "model_path": model_path,
        "context_length": context_length,
        "forecast_length": bounds.forecast_length,
        "training_mode": "fine_tune",
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "beta1": bounds.beta1,
        "beta2": beta2,
        "gradient_clip_val": gradient_clip_val,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "patch_size": bounds.patch_size,
        "num_samples": bounds.num_samples,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "lora_target_modules": LORA_TARGET_PRESETS[lora_target],
        "past_covariate_dim": 0,
        "covariate_cols": [],
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
        past_cols=[],
        known_cols=list(known_cols),
    )
