# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)

"""CLI for the Moirai-2.0 / Moirai-MoE PEFT-LoRA x Optuna sweep.

Each worker must run in the ``.venvs/moirai`` interpreter (uni2ts + gluonts +
``peft``). Usage::

    python -m src.workflows.sweeps.optuna_moirai2.study create --config <yaml>
    python -m src.workflows.sweeps.optuna_moirai2.study run --config <yaml> \
        --n-trials 40 --gpu-label 0
    # tiny end-to-end smoke (1 trial, few patients):
    python -m src.workflows.sweeps.optuna_moirai2.study run --config <yaml> \
        --smoke --n-trials 1
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Any, Dict, Optional, Sequence

import optuna
import yaml

from ..optuna_chronos2.pruning import build_pruner
from ..optuna_chronos2.study import (
    _build_sampler,
    _build_storage,
    _deep_merge,
    _make_early_stopping_callback,
    _parse_objectives,
    build_data,
)
from .objective import Moirai2Objective
from .search_space import Moirai2SearchSpaceBounds

logger = logging.getLogger(__name__)

DEFAULT_CONFIG: Dict[str, Any] = {
    "study_name": "moirai2_metabonet_poc",
    "storage": "sqlite:///results/optuna/chronos2_studies.db",
    "processed_path": "cache/data/metabonet/processed",
    "output_root": "results/optuna/moirai2_metabonet_poc_models",
    "primary_metric": "mard_avg",
    "direction": "minimize",
    "time_limit": None,
    "timeout": None,
    "keep_models": False,
    "report_intermediate": True,
    "enable_pruning": True,
    "pruner": {"type": "median", "n_startup_trials": 8, "n_warmup_steps": 1},
    "early_stopping_patience": None,
    "heartbeat_interval": 60,
    "grace_period": 600,
    "max_retry": 2,
    "data": {
        "forecast_length": 24,
        "max_context": 288,
        "min_context": 96,
        "val_days": 2,
        "interval_mins": 5,
        "imputation_threshold_mins": 30,
        "eval_stride_steps": 3,
        "max_val_windows_per_patient": 8,
        "max_val_windows_total": 2000,
        "patient_limit": 100,
        "source_filter": ["iobp2"],
        "seed": 42,
    },
    "fine_tune": {
        "train_tail_days": 21,
        "val_windows_per_patient": 4,
        "max_val_windows_total": 2000,
        "val_stride_steps": 6,
        "seed": 42,
    },
    "search_space": {},
    "smoke": {
        "data": {"patient_limit": 6, "min_context": 96, "max_context": 96},
        "fine_tune": {
            "train_tail_days": 10,
            "val_windows_per_patient": 2,
            "max_val_windows_total": 16,
            "val_stride_steps": 6,
        },
        "search_space": {
            "context_min": 96,
            "context_max": 96,
            "context_step": 96,
            "lora_r_choices": [8],
            "lora_alpha_choices": [16],
            "lora_target_choices": ["attn"],
            "num_epochs_choices": [1],
            "batch_size_choices": [16],
            "beta2_choices": [0.999],
            "gradient_clip_val_choices": [1.0],
            "known_kinetics_candidates": [],
        },
    },
}


def load_config(path: Optional[str], smoke: bool = False) -> Dict[str, Any]:
    config = {k: v for k, v in DEFAULT_CONFIG.items() if k != "smoke"}
    if path:
        with open(path) as fh:
            config = _deep_merge(config, yaml.safe_load(fh) or {})
    if smoke:
        smoke_over = _deep_merge(dict(DEFAULT_CONFIG["smoke"]), config.get("smoke", {}))
        config = _deep_merge(config, smoke_over)
    return config


def build_bounds(config: Dict[str, Any]) -> Moirai2SearchSpaceBounds:
    return Moirai2SearchSpaceBounds(**config.get("search_space", {}))


def _apply_dashboard_metadata(study: optuna.Study, config: Dict[str, Any]) -> None:
    metric_names, directions = _parse_objectives(config)
    try:
        study.set_metric_names(list(metric_names))
    except Exception:  # noqa: BLE001
        pass
    bounds = build_bounds(config)
    objective_desc = ", ".join(f"{d} {m}" for m, d in zip(metric_names, directions))
    study.set_user_attr("model", f"{bounds.variant} (PEFT-LoRA fine-tune)")
    study.set_user_attr("objective", f"{objective_desc} on internal temporal split")
    purpose = config.get("purpose")
    if purpose:
        study.set_user_attr("purpose", str(purpose))
    study.set_user_attr("space__variant", bounds.variant)
    study.set_user_attr("space__model_path", str(bounds.model_path_choices))
    study.set_user_attr(
        "space__lora",
        f"r={bounds.lora_r_choices} alpha={bounds.lora_alpha_choices} "
        f"dropout[{bounds.lora_dropout_min},{bounds.lora_dropout_max}] "
        f"targets={bounds.lora_target_choices}",
    )
    study.set_user_attr("space__lr", f"loguniform[{bounds.lr_min}, {bounds.lr_max}]")
    study.set_user_attr("space__num_epochs", str(bounds.num_epochs_choices))
    study.set_user_attr("space__batch_size", str(bounds.batch_size_choices))
    study.set_user_attr(
        "space__known_covariates",
        f"always: {bounds.always_known_candidates}; "
        f"toggled kinetics: {bounds.known_kinetics_candidates}; demographics excluded",
    )


def _create_study(config: Dict[str, Any]) -> optuna.Study:
    metric_names, directions = _parse_objectives(config)
    is_multi = len(metric_names) > 1
    create_kwargs: Dict[str, Any] = dict(
        study_name=config["study_name"],
        storage=_build_storage(config),
        load_if_exists=True,
        sampler=_build_sampler(config),
    )
    if is_multi:
        create_kwargs["directions"] = directions
    else:
        create_kwargs["direction"] = directions[0]
        create_kwargs["pruner"] = build_pruner(config)
    study = optuna.create_study(**create_kwargs)
    _apply_dashboard_metadata(study, config)
    return study


def cmd_create(config: Dict[str, Any]) -> int:
    study = _create_study(config)
    print(f"Study '{study.study_name}' ready at {config['storage']}")
    print(f"Existing trials: {len(study.trials)}")
    return 0


def cmd_run(config: Dict[str, Any], n_trials: int, gpu_label: str) -> int:
    data = build_data(config)
    bounds = build_bounds(config)
    metric_names, _ = _parse_objectives(config)
    is_multi = len(metric_names) > 1
    fine_tune = dict(config.get("fine_tune", {}))
    objective = Moirai2Objective(
        data,
        config["output_root"],
        bounds=bounds,  # type: ignore[arg-type]  # Moirai2 bounds used via _suggest
        time_limit=config.get("time_limit"),
        keep_models=bool(config.get("keep_models", False)),
        gpu_label=gpu_label or os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        primary_metric=metric_names[0],
        objective_metrics=metric_names if is_multi else None,
        report_intermediate=bool(config.get("report_intermediate", False)),
        **fine_tune,
    )
    study = _create_study(config)

    timeout = config.get("timeout")
    effective_n_trials: Optional[int] = n_trials if n_trials and n_trials > 0 else None
    if effective_n_trials is None and timeout is None:
        raise ValueError(
            "Set a positive --n-trials or a config 'timeout' so the worker terminates."
        )

    callbacks = []
    patience = config.get("early_stopping_patience")
    if patience and not is_multi:
        callbacks.append(_make_early_stopping_callback(int(patience)))

    study.optimize(
        objective,
        n_trials=effective_n_trials,
        timeout=timeout,
        gc_after_trial=True,
        callbacks=callbacks,
    )

    print(f"Worker finished. Study now has {len(study.trials)} trials.")
    try:
        best = study.best_trial
        print(f"Best {metric_names[0]}: {best.value:.3f} (trial {best.number})")
    except ValueError:
        pass
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Moirai-2/MoE LoRA Optuna sweep")
    parser.add_argument("command", choices=["create", "run"], help="Action to run.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--gpu-label", type=str, default="")
    parser.add_argument("--smoke", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    config = load_config(args.config, smoke=args.smoke)
    if args.command == "create":
        return cmd_create(config)
    return cmd_run(config, n_trials=args.n_trials, gpu_label=args.gpu_label)


if __name__ == "__main__":
    raise SystemExit(main())
