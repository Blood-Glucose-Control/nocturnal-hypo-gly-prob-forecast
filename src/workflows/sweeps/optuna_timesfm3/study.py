# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)

"""Worker entrypoint for the TimesFM 3.0 (LoRA fine-tune) x Optuna sweep.

Reuses the Chronos-2 sweep's generic study infrastructure (storage with
heartbeat, sampler selection, objective parsing, worker-level early stopping)
and the shared MetaboNet data pipeline; only the search space + objective are
TimesFM 3.0-specific.

Each trial is a full LoRA fine-tune (minutes on GPU), so this worker must run in
the ``.venvs/timesfm3`` interpreter (``timesfm3`` + ``peft`` + ``torch``) with
``TORCH_DISABLE_NATIVE_JIT=1`` set on hosts lacking the Triton ``bmm`` build path.

Usage:
    python -m src.workflows.sweeps.optuna_timesfm3.study create --config <yaml>
    python -m src.workflows.sweeps.optuna_timesfm3.study run --config <yaml> \
        --n-trials 40 --gpu-label 0
    # tiny end-to-end smoke (1 trial, few windows, 1 epoch):
    python -m src.workflows.sweeps.optuna_timesfm3.study run --config <yaml> \
        --smoke --n-trials 1
"""

from __future__ import annotations

import argparse
import copy
import logging
import os
from typing import Any, Dict, Optional, Sequence

import numpy as np
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
from .objective import TimesFM3Objective
from .search_space import TimesFM3SearchSpaceBounds

logger = logging.getLogger(__name__)

DEFAULT_CONFIG: Dict[str, Any] = {
    "study_name": "timesfm3_metabonet_poc",
    "storage": "sqlite:///results/optuna/timesfm3_metabonet_poc.db",
    "processed_path": "cache/data/metabonet/processed",
    "output_root": "results/optuna/timesfm3_metabonet_poc_models",
    "primary_metric": "mard_avg",
    "direction": "minimize",
    "time_limit": None,
    "timeout": None,
    "keep_models": False,
    "report_intermediate": True,
    "enable_pruning": True,
    "pruner": {"type": "median", "n_startup_trials": 8, "n_warmup_steps": 1},
    "early_stopping_patience": None,  # worker-level (stop worker if study stalls)
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
        "patient_limit": None,
        "seed": 42,
    },
    # TimesFM 3.0 fine-tune / windowing knobs (per trial).
    "fine_tune": {
        "model_path": "google/timesfm-3.0-pytorch",
        "device": "cuda",
        "train_windows_per_patient": 24,
        "max_train_windows": 6000,
        "val_windows_per_patient": 4,
        "max_val_windows_total": 2000,
        "val_stride_steps": 6,
        "es_val_fraction": 0.2,
        "es_patience": 1,
        "covariate_jitter": 0.001,
        "grad_clip_norm": 1.0,
        "seed": 42,
        "adapters_dir": None,
        "keep_top_k": 0,
    },
    "search_space": {},
    "smoke": {
        "data": {
            "patient_limit": 8,
            "min_context": 288,
            "max_context": 288,
        },
        "fine_tune": {
            "train_windows_per_patient": 3,
            "max_train_windows": 40,
            "val_windows_per_patient": 2,
            "max_val_windows_total": 16,
            "val_stride_steps": 6,
            "es_val_fraction": 0.25,
            "es_patience": 1,
        },
        "search_space": {
            "context_min": 96,
            "context_max": 96,
            "context_step": 32,
            "lora_r_choices": [4],
            "lora_alpha_choices": [8],
            "lora_dropout_min": 0.0,
            "lora_dropout_max": 0.0,
            "lora_target_modules_choices": ["all-linear"],
            "use_dora_choices": [False],
            "use_rslora_choices": [False],
            "use_znorm_choices": [False],
            "lr_schedule_choices": ["constant"],
            "max_epochs_choices": [1],
            "batch_size_choices": [8],
        },
    },
}


def load_config(path: Optional[str], smoke: bool = False) -> Dict[str, Any]:
    config = copy.deepcopy(DEFAULT_CONFIG)
    if path is not None:
        with open(path, "r", encoding="utf-8") as fh:
            config = _deep_merge(config, yaml.safe_load(fh) or {})
    if smoke:
        config = _deep_merge(config, config.get("smoke", {}))
    return config


def build_bounds(config: Dict[str, Any]) -> TimesFM3SearchSpaceBounds:
    return TimesFM3SearchSpaceBounds(**config.get("search_space", {}))


def _apply_dashboard_metadata(study: optuna.Study, config: Dict[str, Any]) -> None:
    """Attach a compact human-readable summary of the TimesFM 3.0 search space."""
    metric_names, directions = _parse_objectives(config)
    try:
        study.set_metric_names(list(metric_names))
    except Exception:  # noqa: BLE001 - older storages may not support it
        pass

    bounds = build_bounds(config)
    data = config.get("data", {})
    ft = config.get("fine_tune", {})
    objective_desc = ", ".join(f"{d} {m}" for m, d in zip(metric_names, directions))
    study.set_user_attr("model", "TimesFM-3.0 (LoRA fine-tune)")
    study.set_user_attr("objective", f"{objective_desc} on internal temporal split")
    purpose = config.get("purpose")
    if purpose:
        study.set_user_attr("purpose", str(purpose))
    study.set_user_attr(
        "forecast_shape",
        f"context in [{bounds.context_min}, {bounds.context_max}] steps -> "
        f"{bounds.forecast_length}-step forecast (+120 min); known-future "
        "covariates span context + horizon",
    )
    study.set_user_attr(
        "validation",
        f"held-out last {data.get('val_days', 2)} days/patient; "
        f"patient_limit={data.get('patient_limit')}; "
        f"val_windows_per_patient={ft.get('val_windows_per_patient')}",
    )
    study.set_user_attr(
        "logged_metrics", "rmse/mard/dts at 30/60/90/120 stored as trial user attrs"
    )
    study.set_user_attr(
        "space__context_length",
        f"int[{bounds.context_min}, {bounds.context_max}] step {bounds.context_step}",
    )
    study.set_user_attr("space__lora_r", str(bounds.lora_r_choices))
    study.set_user_attr("space__lora_alpha", str(bounds.lora_alpha_choices))
    study.set_user_attr(
        "space__lora_dropout",
        f"uniform[{bounds.lora_dropout_min}, {bounds.lora_dropout_max}]",
    )
    study.set_user_attr(
        "space__lora_target_modules", str(bounds.lora_target_modules_choices)
    )
    study.set_user_attr(
        "space__lora_variants",
        f"use_dora={bounds.use_dora_choices} use_rslora={bounds.use_rslora_choices}",
    )
    study.set_user_attr("space__use_znorm", str(bounds.use_znorm_choices))
    study.set_user_attr("space__lr", f"loguniform[{bounds.lr_min}, {bounds.lr_max}]")
    study.set_user_attr(
        "space__weight_decay",
        f"loguniform[{bounds.weight_decay_min}, {bounds.weight_decay_max}]",
    )
    study.set_user_attr(
        "space__lr_schedule",
        f"{bounds.lr_schedule_choices}; warmup_ratio "
        f"uniform[{bounds.warmup_ratio_min}, {bounds.warmup_ratio_max}]",
    )
    study.set_user_attr("space__max_epochs", str(bounds.max_epochs_choices))
    study.set_user_attr("space__batch_size", str(bounds.batch_size_choices))
    study.set_user_attr(
        "space__eval_metric", f"fixed={bounds.eval_metric} (not searched)"
    )
    study.set_user_attr(
        "space__known_covariates",
        "hour_sin, hour_cos (on/off together); insulin as known-future "
        f"(per-channel use_known__*): {bounds.known_candidates}; "
        f"always-on: {bounds.always_known_cols}; "
        f"sex-known={bounds.include_sex}; "
        f"demographic bundle (use_demographics): {bounds.demographic_bundle}",
    )
    study.set_user_attr("space__insulin_as_known", str(bounds.insulin_as_known))


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
    try:
        study.set_metric_names(list(metric_names))
    except Exception:  # noqa: BLE001 - older storages may not support it
        pass
    _apply_dashboard_metadata(study, config)
    return study


def _warm_start(study: optuna.Study, config: Dict[str, Any]) -> None:
    """Seed a fresh single-objective study with a prior study's trials."""
    source = config.get("warm_start_from")
    if not source or len(study.trials) > 0:
        return
    primary = _parse_objectives(config)[0][0]
    src = optuna.load_study(study_name=str(source), storage=_build_storage(config))
    added = 0
    for t in src.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        value = t.user_attrs.get(primary)
        if value is None or not np.isfinite(value):
            continue
        study.add_trial(
            optuna.trial.create_trial(
                state=optuna.trial.TrialState.COMPLETE,
                params=t.params,
                distributions=t.distributions,
                value=float(value),
            )
        )
        added += 1
    print(f"Warm-started {added} trials from '{source}' (metric={primary}).")


def cmd_create(config: Dict[str, Any]) -> int:
    study = _create_study(config)
    _warm_start(study, config)
    print(f"Study '{study.study_name}' ready at {config['storage']}")
    print(f"Existing trials: {len(study.trials)}")
    return 0


def cmd_run(config: Dict[str, Any], n_trials: int, gpu_label: str) -> int:
    data = build_data(config)
    bounds = build_bounds(config)
    metric_names, _ = _parse_objectives(config)
    is_multi = len(metric_names) > 1
    fine_tune = dict(config.get("fine_tune", {}))
    objective = TimesFM3Objective(
        data,
        config["output_root"],
        bounds=bounds,
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
    if is_multi:
        front = study.best_trials
        print(f"Pareto front ({', '.join(metric_names)}): {len(front)} trials")
        for t in front[:5]:
            vals = ", ".join(f"{m}={v:.3f}" for m, v in zip(metric_names, t.values))
            print(f"  trial {t.number}: {vals}")
    else:
        try:
            best = study.best_trial
            print(f"Best {metric_names[0]}: {best.value:.3f} (trial {best.number})")
        except ValueError:
            pass
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["create", "run"], help="Action to run.")
    parser.add_argument("--config", type=str, default=None, help="YAML config path.")
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--gpu-label", type=str, default="")
    parser.add_argument("--study-name", type=str, default=None)
    parser.add_argument("--storage", type=str, default=None)
    parser.add_argument("--patient-limit", type=int, default=None)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--keep-models", action="store_true")
    parser.add_argument(
        "--smoke", action="store_true", help="Use tiny smoke-test settings."
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    args = _build_parser().parse_args(argv)
    config = load_config(args.config, smoke=args.smoke)
    if args.study_name:
        config["study_name"] = args.study_name
    if args.storage:
        config["storage"] = args.storage
    if args.output_root:
        config["output_root"] = args.output_root
    if args.patient_limit is not None:
        config["data"]["patient_limit"] = args.patient_limit
    if args.keep_models:
        config["keep_models"] = True

    if args.command == "create":
        return cmd_create(config)
    return cmd_run(config, n_trials=args.n_trials, gpu_label=args.gpu_label)


if __name__ == "__main__":
    raise SystemExit(main())
