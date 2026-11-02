# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)

"""Shared-study Optuna worker entrypoint for the Chronos-2 MetaboNet PoC.

A worker loads the MetaboNet dataset once, connects to a shared SQLite-backed
Optuna study, and contributes ``--n-trials`` trials. Multiple workers — each
pinned to a GPU via ``CUDA_VISIBLE_DEVICES`` and launched in its own tmux pane —
point at the same ``--storage`` and cooperatively explore one search space.

Usage (single worker)::

    CUDA_VISIBLE_DEVICES=0 python -m src.workflows.sweeps.optuna_chronos2.study \\
        run --config configs/experiments/nocturnal_forecast/chronos2_optuna_poc.yaml \\
        --n-trials 20 --gpu-label 0

Use the ``create`` subcommand once to initialise the study, then launch workers
(see ``scripts/orchestration/sweeps/optuna/launch_optuna_tmux.sh``).
"""

from __future__ import annotations

import argparse
import copy
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import optuna
import yaml

from .data import MetabonetData
from .objective import Chronos2Objective
from .pruning import build_pruner
from .search_space import SearchSpaceBounds

logger = logging.getLogger(__name__)

DEFAULT_CONFIG: Dict[str, Any] = {
    "study_name": "chronos2_metabonet_poc",
    "storage": "sqlite:///optuna_chronos2_poc.db",
    "direction": "minimize",
    "processed_path": "cache/data/metabonet/processed",
    "output_root": "trained_models/optuna_chronos2_poc",
    "time_limit": None,
    "keep_models": False,
    # Which logged metric Optuna minimizes ("rmse_120" default, or "mard_120").
    "primary_metric": "rmse_120",
    # Early-terminate weak trials during fine-tuning by reporting the HF
    # validation loss to Optuna every ~100 steps. `pruner` selects the strategy
    # ("median" | "asha" | "hyperband") and its parameters.
    "enable_pruning": False,
    "pruner": {},
    # Wall-clock budget per worker (seconds); None = run until N_TRIALS.
    "timeout": None,
    # Stop a worker early if the study's best value has not improved for this
    # many completed trials (0/None = disabled; keep exploring for ensembling).
    "early_stopping_patience": None,
    # HF trainer early stopping WITHIN a trial: stop fine-tuning once eval_loss
    # has not improved for this many evals (0/None = disabled). Saves compute on
    # configs that plateau well before their fine_tune_steps budget.
    "fine_tune_early_stopping_patience": None,
    "data": {
        "forecast_length": 24,
        "max_context": 288,
        "min_context": 48,
        "val_days": 2,
        "interval_mins": 5,
        "imputation_threshold_mins": 30,
        "eval_stride_steps": 3,
        "max_val_windows_per_patient": 8,
        "max_val_windows_total": 2000,
        "patient_limit": None,
        "seed": 42,
    },
    "search_space": {},
    "smoke": {
        "data": {
            "patient_limit": 8,
            "max_val_windows_per_patient": 3,
            "max_val_windows_total": 18,
        },
        "search_space": {
            "context_min": 48,
            "context_max": 96,
            "context_step": 48,
            "steps_choices": [10],
            "batch_size_choices": [8],
        },
    },
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: Optional[str], smoke: bool = False) -> Dict[str, Any]:
    """Load the PoC config, layering YAML overrides and smoke settings."""
    config = copy.deepcopy(DEFAULT_CONFIG)
    if path is not None:
        with open(path, "r", encoding="utf-8") as fh:
            config = _deep_merge(config, yaml.safe_load(fh) or {})
    if smoke:
        config = _deep_merge(config, config.get("smoke", {}))
    return config


def build_data(config: Dict[str, Any]) -> MetabonetData:
    data_cfg = dict(config["data"])
    return MetabonetData(config["processed_path"], **data_cfg).load()


def build_bounds(config: Dict[str, Any]) -> SearchSpaceBounds:
    return SearchSpaceBounds(**config.get("search_space", {}))


def _parse_objectives(config: Dict[str, Any]):
    """Return (metric_names, directions). Multi-objective iff len > 1.

    Uses the ``objectives`` list (each: {metric, direction}) when present,
    otherwise falls back to the single ``primary_metric`` + ``direction``.
    """
    objectives = config.get("objectives")
    if objectives:
        metrics = [str(o["metric"]) for o in objectives]
        directions = [str(o.get("direction", "minimize")) for o in objectives]
        return metrics, directions
    return [config.get("primary_metric", "rmse_120")], [
        config.get("direction", "minimize")
    ]


def _apply_dashboard_metadata(study: optuna.Study, config: Dict[str, Any]) -> None:
    """Label the objective(s) and attach study notes for the Optuna dashboard.

    Everything here is stored in the Optuna storage itself, so the VSCode Optuna
    Dashboard extension (or optuna-dashboard) shows a readable objective axis and
    a description panel without any extra tooling.
    """
    metric_names, directions = _parse_objectives(config)
    try:
        study.set_metric_names(metric_names)
    except Exception:  # pragma: no cover - older optuna
        pass

    search = config.get("search_space", {})
    data = config.get("data", {})
    objective_desc = ", ".join(f"{d} {m}" for m, d in zip(metric_names, directions))
    study.set_user_attr("objective", f"{objective_desc} on internal temporal split")
    study.set_user_attr(
        "forecast_shape",
        f"context<= {data.get('max_context', 288)} steps -> "
        f"{data.get('forecast_length', 24)}-step forecast (+120 min); MetaboNet shape",
    )
    study.set_user_attr(
        "validation",
        f"held-out last {data.get('val_days', 2)} days/patient; "
        f"patient_limit={data.get('patient_limit')}",
    )
    study.set_user_attr(
        "logged_metrics", "rmse/mard/dts at 30/60/90/120 stored as trial user attrs"
    )

    # Full search space, so the ranges are visible in the dashboard's study
    # "User Attributes" (trials only show sampled values, not the space).
    from .search_space import (
        INSULIN_OBSERVED_COL,
        PAST_COVARIATE_MASKS,
        PIECEWISE_STATIC_CANDIDATES,
        STATIC_COVARIATE_CANDIDATES,
        SearchSpaceBounds,
    )

    bounds = SearchSpaceBounds(**search) if search else SearchSpaceBounds()
    study.set_user_attr(
        "space__context_length",
        f"int[{bounds.context_min}..{bounds.context_max}] step {bounds.context_step}",
    )
    study.set_user_attr(
        "space__fine_tune_lr", f"loguniform[{bounds.lr_min}, {bounds.lr_max}]"
    )
    study.set_user_attr("space__fine_tune_steps", str(bounds.steps_choices))
    study.set_user_attr("space__fine_tune_batch_size", str(bounds.batch_size_choices))
    study.set_user_attr(
        "space__lora",
        f"r={bounds.lora_r_choices} alpha={bounds.lora_alpha_choices} "
        f"dropout[0,{bounds.lora_dropout_max}] targets={bounds.lora_target_modules_choices}",
    )
    study.set_user_attr(
        "space__optimizer",
        f"weight_decay[0,{bounds.weight_decay_max}] warmup[0,{bounds.warmup_ratio_max}] "
        f"scheduler={bounds.lr_scheduler_choices} grad_accum={bounds.grad_accum_choices}",
    )
    study.set_user_attr("space__cross_learning", str(bounds.cross_learning_choices))
    study.set_user_attr("space__target_scaler", str(bounds.target_scaler_choices))
    study.set_user_attr(
        "space__eval_metric",
        (
            f"fixed={bounds.eval_metric} (not searched; inert for mard_avg)"
            if bounds.eval_metric is not None
            else str(bounds.eval_metric_choices)
        ),
    )
    study.set_user_attr("space__eval_steps", str(bounds.eval_steps))
    # Reflect the EFFECTIVE covariate pools: the config can restrict
    # ``past_candidates`` and, when ``insulin_as_known`` is set, insulin channels
    # move out of the past pool into their own known-future ``use_known__*``
    # toggles (so they never appear as ``use_past__*`` parameters).
    insulin_candidates = [
        col
        for col in bounds.past_candidates
        if PAST_COVARIATE_MASKS.get(col) == INSULIN_OBSERVED_COL
    ]
    known_desc = "hour_sin, hour_cos (on/off together)"
    if bounds.insulin_as_known:
        past_pool = [
            col for col in bounds.past_candidates if col not in set(insulin_candidates)
        ]
        if insulin_candidates:
            known_desc += (
                "; insulin as known-future (per-channel use_known__*): "
                f"{insulin_candidates}"
            )
    else:
        past_pool = list(bounds.past_candidates)
    study.set_user_attr("space__past_covariates", str(past_pool))
    study.set_user_attr(
        "space__static_covariates",
        str([*STATIC_COVARIATE_CANDIDATES, *PIECEWISE_STATIC_CANDIDATES]),
    )
    study.set_user_attr("space__known_covariates", known_desc)
    study.set_user_attr("space__insulin_as_known", str(bounds.insulin_as_known))

    # Optional richer Markdown note when optuna-dashboard is installed.
    try:  # pragma: no cover - optional dependency
        import optuna_dashboard

        optuna_dashboard.save_note(
            study,
            "## Chronos-2 x Optuna (MetaboNet)\n"
            f"**Objective:** {objective_desc} on the internal temporal "
            "validation split.\n\n"
            "**Search:** LoRA fine-tuning of Chronos-2 over context length, LoRA "
            "config, optimizer/scheduler, covariates, and target scaling.\n\n"
            "Per-horizon RMSE/MARD/DTS (30/60/90/120) are on each trial's "
            "*User Attributes* tab.",
        )
    except Exception:
        pass


def _ensure_storage_dir(storage: str) -> None:
    """Create the parent directory for a sqlite:/// storage path if needed."""
    prefix = "sqlite:///"
    if storage.startswith(prefix):
        db_path = Path(storage[len(prefix) :])
        db_path.parent.mkdir(parents=True, exist_ok=True)


def _build_storage(config: Dict[str, Any]):
    """Build an RDB storage with heartbeat so crashed trials auto-fail/retry.

    Without a heartbeat, Optuna cannot tell a dead worker's trial from a slow
    one, so killed-worker trials linger forever as RUNNING. Heartbeat lets any
    live worker mark stale trials FAILED (grace_period after the last beat) and
    RetryFailedTrialCallback re-enqueues them.
    """
    from optuna.storages import RDBStorage, RetryHeartbeatStaleTrialCallback

    url = config["storage"]
    _ensure_storage_dir(url)
    heartbeat_interval = int(config.get("heartbeat_interval", 60))
    grace_period = int(config.get("grace_period", 300))
    max_retry = config.get("max_retry", 2)
    return RDBStorage(
        url=url,
        heartbeat_interval=heartbeat_interval,
        grace_period=grace_period,
        heartbeat_stale_trial_callback=RetryHeartbeatStaleTrialCallback(
            max_retry=max_retry
        ),
    )


def _build_sampler(config: Dict[str, Any]):
    """Build the Optuna sampler from config (default multivariate TPE).

    Options: "tpe" (default), "nsga2" (genetic/population-based; usable for
    single-objective and the natural choice if we go multi-objective), "random",
    or "cmaes". TPE is the strong default for our large mixed/conditional space.
    """
    spec = config.get("sampler", {}) or {}
    kind = str(spec.get("type", "tpe")).lower()
    seed = spec.get("seed")
    if seed is not None:
        # CRITICAL for distributed runs: workers must NOT share a seed, or they
        # sample identical initial trials (and stay synchronized for several
        # generations), wasting compute. Offset by a per-process value so each
        # worker's sampler diverges while a base seed still aids debuggability.
        seed = int(seed) + (os.getpid() % 100_000)
    if kind == "nsga2":
        return optuna.samplers.NSGAIISampler(
            population_size=spec.get("population_size", 50), seed=seed
        )
    if kind == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    if kind == "cmaes":
        return optuna.samplers.CmaEsSampler(seed=seed)
    return optuna.samplers.TPESampler(multivariate=True, group=True, seed=seed)


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
        # Optuna cannot prune multi-objective studies; ignore any pruner config.
        create_kwargs["directions"] = directions
    else:
        create_kwargs["direction"] = directions[0]
        create_kwargs["pruner"] = build_pruner(config)
    study = optuna.create_study(**create_kwargs)
    # Label objectives so the dashboard shows metric names (e.g. "mard_120")
    # instead of "Objective 0/1". Idempotent and safe to re-apply on resume.
    try:
        study.set_metric_names(list(metric_names))
    except Exception:  # noqa: BLE001 - older storages may not support it
        pass
    _apply_dashboard_metadata(study, config)
    return study


def cmd_create(config: Dict[str, Any]) -> int:
    study = _create_study(config)
    print(f"Study '{study.study_name}' ready at {config['storage']}")
    print(f"Existing trials: {len(study.trials)}")
    return 0


def _make_early_stopping_callback(patience: int):
    """Callback that stops the worker if best value stalls for `patience` trials."""
    state = {"best": None, "since_improve": 0}

    def _callback(study: optuna.Study, _trial: optuna.trial.FrozenTrial) -> None:
        try:
            current = study.best_value
        except ValueError:
            return
        if state["best"] is None or current < state["best"] - 1e-9:
            state["best"] = current
            state["since_improve"] = 0
        else:
            state["since_improve"] += 1
            if state["since_improve"] >= patience:
                logger.info(
                    "Early stopping: no improvement in %d trials (best=%.4f).",
                    patience,
                    state["best"],
                )
                study.stop()

    return _callback


def cmd_run(config: Dict[str, Any], n_trials: int, gpu_label: str) -> int:
    data = build_data(config)
    bounds = build_bounds(config)
    metric_names, _ = _parse_objectives(config)
    is_multi = len(metric_names) > 1
    objective = Chronos2Objective(
        data,
        config["output_root"],
        bounds=bounds,
        time_limit=config.get("time_limit"),
        keep_models=bool(config.get("keep_models", False)),
        gpu_label=gpu_label or os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        primary_metric=metric_names[0],
        objective_metrics=metric_names if is_multi else None,
        enable_pruning=bool(config.get("enable_pruning", False)),
        report_intermediate=bool(config.get("report_intermediate", True)),
        early_stopping_patience=config.get("fine_tune_early_stopping_patience"),
    )
    study = _create_study(config)

    timeout = config.get("timeout")
    # n_trials <= 0 means "unlimited until timeout" (requires a timeout).
    effective_n_trials: Optional[int] = n_trials if n_trials and n_trials > 0 else None
    if effective_n_trials is None and timeout is None:
        raise ValueError(
            "Set a positive --n-trials or a config 'timeout' so the worker terminates."
        )

    callbacks = []
    patience = config.get("early_stopping_patience")
    if patience and not is_multi:
        # study.best_value is undefined for multi-objective; skip the callback.
        callbacks.append(_make_early_stopping_callback(int(patience)))

    study.optimize(
        objective,
        n_trials=effective_n_trials,
        timeout=timeout,
        gc_after_trial=True,
        callbacks=callbacks,
        # Resilience: a single failing trial (e.g. an unlucky hyperparameter
        # combo that errors during fit) should mark that trial FAILED and let
        # the worker keep going, not tear down the whole worker. Optuna still
        # logs each failure at WARNING level, so failures stay visible.
        catch=(Exception,),
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
            print("No completed trials yet.")
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
    logging.basicConfig(level=logging.INFO)
    args = _build_parser().parse_args(argv)
    config = load_config(args.config, smoke=args.smoke)

    # CLI overrides take precedence over config/defaults.
    if args.study_name:
        config["study_name"] = args.study_name
    if args.storage:
        config["storage"] = args.storage
    if args.output_root:
        config["output_root"] = args.output_root
    if args.keep_models:
        config["keep_models"] = True
    if args.patient_limit is not None:
        config["data"]["patient_limit"] = args.patient_limit

    if args.command == "create":
        return cmd_create(config)
    return cmd_run(config, n_trials=args.n_trials, gpu_label=args.gpu_label)


if __name__ == "__main__":
    raise SystemExit(main())
