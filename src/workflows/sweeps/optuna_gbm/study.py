# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)

"""Worker entrypoint for the GBM (gradient-boosted tree) x Optuna sweep.

Reuses the Chronos-2 sweep's generic study infrastructure (storage with
heartbeat, sampler selection, objective parsing, worker-level early stopping)
and the shared MetaboNet data pipeline; only the search space + objective are
GBM-specific. The tabular GBM backends are CPU-only, so this sweep is designed
to run concurrently with GPU-bound sweeps.

Usage:
    python -m src.workflows.sweeps.optuna_gbm.study create --config <yaml>
    python -m src.workflows.sweeps.optuna_gbm.study run --config <yaml> \
        --n-trials 200 --gpu-label cpu0
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
from .objective import GBMObjective
from .search_space import GBMSearchSpaceBounds

logger = logging.getLogger(__name__)

DEFAULT_CONFIG: Dict[str, Any] = {
    "study_name": "gbm_metabonet_poc",
    "storage": "sqlite:///results/optuna/gbm_metabonet_poc.db",
    "processed_path": "cache/data/metabonet/processed",
    "output_root": "results/optuna/gbm_metabonet_poc_models",
    "primary_metric": "mard_avg",
    "direction": "minimize",
    "time_limit": None,
    "timeout": None,
    "keep_models": False,
    "num_cpus": None,  # per-worker CPU budget (None -> AutoGluon auto-detect)
    "enable_pruning": False,
    "pruner": {},
    "early_stopping_patience": None,  # worker-level (stop worker if study stalls)
    "heartbeat_interval": 60,
    "grace_period": 300,
    "max_retry": 2,
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
        "num_cpus": 2,
        "data": {
            "patient_limit": 8,
            "max_val_windows_per_patient": 3,
            "max_val_windows_total": 18,
        },
        "search_space": {
            "num_boost_round": 200,
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


def build_bounds(config: Dict[str, Any]) -> GBMSearchSpaceBounds:
    return GBMSearchSpaceBounds(**config.get("search_space", {}))


def _apply_dashboard_metadata(study: optuna.Study, config: Dict[str, Any]) -> None:
    """Attach a compact human-readable summary of the GBM search space."""
    metric_names, directions = _parse_objectives(config)
    try:
        study.set_metric_names(list(metric_names))
    except Exception:  # noqa: BLE001 - older storages may not support it
        pass

    bounds = build_bounds(config)
    data = config.get("data", {})
    objective_desc = ", ".join(f"{d} {m}" for m, d in zip(metric_names, directions))
    study.set_user_attr("model", "GBM (RecursiveTabular/DirectTabular)")
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
    study.set_user_attr("space__strategy", str(bounds.strategy_choices))
    study.set_user_attr("space__backend", str(bounds.backend_choices))
    study.set_user_attr("space__lag_set", str(bounds.lag_set_choices))
    study.set_user_attr("space__target_scaler", str(bounds.target_scaler_choices))
    study.set_user_attr(
        "space__use_first_difference", str(bounds.use_first_difference_choices)
    )
    study.set_user_attr(
        "space__num_boost_round",
        f"fixed={bounds.num_boost_round} (early stopping picks effective count)",
    )
    study.set_user_attr(
        "space__learning_rate",
        f"loguniform[{bounds.learning_rate_min}, {bounds.learning_rate_max}]",
    )
    study.set_user_attr(
        "space__lightgbm",
        f"num_leaves[{bounds.num_leaves_min}, {bounds.num_leaves_max}] "
        f"feature_fraction[{bounds.feature_fraction_min}, {bounds.feature_fraction_max}] "
        f"min_data_in_leaf[{bounds.min_data_in_leaf_min}, {bounds.min_data_in_leaf_max}] "
        f"bagging_fraction[{bounds.bagging_fraction_min}, {bounds.bagging_fraction_max}] "
        f"lambda_l1/l2 (log)",
    )
    study.set_user_attr(
        "space__catboost",
        f"depth[{bounds.depth_min}, {bounds.depth_max}] "
        f"l2_leaf_reg[{bounds.l2_leaf_reg_min}, {bounds.l2_leaf_reg_max}] "
        f"random_strength (log) bagging_temperature"
        f"[{bounds.bagging_temperature_min}, {bounds.bagging_temperature_max}]",
    )
    study.set_user_attr(
        "space__xgboost",
        f"max_depth[{bounds.max_depth_min}, {bounds.max_depth_max}] "
        f"min_child_weight[{bounds.min_child_weight_min}, {bounds.min_child_weight_max}] "
        f"subsample[{bounds.subsample_min}, {bounds.subsample_max}] "
        f"colsample_bytree[{bounds.colsample_bytree_min}, {bounds.colsample_bytree_max}] "
        f"reg_alpha/reg_lambda/gamma (log)",
    )
    study.set_user_attr(
        "space__eval_metric", f"fixed={bounds.eval_metric} (not searched)"
    )
    study.set_user_attr(
        "space__known_covariates",
        "hour_sin, hour_cos (on/off together); insulin as known-future "
        f"(per-channel use_known__*): {bounds.known_candidates}; "
        f"include_meal={bounds.include_meal}",
    )
    study.set_user_attr("space__static_covariates", str(bounds.static_candidates))


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
    objective = GBMObjective(
        data,
        config["output_root"],
        bounds=bounds,
        time_limit=config.get("time_limit"),
        keep_models=bool(config.get("keep_models", False)),
        gpu_label=gpu_label or os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        num_cpus=config.get("num_cpus"),
        primary_metric=metric_names[0],
        objective_metrics=metric_names if is_multi else None,
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
