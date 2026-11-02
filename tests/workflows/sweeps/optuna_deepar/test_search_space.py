# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)

"""Unit tests for the DeepAR sweep search space."""

import pytest

optuna = pytest.importorskip("optuna")

from src.workflows.sweeps.optuna_deepar.search_space import (  # noqa: E402
    DeepARSearchSpaceBounds,
    suggest_deepar_point,
)


def _base_params() -> dict:
    """Non-covariate trial params shared by every case."""
    return {
        "context_length": 216,
        "num_layers": 2,
        "hidden_size": 40,
        "dropout_rate": 0.2,
        "lr": 1.0e-3,
        "weight_decay": 1.0e-4,
        "max_epochs": 50,
        "batch_size": 128,
        "num_batches_per_epoch": 50,
        "num_parallel_samples": 100,
        "target_scaler": None,
        "distr_output": "studentt",
    }


def test_suggest_deepar_point_routes_insulin_to_known_covariates():
    bounds = DeepARSearchSpaceBounds(
        known_candidates=["total_insulin", "iob_exp"],
        static_candidates=["age", "gender"],
    )
    params = _base_params()
    params["use_known_time"] = True
    params["use_known__total_insulin"] = True
    params["use_known__iob_exp"] = False
    params["use_static__age"] = True
    params["use_static__gender"] = False

    point = suggest_deepar_point(optuna.trial.FixedTrial(params), bounds)

    assert point.known_cols == ["hour_sin", "hour_cos", "total_insulin"]
    assert point.static_cols == ["age"]
    assert not hasattr(point, "past_cols")
    assert point.hyperparameters
    assert point.hyperparameters["context_length"] == 216


def test_suggest_deepar_point_can_select_only_static_covariates():
    bounds = DeepARSearchSpaceBounds(
        known_candidates=["total_insulin"],
        static_candidates=["age", "height"],
        insulin_as_known=False,
    )
    params = _base_params()
    params["use_known_time"] = False
    params["use_static__age"] = False
    params["use_static__height"] = True

    point = suggest_deepar_point(optuna.trial.FixedTrial(params), bounds)

    assert point.known_cols == []
    assert point.static_cols == ["height"]
    assert "target_scaler" not in point.hyperparameters


def test_suggest_deepar_point_fixed_eval_metric_not_sampled():
    bounds = DeepARSearchSpaceBounds(
        known_candidates=["total_insulin"],
        static_candidates=["age"],
        eval_metric="MAPE",
    )
    params = _base_params()
    params["use_known_time"] = False
    params["use_known__total_insulin"] = True
    params["use_static__age"] = False
    assert "eval_metric" not in params

    point = suggest_deepar_point(optuna.trial.FixedTrial(params), bounds)

    assert point.eval_metric == "MAPE"
    assert point.known_cols == ["total_insulin"]
