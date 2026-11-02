# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)

"""Unit tests for the GBM sweep search space."""

import pytest

optuna = pytest.importorskip("optuna")

from src.workflows.sweeps.optuna_chronos2.features import (  # noqa: E402
    MEAL_EVENT_COL,
    MINS_TO_MEAL_COL,
)
from src.workflows.sweeps.optuna_gbm.search_space import (  # noqa: E402
    LAG_SETS,
    GBMSearchSpaceBounds,
    suggest_gbm_point,
)

# Fixed round cap (early stopping picks the effective count); not searched.
FIXED_ROUNDS = GBMSearchSpaceBounds().num_boost_round


def _lightgbm_params() -> dict:
    return {
        "strategy": "recursive",
        "backend": "lightgbm",
        "lag_set": "hourly",
        "target_scaler": "standard",
        "use_first_difference": False,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "feature_fraction": 0.9,
        "min_data_in_leaf": 20,
        "bagging_fraction": 0.8,
        "lambda_l1": 0.1,
        "lambda_l2": 0.5,
    }


def test_suggest_gbm_point_routes_insulin_and_maps_lightgbm():
    bounds = GBMSearchSpaceBounds(
        known_candidates=["total_insulin", "iob_exp"],
        static_candidates=["age", "gender"],
    )
    params = _lightgbm_params()
    params.update(
        {
            "use_known_time": True,
            "use_known__total_insulin": True,
            "use_known__iob_exp": False,
            "use_static__age": True,
            "use_static__gender": False,
        }
    )

    point = suggest_gbm_point(optuna.trial.FixedTrial(params), bounds)

    assert point.model_key == "RecursiveTabular"
    assert point.strategy == "recursive"
    assert point.backend == "lightgbm"
    assert point.hyperparameters["model_name"] == "GBM"
    assert point.known_cols == ["hour_sin", "hour_cos", "total_insulin"]
    assert point.static_cols == ["age"]
    assert not hasattr(point, "past_cols")
    assert point.hyperparameters["lags"] == LAG_SETS["hourly"]
    assert point.hyperparameters["target_scaler"] == "standard"
    assert "differences" not in point.hyperparameters
    mhp = point.hyperparameters["model_hyperparameters"]
    assert mhp["num_boost_round"] == FIXED_ROUNDS
    assert mhp["num_leaves"] == 31
    assert mhp["bagging_fraction"] == 0.8
    assert mhp["lambda_l1"] == 0.1
    assert mhp["lambda_l2"] == 0.5


def test_suggest_gbm_point_direct_catboost_with_first_difference():
    bounds = GBMSearchSpaceBounds(
        known_candidates=["total_insulin"],
        static_candidates=["age"],
        insulin_as_known=False,
    )
    params = {
        "strategy": "direct",
        "backend": "catboost",
        "lag_set": "recent",
        "target_scaler": None,
        "use_first_difference": True,
        "learning_rate": 0.1,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "random_strength": 1.0,
        "bagging_temperature": 0.5,
        "use_known_time": False,
        "use_static__age": True,
    }

    point = suggest_gbm_point(optuna.trial.FixedTrial(params), bounds)

    assert point.model_key == "DirectTabular"
    assert point.hyperparameters["model_name"] == "CAT"
    assert point.hyperparameters["differences"] == [1]
    assert "target_scaler" not in point.hyperparameters
    assert point.known_cols == []
    assert point.static_cols == ["age"]
    mhp = point.hyperparameters["model_hyperparameters"]
    assert mhp["iterations"] == FIXED_ROUNDS
    assert mhp["depth"] == 6
    assert mhp["random_strength"] == 1.0
    assert mhp["bagging_temperature"] == 0.5


def test_suggest_gbm_point_xgboost_routes_meal_when_enabled():
    bounds = GBMSearchSpaceBounds(
        known_candidates=["total_insulin"],
        static_candidates=["age"],
        include_meal=True,
    )
    params = {
        "strategy": "recursive",
        "backend": "xgboost",
        "lag_set": "multiscale",
        "target_scaler": "robust",
        "use_first_difference": False,
        "learning_rate": 0.03,
        "max_depth": 6,
        "min_child_weight": 2,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.01,
        "reg_lambda": 1.0,
        "gamma": 0.1,
        "use_known_time": False,
        "use_known__total_insulin": True,
        f"use_known__{MEAL_EVENT_COL}": True,
        f"use_known__{MINS_TO_MEAL_COL}": False,
        "use_static__age": False,
    }

    point = suggest_gbm_point(optuna.trial.FixedTrial(params), bounds)

    assert point.hyperparameters["model_name"] == "XGB"
    assert point.known_cols == ["total_insulin", MEAL_EVENT_COL]
    assert point.static_cols == []
    assert point.hyperparameters["lags"] == LAG_SETS["multiscale"]
    mhp = point.hyperparameters["model_hyperparameters"]
    assert mhp["n_estimators"] == FIXED_ROUNDS
    assert mhp["max_depth"] == 6
    assert mhp["reg_lambda"] == 1.0
    assert mhp["gamma"] == 0.1


def test_suggest_gbm_point_fixed_eval_metric_and_passthrough_limits():
    bounds = GBMSearchSpaceBounds(
        known_candidates=["total_insulin"],
        static_candidates=["age"],
        eval_metric="MAPE",
    )
    params = _lightgbm_params()
    params.update(
        {
            "use_known_time": False,
            "use_known__total_insulin": False,
            "use_static__age": False,
        }
    )
    assert "eval_metric" not in params

    point = suggest_gbm_point(optuna.trial.FixedTrial(params), bounds)

    assert point.eval_metric == "MAPE"
    assert point.known_cols == []
    assert point.hyperparameters["max_num_items"] == 20000
    assert point.hyperparameters["max_num_samples"] == 1000000
