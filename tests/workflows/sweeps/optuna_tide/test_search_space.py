# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)

"""Unit tests for the TiDE sweep search space."""

import pytest

optuna = pytest.importorskip("optuna")

from src.workflows.sweeps.optuna_tide.search_space import (  # noqa: E402
    TiDESearchSpaceBounds,
    suggest_tide_point,
)


def _base_params() -> dict:
    """Non-covariate trial params shared by every case."""
    return {
        "context_length": 216,
        "encoder_hidden_dim": 64,
        "temporal_hidden_dim": 64,
        "distr_hidden_dim": 64,
        "feat_proj_hidden_dim": 16,
        "num_layers_encoder": 2,
        "num_layers_decoder": 2,
        "decoder_output_dim": 16,
        "dropout_rate": 0.2,
        "layer_norm": True,
        "lr": 1.0e-3,
        "weight_decay": 1.0e-4,
        "max_epochs": 50,
        "batch_size": 256,
        "num_batches_per_epoch": 50,
        "target_scaler": None,
        "distr_output": "studentt",
        "num_feat_dynamic_proj": 4,
        "scaling": "mean",
    }


def test_suggest_tide_point_routes_insulin_to_known_covariates():
    bounds = TiDESearchSpaceBounds(
        known_candidates=["total_insulin", "iob_exp"],
        static_candidates=["age", "gender"],
    )
    params = _base_params()
    params["use_known_time"] = True
    params["use_known__total_insulin"] = True
    params["use_known__iob_exp"] = False
    params["use_static__age"] = True
    params["use_static__gender"] = False

    point = suggest_tide_point(optuna.trial.FixedTrial(params), bounds)

    assert point.known_cols == ["hour_sin", "hour_cos", "total_insulin"]
    assert point.static_cols == ["age"]
    assert not hasattr(point, "past_cols")
    assert point.hyperparameters
    assert point.hyperparameters["context_length"] == 216


def test_suggest_tide_point_can_select_only_static_covariates():
    bounds = TiDESearchSpaceBounds(
        known_candidates=["total_insulin"],
        static_candidates=["age", "height"],
        insulin_as_known=False,
    )
    params = _base_params()
    params["use_known_time"] = False
    params["use_static__age"] = False
    params["use_static__height"] = True

    point = suggest_tide_point(optuna.trial.FixedTrial(params), bounds)

    assert point.known_cols == []
    assert point.static_cols == ["height"]
    assert "target_scaler" not in point.hyperparameters


def test_suggest_tide_point_fixed_eval_metric_not_sampled():
    bounds = TiDESearchSpaceBounds(
        known_candidates=["total_insulin"],
        static_candidates=["age"],
        eval_metric="MAPE",
    )
    params = _base_params()
    params["use_known_time"] = False
    params["use_known__total_insulin"] = True
    params["use_static__age"] = False
    assert "eval_metric" not in params

    point = suggest_tide_point(optuna.trial.FixedTrial(params), bounds)

    assert point.eval_metric == "MAPE"
    assert point.known_cols == ["total_insulin"]
