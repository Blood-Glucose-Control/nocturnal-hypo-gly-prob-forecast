# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)

"""Unit tests for the TimesFM 3.0 sweep search space."""

import pytest

optuna = pytest.importorskip("optuna")

from src.workflows.sweeps.optuna_chronos2.features import (  # noqa: E402
    MEAL_EVENT_COL,
)
from src.workflows.sweeps.optuna_timesfm3.search_space import (  # noqa: E402
    INSULIN_KNOWN_CANDIDATES,
    TimesFM3SearchSpaceBounds,
    suggest_timesfm3_point,
)


def _base_params() -> dict:
    """Non-covariate trial params shared by every case."""
    return {
        "context_length": 288,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.0,
        "lora_target_modules": "all-linear",
        "use_dora": False,
        "use_rslora": False,
        "use_znorm": False,
        "lr": 1.0e-4,
        "weight_decay": 1.0e-2,
        "lr_schedule": "cosine",
        "warmup_ratio": 0.05,
        "max_epochs": 4,
        "batch_size": 32,
        # v3: meal-flag toggle is sampled for every trial (include_meal default);
        # default it off here so insulin-routing cases are unaffected.
        "use_known__meal_event": False,
    }


def test_suggest_timesfm3_point_routes_insulin_to_known_future_covariates():
    bounds = TimesFM3SearchSpaceBounds(
        known_candidates=["total_insulin", "iob_exp"],
    )
    params = _base_params()
    params["use_known_time"] = True
    params["use_known__total_insulin"] = True
    params["use_known__iob_exp"] = False

    point = suggest_timesfm3_point(optuna.trial.FixedTrial(params), bounds)

    # hour features (on/off together) come first, then per-channel insulin.
    assert point.known_cols == ["hour_sin", "hour_cos", "total_insulin"]
    # This sweep has no past-only covariate path.
    assert not hasattr(point, "past_cols")
    assert point.context_length == 288
    assert point.forecast_length == 24


def test_suggest_timesfm3_point_insulin_only_without_time():
    bounds = TimesFM3SearchSpaceBounds(
        known_candidates=["total_insulin", "iob_exp"],
    )
    params = _base_params()
    params["use_known_time"] = False
    params["use_known__total_insulin"] = False
    params["use_known__iob_exp"] = True

    point = suggest_timesfm3_point(optuna.trial.FixedTrial(params), bounds)

    assert point.known_cols == ["iob_exp"]


def test_suggest_timesfm3_point_can_select_no_known_covariates():
    bounds = TimesFM3SearchSpaceBounds(
        known_candidates=["total_insulin"],
    )
    params = _base_params()
    params["use_known_time"] = False
    params["use_known__total_insulin"] = False

    point = suggest_timesfm3_point(optuna.trial.FixedTrial(params), bounds)

    assert point.known_cols == []


def test_suggest_timesfm3_point_insulin_as_known_disabled_skips_channel_toggles():
    bounds = TimesFM3SearchSpaceBounds(
        known_candidates=["total_insulin"],
        insulin_as_known=False,
    )
    params = _base_params()
    params["use_known_time"] = True
    # No use_known__* params provided: they must NOT be sampled when disabled.

    point = suggest_timesfm3_point(optuna.trial.FixedTrial(params), bounds)

    assert point.known_cols == ["hour_sin", "hour_cos"]


def test_suggest_timesfm3_point_fixed_eval_metric_not_sampled():
    bounds = TimesFM3SearchSpaceBounds(known_candidates=["total_insulin"])
    params = _base_params()
    params["use_known_time"] = False
    params["use_known__total_insulin"] = True
    assert "eval_metric" not in params

    point = suggest_timesfm3_point(optuna.trial.FixedTrial(params), bounds)

    assert point.eval_metric == "mard_avg"
    assert point.known_cols == ["total_insulin"]


def test_suggest_timesfm3_point_samples_lora_and_optimizer_hyperparameters():
    bounds = TimesFM3SearchSpaceBounds(known_candidates=["total_insulin"])
    params = _base_params()
    params["lora_r"] = 16
    params["lora_alpha"] = 32
    params["lora_dropout"] = 0.1
    params["max_epochs"] = 6
    params["batch_size"] = 16
    params["use_known_time"] = False
    params["use_known__total_insulin"] = True

    point = suggest_timesfm3_point(optuna.trial.FixedTrial(params), bounds)

    hp = point.hyperparameters
    assert hp["lora_r"] == 16
    assert hp["lora_alpha"] == 32
    assert hp["lora_dropout"] == 0.1
    assert hp["max_epochs"] == 6
    assert hp["batch_size"] == 16
    assert hp["lr"] == pytest.approx(1.0e-4)
    assert hp["weight_decay"] == pytest.approx(1.0e-2)


def test_insulin_known_candidates_are_insulin_only():
    # Sanity: candidate list is derived from the insulin-masked covariates and
    # excludes carb-derived channels (unavailable at the holdout).
    assert "total_insulin" in INSULIN_KNOWN_CANDIDATES
    assert "iob_exp" in INSULIN_KNOWN_CANDIDATES
    assert all("cob" not in c and "carb" not in c for c in INSULIN_KNOWN_CANDIDATES)


def test_suggest_timesfm3_point_routes_meal_event_to_known_future():
    bounds = TimesFM3SearchSpaceBounds(known_candidates=["total_insulin"])
    params = _base_params()
    params["use_known_time"] = False
    params["use_known__total_insulin"] = True
    params["use_known__meal_event"] = True

    point = suggest_timesfm3_point(optuna.trial.FixedTrial(params), bounds)

    assert MEAL_EVENT_COL in point.known_cols
    assert point.known_cols == ["total_insulin", MEAL_EVENT_COL]


def test_suggest_timesfm3_point_meal_event_off_excludes_channel():
    bounds = TimesFM3SearchSpaceBounds(known_candidates=["total_insulin"])
    params = _base_params()
    params["use_known_time"] = False
    params["use_known__total_insulin"] = True
    params["use_known__meal_event"] = False

    point = suggest_timesfm3_point(optuna.trial.FixedTrial(params), bounds)

    assert MEAL_EVENT_COL not in point.known_cols
    assert point.known_cols == ["total_insulin"]


def test_suggest_timesfm3_point_include_meal_disabled_skips_meal_toggle():
    bounds = TimesFM3SearchSpaceBounds(
        known_candidates=["total_insulin"], include_meal=False
    )
    params = _base_params()
    params["use_known_time"] = False
    params["use_known__total_insulin"] = True
    # No use_known__meal_event provided: it must NOT be sampled when disabled.
    del params["use_known__meal_event"]

    point = suggest_timesfm3_point(optuna.trial.FixedTrial(params), bounds)

    assert MEAL_EVENT_COL not in point.known_cols
    assert point.known_cols == ["total_insulin"]
