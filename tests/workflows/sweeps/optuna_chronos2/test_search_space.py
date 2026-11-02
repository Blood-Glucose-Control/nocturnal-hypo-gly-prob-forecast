# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)

"""Unit tests for the Chronos-2 Optuna search space."""

import pytest

optuna = pytest.importorskip("optuna")

from src.workflows.sweeps.optuna_chronos2.data import (  # noqa: E402
    PIECEWISE_STATIC_CANDIDATES,
    STATIC_COVARIATE_CANDIDATES,
)
from src.workflows.sweeps.optuna_chronos2.search_space import (  # noqa: E402
    SearchSpaceBounds,
    suggest_search_point,
)


def _base_params() -> dict:
    """Non-covariate trial params shared by every case."""
    params = {
        "context_length": 216,
        "fine_tune_lr": 1.0e-4,
        "fine_tune_steps": 1000,
        "fine_tune_batch_size": 256,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "lora_target_modules": "attn_qv",
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "lr_scheduler_type": "linear",
        "gradient_accumulation_steps": 1,
        "cross_learning": False,
        "target_scaler": "standard",
        "eval_metric": "RMSE",
    }
    for col in (*STATIC_COVARIATE_CANDIDATES, *PIECEWISE_STATIC_CANDIDATES):
        params[f"use_static__{col}"] = False
    return params


def test_suggest_search_point_keeps_insulin_as_past_by_default():
    bounds = SearchSpaceBounds(
        past_candidates=["total_insulin", "carbs"],
        insulin_as_known=False,
    )
    params = _base_params()
    # insulin_as_known=False: every candidate (insulin included) is a use_past__.
    params["use_past__total_insulin"] = True
    params["use_past__carbs"] = True
    params["use_known_time"] = False
    point = suggest_search_point(optuna.trial.FixedTrial(params), bounds)

    assert point.past_cols == ["total_insulin", "carbs"]
    assert point.known_cols == []


def test_suggest_search_point_routes_insulin_to_known_covariates():
    bounds = SearchSpaceBounds(
        past_candidates=["total_insulin", "carbs"],
        insulin_as_known=True,
    )
    params = _base_params()
    # insulin_as_known=True: insulin uses its own use_known__ toggle; the
    # non-insulin candidate (carbs) remains a use_past__ toggle.
    params["use_past__carbs"] = True
    params["use_known_time"] = True
    params["use_known__total_insulin"] = True
    point = suggest_search_point(optuna.trial.FixedTrial(params), bounds)

    assert point.past_cols == ["carbs"]
    assert point.known_cols == ["hour_sin", "hour_cos", "total_insulin"]


def test_suggest_search_point_insulin_known_toggle_off_excludes_insulin():
    bounds = SearchSpaceBounds(
        past_candidates=["total_insulin", "carbs"],
        insulin_as_known=True,
    )
    params = _base_params()
    params["use_past__carbs"] = True
    params["use_known_time"] = False
    params["use_known__total_insulin"] = False
    point = suggest_search_point(optuna.trial.FixedTrial(params), bounds)

    assert point.past_cols == ["carbs"]
    assert point.known_cols == []


def test_suggest_search_point_optional_knobs_flow_into_config():
    bounds = SearchSpaceBounds(
        past_candidates=["total_insulin"],
        insulin_as_known=False,
        max_grad_norm_min=0.1,
        max_grad_norm_max=5.0,
        optim_choices=["adamw_torch_fused", "adafactor"],
        min_past_choices=[24, 48, 96],
        covariate_scaler_choices=[None, "local", "global"],
    )
    params = _base_params()
    params["use_past__total_insulin"] = True
    params["use_known_time"] = False
    params["max_grad_norm"] = 1.0
    params["optim"] = "adafactor"
    params["min_past"] = 48
    params["covariate_scaler"] = "global"
    point = suggest_search_point(optuna.trial.FixedTrial(params), bounds)

    trainer_kwargs = point.config_kwargs["extra_fine_tune_trainer_kwargs"]
    assert trainer_kwargs["max_grad_norm"] == 1.0
    assert trainer_kwargs["optim"] == "adafactor"
    assert point.config_kwargs["min_past"] == 48
    assert point.config_kwargs["covariate_scaler"] == "global"


def test_suggest_search_point_optional_knobs_absent_by_default():
    bounds = SearchSpaceBounds(
        past_candidates=["total_insulin"], insulin_as_known=False
    )
    params = _base_params()
    params["use_past__total_insulin"] = True
    params["use_known_time"] = False
    point = suggest_search_point(optuna.trial.FixedTrial(params), bounds)

    trainer_kwargs = point.config_kwargs["extra_fine_tune_trainer_kwargs"]
    assert "max_grad_norm" not in trainer_kwargs
    assert "optim" not in trainer_kwargs
    assert "min_past" not in point.config_kwargs
    assert "covariate_scaler" not in point.config_kwargs


def test_suggest_search_point_fixed_eval_metric_not_sampled():
    bounds = SearchSpaceBounds(
        past_candidates=["total_insulin"], insulin_as_known=False, eval_metric="WQL"
    )
    params = _base_params()
    # eval_metric is fixed, so it must NOT be sampled as a trial param.
    params.pop("eval_metric", None)
    params["use_past__total_insulin"] = True
    params["use_known_time"] = False
    point = suggest_search_point(optuna.trial.FixedTrial(params), bounds)

    assert point.eval_metric == "WQL"
    assert point.config_kwargs["eval_metric"] == "WQL"


def test_suggest_search_point_static_exclude_drops_holdout_missing_statics():
    # Excluded statics must never enter static_cols, even when their (would-be)
    # toggle is on -- they collapse to unknown/NaN at holdout inference.
    bounds = SearchSpaceBounds(
        past_candidates=["total_insulin"],
        insulin_as_known=False,
        static_exclude=[
            "source_file",
            "treatment_group",
            "ethnicity",
            "age_of_diagnosis",
        ],
    )
    params = _base_params()
    params["use_past__total_insulin"] = True
    params["use_known_time"] = False
    params["use_static__age"] = True
    params["use_static__gender"] = True
    # Excluded toggles are short-circuited (never sampled); set them True anyway
    # to prove they are still dropped.
    for col in ("source_file", "treatment_group", "ethnicity", "age_of_diagnosis"):
        params[f"use_static__{col}"] = True
    point = suggest_search_point(optuna.trial.FixedTrial(params), bounds)

    assert "age" in point.static_cols
    assert "gender" in point.static_cols
    for col in ("source_file", "treatment_group", "ethnicity", "age_of_diagnosis"):
        assert col not in point.static_cols


def test_suggest_search_point_include_sex_known_routes_sex_covariate():
    # include_sex_known routes the broadcast per-patient sex column into the
    # known covariate set (Chronos-2 ignores static_features, so sex must be a
    # known covariate to be used).
    base = SearchSpaceBounds(past_candidates=["total_insulin"], insulin_as_known=False)
    on = SearchSpaceBounds(
        past_candidates=["total_insulin"],
        insulin_as_known=False,
        include_sex_known=True,
    )
    params = _base_params()
    params["use_past__total_insulin"] = True
    params["use_known_time"] = False
    off_point = suggest_search_point(optuna.trial.FixedTrial(params), base)
    on_point = suggest_search_point(optuna.trial.FixedTrial(params), on)

    assert "sex" not in off_point.known_cols
    assert "sex" in on_point.known_cols
