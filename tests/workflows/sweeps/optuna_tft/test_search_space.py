# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)

"""Unit tests for the TFT sweep search space."""

import pytest

optuna = pytest.importorskip("optuna")

from src.workflows.sweeps.optuna_tft.search_space import (  # noqa: E402
    TFTSearchSpaceBounds,
    suggest_tft_point,
)


def _sample(seed: int):
    bounds = TFTSearchSpaceBounds()
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=seed))
    captured = {}

    def _obj(trial):
        captured["point"] = suggest_tft_point(trial, bounds)
        return 0.0

    study.optimize(_obj, n_trials=1)
    return captured["point"], bounds


def test_suggest_tft_point_hidden_dim_divisible_by_heads():
    # GluonTS TFT requires hidden_dim % num_heads == 0; every sampled combo must
    # satisfy it, otherwise the model would fail to build.
    for seed in range(25):
        point, _ = _sample(seed)
        hp = point.hyperparameters
        assert hp["hidden_dim"] % hp["num_heads"] == 0


def test_suggest_tft_point_within_bounds_and_shape():
    point, bounds = _sample(0)
    hp = point.hyperparameters
    assert bounds.context_min <= hp["context_length"] <= bounds.context_max
    assert bounds.lr_min <= hp["lr"] <= bounds.lr_max
    assert 0.0 <= hp["dropout_rate"] <= bounds.dropout_max
    assert hp["early_stopping_patience"] == bounds.early_stopping_patience
    assert point.forecast_length == 24
    # Covariate lists are subsets of the shared candidate pools.
    from src.workflows.sweeps.optuna_chronos2.features import PAST_COVARIATE_CANDIDATES

    assert set(point.past_cols).issubset(set(PAST_COVARIATE_CANDIDATES))
