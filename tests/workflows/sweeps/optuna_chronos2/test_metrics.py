# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)

"""Unit tests for the Chronos-2 Optuna challenge metrics.

These metrics directly determine the search objective and per-trial zone
reporting, so a boundary or off-by-one error would silently bias every trial.
"""

import numpy as np
import pytest

from src.workflows.sweeps.optuna_chronos2.metrics import (
    HORIZON_STEPS,
    dts_error_grid,
    evaluate_horizons,
    mard,
    rmse,
)


def test_rmse_basic():
    assert rmse([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0
    assert rmse([0.0], [3.0]) == pytest.approx(3.0)
    assert rmse([1.0, 3.0], [2.0, 1.0]) == pytest.approx(np.sqrt((1 + 4) / 2))
    assert np.isnan(rmse([], []))


def test_mard_excludes_zero_reference():
    # 10% relative error at every point.
    assert mard([110.0, 220.0], [100.0, 200.0]) == pytest.approx(10.0)
    # Zero references are dropped; remaining point is exact -> 0%.
    assert mard([50.0, 100.0], [0.0, 100.0]) == pytest.approx(0.0)
    assert np.isnan(mard([1.0], [0.0]))


def test_dts_perfect_prediction_all_zone_a():
    ref = np.array([40.0, 80.0, 120.0, 200.0, 300.0])
    zones = dts_error_grid(ref, ref)
    assert zones["dts_a_pct"] == pytest.approx(100.0)
    for other in ("b", "c", "d", "e"):
        assert zones[f"dts_{other}_pct"] == pytest.approx(0.0)


def test_dts_gross_error_is_zone_e():
    # ref=100 -> e_up ≈ 358.9; pred=400 exceeds it -> Zone E.
    zones = dts_error_grid([400.0], [100.0])
    assert zones["dts_e_pct"] == pytest.approx(100.0)


def test_dts_mild_overprediction_is_zone_b():
    # ref=100 -> b_up=120, c_up≈172.9; pred=170 lands in Zone B.
    zones = dts_error_grid([170.0], [100.0])
    assert zones["dts_b_pct"] == pytest.approx(100.0)


def test_evaluate_horizons_column_mapping():
    n = 8
    true = np.full((n, 24), 100.0)
    pred = true.copy()
    # Inject a distinct constant error at each clinical horizon's column.
    errors = {30: 10.0, 60: 20.0, 90: 30.0, 120: 40.0}
    for minutes, err in errors.items():
        col = HORIZON_STEPS[minutes] - 1
        pred[:, col] = true[:, col] + err

    result = evaluate_horizons(pred, true)
    for minutes, err in errors.items():
        assert result[f"rmse_{minutes}"] == pytest.approx(err)
        assert result[f"n_{minutes}"] == pytest.approx(n)


def test_evaluate_horizons_excludes_nan_targets():
    n = 5
    true = np.full((n, 24), 100.0)
    pred = np.full((n, 24), 100.0)
    col120 = HORIZON_STEPS[120] - 1
    true[0, col120] = np.nan  # one unusable ground-truth at +120
    result = evaluate_horizons(pred, true)
    assert result["n_120"] == pytest.approx(n - 1)
    assert result["rmse_120"] == pytest.approx(0.0)


def test_evaluate_horizons_rejects_short_matrix():
    with pytest.raises(ValueError):
        evaluate_horizons(np.zeros((3, 12)), np.zeros((3, 12)))


def test_evaluate_horizons_averages_match_manual_mean():
    rng = np.random.default_rng(0)
    true = rng.uniform(70.0, 180.0, size=(40, 24))
    pred = true + rng.normal(0.0, 12.0, size=(40, 24))
    result = evaluate_horizons(pred, true)
    for base in ("rmse", "mard", "dts_a_pct"):
        manual = np.mean([result[f"{base}_{m}"] for m in (30, 60, 90, 120)])
        assert result[f"{base}_avg"] == pytest.approx(manual)
