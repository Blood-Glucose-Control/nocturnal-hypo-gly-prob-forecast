# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)
# For commercial licensing, contact: christopher/cjrisi AT gluroo/uwaterloo DOT com/ca

import json

import numpy as np
import pytest

from src.visualization.nocturnal import (
    compute_horizon_rmse_quantiles,
    compute_horizon_rmse_stats,
    interpolate_quantile_trace,
    load_prediction_actual_arrays,
    load_probabilistic_forecast_arrays,
)


def test_load_prediction_actual_arrays_from_npz(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    predictions = np.array([[2.0, 3.0], [2.0, 3.0]])
    actuals = np.array([[1.0, 1.0], [1.0, 1.0]])
    np.savez(run_dir / "forecasts.npz", predictions=predictions, actuals=actuals)

    loaded_predictions, loaded_actuals = load_prediction_actual_arrays(run_dir)
    np.testing.assert_allclose(loaded_predictions, predictions)
    np.testing.assert_allclose(loaded_actuals, actuals)


def test_load_prediction_actual_arrays_from_legacy_json(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    payload = {
        "per_episode": [
            {"pred": [2.0, 3.0], "target_bg": [1.0, 1.0]},
            {"pred": [2.0, 3.0], "target_bg": [1.0, 1.0]},
            {"pred": [1.0], "target_bg": [1.0, 1.0]},  # filtered (length mismatch)
        ]
    }
    (run_dir / "nocturnal_results.json").write_text(json.dumps(payload))

    loaded_predictions, loaded_actuals = load_prediction_actual_arrays(run_dir)
    assert loaded_predictions.shape == (2, 2)
    assert loaded_actuals.shape == (2, 2)


def test_load_prediction_actual_arrays_missing_outputs(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_prediction_actual_arrays(tmp_path)


def test_horizon_rmse_helpers_return_expected_values():
    predictions = np.array([[2.0, 3.0], [2.0, 3.0]])
    actuals = np.array([[1.0, 1.0], [1.0, 1.0]])

    quantiles = compute_horizon_rmse_quantiles(predictions, actuals)
    assert len(quantiles) == 2
    assert quantiles[0]["horizon_minutes"] == 5.0
    assert quantiles[1]["horizon_minutes"] == 10.0
    assert quantiles[0]["rmse"] == pytest.approx(1.0)
    assert quantiles[1]["rmse"] == pytest.approx(2.0)
    assert quantiles[0]["median"] == pytest.approx(1.0)
    assert quantiles[1]["median"] == pytest.approx(2.0)
    assert quantiles[0]["whisker_low"] == pytest.approx(1.0)
    assert quantiles[1]["whisker_high"] == pytest.approx(2.0)
    assert quantiles[0]["q50"] == pytest.approx(1.0)
    assert quantiles[1]["q50"] == pytest.approx(2.0)

    stats = compute_horizon_rmse_stats(predictions, actuals)
    np.testing.assert_allclose(stats["hours"], np.array([1 / 12, 1 / 6]))
    np.testing.assert_allclose(stats["mean"], np.array([1.0, 2.0]))
    np.testing.assert_allclose(stats["band_low"], np.array([1.0, 2.0]))
    np.testing.assert_allclose(stats["band_high"], np.array([1.0, 2.0]))
    np.testing.assert_allclose(stats["q25"], np.array([1.0, 2.0]))
    np.testing.assert_allclose(stats["q75"], np.array([1.0, 2.0]))
    np.testing.assert_allclose(stats["cumulative"], np.array([1.0, np.sqrt(2.5)]))


def test_horizon_rmse_helpers_allow_custom_quantiles():
    predictions = np.array([[1.0, 3.0], [3.0, 5.0]])
    actuals = np.array([[1.0, 1.0], [1.0, 1.0]])

    custom_boxplot = (5.0, 20.0, 50.0, 80.0, 95.0)
    quantiles = compute_horizon_rmse_quantiles(
        predictions,
        actuals,
        quantiles=custom_boxplot,
    )
    assert "q5" in quantiles[0]
    assert "q95" in quantiles[0]
    assert quantiles[0]["whisker_low"] == pytest.approx(quantiles[0]["q5"])
    assert quantiles[0]["box_low"] == pytest.approx(quantiles[0]["q20"])
    assert quantiles[0]["box_high"] == pytest.approx(quantiles[0]["q80"])
    assert quantiles[0]["whisker_high"] == pytest.approx(quantiles[0]["q95"])

    custom_iqr = (10.0, 90.0)
    stats = compute_horizon_rmse_stats(predictions, actuals, iqr_quantiles=custom_iqr)
    np.testing.assert_allclose(stats["band_low"], stats["q10"])
    np.testing.assert_allclose(stats["band_high"], stats["q90"])


def test_load_probabilistic_forecast_arrays(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    actuals = np.array([[1.0, 2.0], [3.0, 4.0]])
    quantile_forecasts = np.array(
        [
            [[0.8, 1.8], [1.0, 2.0], [1.2, 2.2]],
            [[2.8, 3.8], [3.0, 4.0], [3.2, 4.2]],
        ]
    )
    quantile_levels = np.array([0.1, 0.5, 0.9])
    episode_ids = np.array(["ep_a", "ep_b"])
    np.savez(
        run_dir / "forecasts.npz",
        actuals=actuals,
        quantile_forecasts=quantile_forecasts,
        quantile_levels=quantile_levels,
        episode_ids=episode_ids,
    )

    loaded_actuals, loaded_qf, loaded_ql, loaded_ids = (
        load_probabilistic_forecast_arrays(run_dir)
    )
    np.testing.assert_allclose(loaded_actuals, actuals)
    np.testing.assert_allclose(loaded_qf, quantile_forecasts)
    np.testing.assert_allclose(loaded_ql, quantile_levels)
    assert list(loaded_ids) == ["ep_a", "ep_b"]


def test_interpolate_quantile_trace():
    quantile_levels = np.array([0.1, 0.5, 0.9])
    quantile_forecast = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]
    )
    q25 = interpolate_quantile_trace(quantile_forecast, quantile_levels, 0.25)
    np.testing.assert_allclose(q25, np.array([1.75, 2.75]))
