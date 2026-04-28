"""
Tests for AutoGluonBaseModel shared base class.

Verifies config validation, flat_df → TimeSeriesDataFrame conversion,
and save/load round-trip using NaiveBaselineForecaster as the concrete impl.

Run:
    .venvs/chronos2/bin/python -m pytest tests/models/test_autogluon_base.py -v
"""

import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("autogluon.timeseries")

from src.models.naive_baseline import NaiveBaselineConfig, NaiveBaselineForecaster  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_flat_df(n_patients=2, n_days=3, include_iob=False):
    """Flat DataFrame with DatetimeIndex-aware rows at 5-min cadence."""
    frames = []
    rng = np.random.default_rng(0)
    for i in range(1, n_patients + 1):
        n = n_days * 288
        df = pd.DataFrame(
            {
                "datetime": pd.date_range(f"2024-{i:02d}-01", periods=n, freq="5min"),
                "bg_mM": rng.normal(8.0, 1.5, n).clip(2.2, 22.0),
                "p_num": float(i),
            }
        )
        if include_iob:
            df["iob"] = rng.exponential(0.5, n).clip(0, 5)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAutoGluonBaseModel:
    """Tests for AutoGluonBaseModel via NaiveBaselineForecaster."""

    def test_config_post_init_sets_min_segment_length(self):
        """min_segment_length defaults to forecast_length when None."""
        cfg = NaiveBaselineConfig(context_length=512, forecast_length=96)
        # Default min_segment_length = forecast_length
        assert cfg.min_segment_length == 96

    def test_config_min_segment_length_explicit(self):
        """Explicit min_segment_length is respected."""
        cfg = NaiveBaselineConfig(
            context_length=512, forecast_length=96, min_segment_length=200
        )
        assert cfg.min_segment_length == 200

    def test_model_instantiation(self):
        """Model can be instantiated and is initially unfitted."""
        cfg = NaiveBaselineConfig(context_length=512, forecast_length=96)
        model = NaiveBaselineForecaster(cfg)
        assert not model.is_fitted

    def test_supports_zero_shot_false(self):
        """Naive baseline does not support zero-shot prediction."""
        cfg = NaiveBaselineConfig()
        model = NaiveBaselineForecaster(cfg)
        assert model.supports_zero_shot is False

    def test_prepare_training_data_returns_tsdf(self):
        """_prepare_training_data converts flat_df to TimeSeriesDataFrame."""
        from autogluon.timeseries import TimeSeriesDataFrame

        cfg = NaiveBaselineConfig(context_length=512, forecast_length=96)
        model = NaiveBaselineForecaster(cfg)
        flat_df = _make_flat_df(n_patients=2, n_days=3)
        tsdf = model._prepare_training_data(flat_df)
        assert isinstance(tsdf, TimeSeriesDataFrame)
        # Should have item_ids from patient IDs
        assert len(tsdf.item_ids) > 0

    def test_prepare_training_data_with_covariate(self):
        """_prepare_training_data preserves covariate columns."""
        from autogluon.timeseries import TimeSeriesDataFrame

        cfg = NaiveBaselineConfig(
            context_length=512, forecast_length=96, covariate_cols=["iob"]
        )
        model = NaiveBaselineForecaster(cfg)
        flat_df = _make_flat_df(n_patients=2, n_days=3, include_iob=True)
        tsdf = model._prepare_training_data(flat_df)
        assert isinstance(tsdf, TimeSeriesDataFrame)
        assert "iob" in tsdf.columns

    def test_save_load_roundtrip(self):
        """_save_checkpoint and _load_checkpoint round-trip without errors."""
        from autogluon.timeseries import TimeSeriesPredictor

        cfg = NaiveBaselineConfig(
            context_length=48, forecast_length=12, eval_metric="WQL"
        )
        model = NaiveBaselineForecaster(cfg)

        flat_df = _make_flat_df(n_patients=2, n_days=3)
        tsdf = model._prepare_training_data(flat_df)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Manually fit a predictor so we have something to save
            predictor = TimeSeriesPredictor(
                prediction_length=cfg.forecast_length,
                path=os.path.join(tmpdir, "ag_predictor"),
                eval_metric=cfg.eval_metric,
                verbosity=0,
            )
            predictor.fit(tsdf, hyperparameters={"Naive": {}})
            model.predictor = predictor
            model.is_fitted = True

            ckpt_dir = os.path.join(tmpdir, "checkpoint")
            os.makedirs(ckpt_dir)
            model._save_checkpoint(ckpt_dir)

            # Verify JSON reference was written
            json_path = os.path.join(ckpt_dir, model._PREDICTOR_JSON_NAME)
            assert os.path.exists(json_path)
            with open(json_path) as f:
                data = json.load(f)
            assert "predictor_path" in data

            # Load into a fresh model
            model2 = NaiveBaselineForecaster(cfg)
            model2._load_checkpoint(ckpt_dir)
            assert model2.is_fitted
            assert model2.predictor is not None
