"""
Tests for GBMForecaster (gradient-boosted tree tabular family).

Verifies registry registration, config defaults, hyperparameter routing, and a
lightweight end-to-end fit/predict on synthetic data.

Run:
    .venvs/autogluon/bin/python -m pytest tests/models/test_gbm.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("autogluon.timeseries")

from src.models.base.registry import ModelRegistry  # noqa: E402
from src.models.gbm import (  # noqa: E402
    GBMConfig,
    GBMForecaster,
    create_default_gbm_config,
)


class TestGBMConfig:
    def test_model_type_field(self):
        assert GBMConfig().model_type == "gbm"

    def test_default_strategy_and_backend(self):
        cfg = GBMConfig()
        assert cfg.strategy == "recursive"
        assert cfg.backend == "lightgbm"

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError, match="strategy"):
            GBMConfig(strategy="bogus")

    def test_invalid_backend_raises(self):
        with pytest.raises(ValueError, match="backend"):
            GBMConfig(backend="bogus")

    def test_recursive_maps_to_recursive_tabular_key(self):
        hp = GBMConfig(strategy="recursive").get_autogluon_hyperparameters()
        assert set(hp) == {"RecursiveTabular"}

    def test_direct_maps_to_direct_tabular_key(self):
        hp = GBMConfig(strategy="direct").get_autogluon_hyperparameters()
        assert set(hp) == {"DirectTabular"}

    @pytest.mark.parametrize(
        "backend,expected",
        [("lightgbm", "GBM"), ("catboost", "CAT"), ("xgboost", "XGB")],
    )
    def test_backend_maps_to_tabular_model_name(self, backend, expected):
        hp = GBMConfig(backend=backend).get_autogluon_hyperparameters()
        (params,) = hp.values()
        assert params["model_name"] == expected

    def test_optional_knobs_omitted_when_none(self):
        params = next(iter(GBMConfig().get_autogluon_hyperparameters().values()))
        for key in ("lags", "differences", "target_scaler", "max_num_items"):
            assert key not in params

    def test_optional_knobs_forwarded_when_set(self):
        cfg = GBMConfig(
            lags=[1, 2, 3],
            differences=[1],
            target_scaler="standard",
            max_num_items=5000,
            max_num_samples=10000,
            model_hyperparameters={"num_boost_round": 50},
        )
        params = next(iter(cfg.get_autogluon_hyperparameters().values()))
        assert params["lags"] == [1, 2, 3]
        assert params["differences"] == [1]
        assert params["target_scaler"] == "standard"
        assert params["max_num_items"] == 5000
        assert params["max_num_samples"] == 10000
        assert params["model_hyperparameters"] == {"num_boost_round": 50}

    def test_default_min_segment_length_set_by_post_init(self):
        cfg = GBMConfig(context_length=288, forecast_length=24)
        assert cfg.min_segment_length == 288 + 24

    def test_explicit_min_segment_length(self):
        assert GBMConfig(min_segment_length=300).min_segment_length == 300

    def test_create_default_helper(self):
        cfg = create_default_gbm_config(backend="catboost")
        assert isinstance(cfg, GBMConfig)
        assert cfg.backend == "catboost"


class TestGBMForecaster:
    def test_registry_registration(self):
        assert ModelRegistry.get("gbm") is GBMForecaster

    def test_config_class(self):
        assert GBMForecaster.config_class is GBMConfig

    def test_predictor_json_name(self):
        assert GBMForecaster._PREDICTOR_JSON_NAME == "gbm_predictor.json"

    def test_supports_zero_shot_false(self):
        assert GBMForecaster(GBMConfig()).supports_zero_shot is False


def _synthetic_panel(n_patients: int = 6, n_steps: int = 420) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for pid in range(n_patients):
        start = pd.Timestamp("2020-01-01")
        series = 5.0 + rng.normal(0, 0.2) + np.cumsum(rng.normal(0, 0.05, n_steps))
        for i, value in enumerate(series):
            rows.append(
                {
                    "patient_id": f"p{pid}",
                    "datetime": start + pd.Timedelta(minutes=5 * i),
                    "bg_mM": float(value),
                }
            )
    return pd.DataFrame(rows)


class TestGBMIntegration:
    def test_fit_predict_save_load_roundtrip(self, tmp_path):
        df = _synthetic_panel()
        cfg = GBMConfig(
            context_length=96,
            forecast_length=24,
            strategy="recursive",
            backend="lightgbm",
            lags=[1, 2, 3, 6, 12],
            target_scaler="standard",
            eval_metric="MAPE",
            model_hyperparameters={"num_boost_round": 30},
        )
        model = GBMForecaster(cfg)
        model.fit(df, output_dir=str(tmp_path / "fit"))

        # Episode must exceed the recursive min length so GBM (not the
        # SeasonalNaive fallback) produces the forecast.
        episode = df[df.patient_id == "p0"].iloc[:360]
        preds = np.asarray(model.predict(episode), dtype=float)
        assert preds.shape == (24,)
        assert np.isfinite(preds).all()

        save_dir = tmp_path / "saved"
        model.save(str(save_dir))
        reloaded = GBMForecaster.load(str(save_dir))
        reloaded_preds = np.asarray(reloaded.predict(episode), dtype=float)
        assert reloaded_preds.shape == (24,)
        assert np.isfinite(reloaded_preds).all()
