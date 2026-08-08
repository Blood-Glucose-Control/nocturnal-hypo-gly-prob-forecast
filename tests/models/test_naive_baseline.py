"""
Tests for NaiveBaselineForecaster (Naive and Average models).

Verifies registry registration, config validation, and hyperparameter
output. Full integration is covered by test_autogluon_base.py.

Run:
    .venvs/chronos2/bin/python -m pytest tests/models/test_naive_baseline.py -v
"""

import pytest

pytest.importorskip("autogluon.timeseries")

from src.models.naive_baseline import NaiveBaselineConfig, NaiveBaselineForecaster  # noqa: E402
from src.models.base.registry import ModelRegistry  # noqa: E402


class TestNaiveBaselineConfig:
    def test_default_model_name_is_naive(self):
        cfg = NaiveBaselineConfig()
        assert cfg.model_name == "Naive"

    def test_model_name_average_accepted(self):
        cfg = NaiveBaselineConfig(model_name="Average")
        assert cfg.model_name == "Average"

    def test_invalid_model_name_raises(self):
        with pytest.raises(ValueError, match="model_name"):
            NaiveBaselineConfig(model_name="ARIMA")

    def test_hyperparameters_naive(self):
        cfg = NaiveBaselineConfig(model_name="Naive")
        hp = cfg.get_autogluon_hyperparameters()
        assert "Naive" in hp
        assert hp["Naive"] == {}

    def test_hyperparameters_average(self):
        cfg = NaiveBaselineConfig(model_name="Average")
        hp = cfg.get_autogluon_hyperparameters()
        assert "Average" in hp

    def test_model_type_field(self):
        cfg = NaiveBaselineConfig()
        assert cfg.model_type == "naive_baseline"


class TestNaiveBaselineForecaster:
    def test_registry_registration(self):
        """NaiveBaselineForecaster must be registered under 'naive_baseline'."""
        cls = ModelRegistry.get("naive_baseline")
        assert cls is NaiveBaselineForecaster

    def test_config_class(self):
        assert NaiveBaselineForecaster.config_class is NaiveBaselineConfig

    def test_predictor_json_name(self):
        assert (
            NaiveBaselineForecaster._PREDICTOR_JSON_NAME
            == "naive_baseline_predictor.json"
        )

    def test_supports_zero_shot_false(self):
        model = NaiveBaselineForecaster(NaiveBaselineConfig())
        assert model.supports_zero_shot is False
