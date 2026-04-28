"""
Tests for StatisticalForecaster (AutoARIMA, Theta, NPTS).

Verifies registry registration, config validation, and hyperparameter
output for each supported model.

Run:
    .venvs/chronos2/bin/python -m pytest tests/models/test_statistical.py -v
"""

import pytest

pytest.importorskip("autogluon.timeseries")

from src.models.statistical import StatisticalConfig, StatisticalForecaster  # noqa: E402
from src.models.base.registry import ModelRegistry  # noqa: E402


class TestStatisticalConfig:
    def test_default_model_name_is_autoarima(self):
        cfg = StatisticalConfig()
        assert cfg.model_name == "AutoARIMA"

    def test_theta_accepted(self):
        cfg = StatisticalConfig(model_name="Theta")
        assert cfg.model_name == "Theta"

    def test_npts_accepted(self):
        cfg = StatisticalConfig(model_name="NPTS")
        assert cfg.model_name == "NPTS"

    def test_invalid_model_name_raises(self):
        with pytest.raises(ValueError, match="model_name"):
            StatisticalConfig(model_name="DeepAR")

    def test_model_type_field(self):
        assert StatisticalConfig().model_type == "statistical"

    def test_default_time_limit(self):
        """Default time_limit should be 2h (7200s) to cap CPU fitting time."""
        cfg = StatisticalConfig()
        assert cfg.time_limit == 7200

    # -----------------------------------------------------------------------
    # AutoARIMA hyperparameters
    # -----------------------------------------------------------------------
    def test_autoarima_hyperparameters_seasonal_disabled(self):
        """CGM has no seasonal period — AutoARIMA must not attempt seasonal fit."""
        cfg = StatisticalConfig(model_name="AutoARIMA")
        hp = cfg.get_autogluon_hyperparameters()
        assert "AutoARIMA" in hp
        assert hp["AutoARIMA"]["seasonal"] is False
        assert hp["AutoARIMA"]["seasonal_period"] == 1

    def test_autoarima_max_p_q_propagated(self):
        cfg = StatisticalConfig(
            model_name="AutoARIMA", autoarima_max_p=5, autoarima_max_q=2
        )
        hp = cfg.get_autogluon_hyperparameters()
        assert hp["AutoARIMA"]["max_p"] == 5
        assert hp["AutoARIMA"]["max_q"] == 2

    # -----------------------------------------------------------------------
    # Theta hyperparameters
    # -----------------------------------------------------------------------
    def test_theta_season_length_one(self):
        """Theta season_length=1 disables seasonal adjustment."""
        cfg = StatisticalConfig(model_name="Theta")
        hp = cfg.get_autogluon_hyperparameters()
        assert "Theta" in hp
        assert hp["Theta"]["season_length"] == 1

    def test_theta_decomposition_type_propagated(self):
        cfg = StatisticalConfig(
            model_name="Theta", theta_decomposition_type="multiplicative"
        )
        hp = cfg.get_autogluon_hyperparameters()
        assert hp["Theta"]["decomposition_type"] == "multiplicative"

    # -----------------------------------------------------------------------
    # NPTS hyperparameters
    # -----------------------------------------------------------------------
    def test_npts_hyperparameters(self):
        cfg = StatisticalConfig(model_name="NPTS")
        hp = cfg.get_autogluon_hyperparameters()
        assert "NPTS" in hp
        assert isinstance(hp["NPTS"], dict)


class TestStatisticalForecaster:
    def test_registry_registration(self):
        cls = ModelRegistry.get("statistical")
        assert cls is StatisticalForecaster

    def test_config_class(self):
        assert StatisticalForecaster.config_class is StatisticalConfig

    def test_predictor_json_name(self):
        assert (
            StatisticalForecaster._PREDICTOR_JSON_NAME == "statistical_predictor.json"
        )

    def test_supports_zero_shot_false(self):
        model = StatisticalForecaster(StatisticalConfig())
        assert model.supports_zero_shot is False
