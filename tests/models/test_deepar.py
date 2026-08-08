"""
Tests for DeepARForecaster.

Verifies registry registration, config defaults, and hyperparameter output.

Run:
    .venvs/chronos2/bin/python -m pytest tests/models/test_deepar.py -v
"""

import pytest

pytest.importorskip("autogluon.timeseries")

from src.models.deepar import DeepARConfig, DeepARForecaster  # noqa: E402
from src.models.base.registry import ModelRegistry  # noqa: E402


class TestDeepARConfig:
    def test_model_type_field(self):
        assert DeepARConfig().model_type == "deepar"

    def test_default_dropout_rate(self):
        cfg = DeepARConfig()
        assert cfg.dropout_rate == 0.1

    def test_hyperparameters_contain_deepar_key(self):
        cfg = DeepARConfig()
        hp = cfg.get_autogluon_hyperparameters()
        assert "DeepAR" in hp

    def test_hyperparameters_context_length(self):
        cfg = DeepARConfig(context_length=256)
        hp = cfg.get_autogluon_hyperparameters()
        assert hp["DeepAR"]["context_length"] == 256

    def test_hyperparameters_trainer_kwargs(self):
        cfg = DeepARConfig(gradient_clip_val=5.0)
        hp = cfg.get_autogluon_hyperparameters()
        assert hp["DeepAR"]["trainer_kwargs"]["gradient_clip_val"] == 5.0

    def test_default_min_segment_length_set_by_post_init(self):
        cfg = DeepARConfig(context_length=512, forecast_length=96)
        assert cfg.min_segment_length == 512 + 96

    def test_explicit_min_segment_length(self):
        cfg = DeepARConfig(min_segment_length=300)
        assert cfg.min_segment_length == 300


class TestDeepARForecaster:
    def test_registry_registration(self):
        cls = ModelRegistry.get("deepar")
        assert cls is DeepARForecaster

    def test_config_class(self):
        assert DeepARForecaster.config_class is DeepARConfig

    def test_predictor_json_name(self):
        assert DeepARForecaster._PREDICTOR_JSON_NAME == "deepar_predictor.json"

    def test_supports_zero_shot_false(self):
        model = DeepARForecaster(DeepARConfig())
        assert model.supports_zero_shot is False
