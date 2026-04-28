"""
Tests for TFTForecaster (Temporal Fusion Transformer).

Verifies registry registration, config defaults, and hyperparameter output.

Run:
    .venvs/chronos2/bin/python -m pytest tests/models/test_tft.py -v
"""

import pytest

pytest.importorskip("autogluon.timeseries")

from src.models.tft import TFTConfig, TFTForecaster  # noqa: E402
from src.models.base.registry import ModelRegistry  # noqa: E402


class TestTFTConfig:
    def test_model_type_field(self):
        assert TFTConfig().model_type == "tft"

    def test_hyperparameters_contain_tft_key(self):
        cfg = TFTConfig()
        hp = cfg.get_autogluon_hyperparameters()
        assert "TemporalFusionTransformer" in hp

    def test_hidden_size_and_heads_propagated(self):
        cfg = TFTConfig(hidden_size=128, num_heads=8)
        hp = cfg.get_autogluon_hyperparameters()
        tft = hp["TemporalFusionTransformer"]
        assert tft["hidden_size"] == 128
        assert tft["num_heads"] == 8

    def test_context_length_propagated(self):
        cfg = TFTConfig(context_length=256)
        hp = cfg.get_autogluon_hyperparameters()
        assert hp["TemporalFusionTransformer"]["context_length"] == 256

    def test_gradient_clip_val_in_trainer_kwargs(self):
        cfg = TFTConfig(gradient_clip_val=2.0)
        hp = cfg.get_autogluon_hyperparameters()
        assert (
            hp["TemporalFusionTransformer"]["trainer_kwargs"]["gradient_clip_val"]
            == 2.0
        )

    def test_default_min_segment_length(self):
        cfg = TFTConfig(context_length=512, forecast_length=96)
        assert cfg.min_segment_length == 512 + 96

    def test_explicit_min_segment_length(self):
        cfg = TFTConfig(min_segment_length=500)
        assert cfg.min_segment_length == 500


class TestTFTForecaster:
    def test_registry_registration(self):
        cls = ModelRegistry.get("tft")
        assert cls is TFTForecaster

    def test_config_class(self):
        assert TFTForecaster.config_class is TFTConfig

    def test_predictor_json_name(self):
        assert TFTForecaster._PREDICTOR_JSON_NAME == "tft_predictor.json"

    def test_supports_zero_shot_false(self):
        model = TFTForecaster(TFTConfig())
        assert model.supports_zero_shot is False
