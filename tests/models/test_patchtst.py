"""
Tests for PatchTSTForecaster.

Verifies registry registration, config defaults, and hyperparameter output.

Run:
    .venvs/chronos2/bin/python -m pytest tests/models/test_patchtst.py -v
"""

import pytest

pytest.importorskip("autogluon.timeseries")

from src.models.patchtst import PatchTSTConfig, PatchTSTForecaster  # noqa: E402
from src.models.base.registry import ModelRegistry  # noqa: E402


class TestPatchTSTConfig:
    def test_model_type_field(self):
        assert PatchTSTConfig().model_type == "patchtst"

    def test_hyperparameters_contain_patchtst_key(self):
        cfg = PatchTSTConfig()
        hp = cfg.get_autogluon_hyperparameters()
        assert "PatchTST" in hp

    def test_patch_len_and_stride_propagated(self):
        cfg = PatchTSTConfig(patch_len=32, stride=16)
        hp = cfg.get_autogluon_hyperparameters()
        assert hp["PatchTST"]["patch_len"] == 32
        assert hp["PatchTST"]["stride"] == 16

    def test_context_length_propagated(self):
        cfg = PatchTSTConfig(context_length=256)
        hp = cfg.get_autogluon_hyperparameters()
        assert hp["PatchTST"]["context_length"] == 256

    def test_d_model_and_nhead_propagated(self):
        cfg = PatchTSTConfig(d_model=64, nhead=8)
        hp = cfg.get_autogluon_hyperparameters()
        assert hp["PatchTST"]["d_model"] == 64
        assert hp["PatchTST"]["nhead"] == 8

    def test_default_min_segment_length(self):
        cfg = PatchTSTConfig(context_length=512, forecast_length=96)
        assert cfg.min_segment_length == 512 + 96

    def test_explicit_min_segment_length(self):
        cfg = PatchTSTConfig(min_segment_length=400)
        assert cfg.min_segment_length == 400


class TestPatchTSTForecaster:
    def test_registry_registration(self):
        cls = ModelRegistry.get("patchtst")
        assert cls is PatchTSTForecaster

    def test_config_class(self):
        assert PatchTSTForecaster.config_class is PatchTSTConfig

    def test_predictor_json_name(self):
        assert PatchTSTForecaster._PREDICTOR_JSON_NAME == "patchtst_predictor.json"

    def test_supports_zero_shot_false(self):
        model = PatchTSTForecaster(PatchTSTConfig())
        assert model.supports_zero_shot is False
