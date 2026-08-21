"""Tests for schema-validated model config loading."""

from pathlib import Path
import sys
import types

import pytest

from src.config.schemas import (
    build_model_runtime_config,
    get_model_config_schema,
    get_registered_model_config_types,
)
from src.workflows.forecasting.modeling import GenericModelConfig, ModelFactory
from src.workflows.forecasting.modeling import load_model_config_from_yaml


def _write(path: Path, content: str) -> str:
    path.write_text(content, encoding="utf-8")
    return str(path)


def _install_fake_tsmixer_module(monkeypatch: pytest.MonkeyPatch) -> type:
    fake_models_pkg = types.ModuleType("src.models")
    fake_models_pkg.__path__ = []  # type: ignore[attr-defined]

    class FakeTSMixerConfig:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
            if getattr(self, "min_segment_length", None) is None:
                self.min_segment_length = self.context_length + self.forecast_length

    class FakeTSMixerForecaster:
        def __init__(self, config):
            self.config = config

    fake_tsmixer_module = types.ModuleType("src.models.tsmixer")
    fake_tsmixer_module.TSMixerConfig = FakeTSMixerConfig
    fake_tsmixer_module.TSMixerForecaster = FakeTSMixerForecaster

    monkeypatch.setitem(sys.modules, "src.models", fake_models_pkg)
    monkeypatch.setitem(sys.modules, "src.models.tsmixer", fake_tsmixer_module)
    return FakeTSMixerConfig


def test_tsmixer_model_config_validates_and_loads(tmp_path: Path) -> None:
    config_path = _write(
        tmp_path / "tsmixer_valid.yaml",
        """
model_type: tsmixer
training_mode: from_scratch
context_length: 128
forecast_length: 96
hidden_size: 32
ff_size: 32
num_blocks: 1
dropout: 0.1
learning_rate: 0.001
covariate_cols: [iob, cob]
target_col: bg_mM
patient_col: p_num
time_col: datetime
interval_mins: 5
imputation_threshold_mins: 45
""".strip(),
    )

    loaded = load_model_config_from_yaml(config_path, model_type="tsmixer")
    assert loaded["model_type"] == "tsmixer"
    assert loaded["context_length"] == 128
    assert loaded["covariate_cols"] == ["iob", "cob"]


def test_tsmixer_model_config_normalizes_lr_alias(tmp_path: Path) -> None:
    config_path = _write(
        tmp_path / "tsmixer_lr_alias.yaml",
        """
model_type: tsmixer
context_length: 128
forecast_length: 96
lr: 0.004
""".strip(),
    )

    loaded = load_model_config_from_yaml(config_path, model_type="tsmixer")
    assert loaded["learning_rate"] == pytest.approx(0.004)
    assert "lr" not in loaded


def test_tsmixer_model_config_reports_schema_errors(tmp_path: Path) -> None:
    config_path = _write(
        tmp_path / "tsmixer_invalid.yaml",
        """
model_type: tsmixer
context_length: "128"
unknown_field: true
""".strip(),
    )

    with pytest.raises(ValueError) as exc_info:
        load_model_config_from_yaml(config_path, model_type="tsmixer")

    message = str(exc_info.value)
    assert config_path in message
    assert "context_length" in message
    assert "unknown_field" in message


def test_tsmixer_runtime_adapter_builds_runtime_config() -> None:
    runtime_config = build_model_runtime_config(
        model_type="tsmixer",
        config_data={
            "context_length": 128,
            "forecast_length": 96,
            "lr": 0.002,
            "covariate_cols": ["iob", "cob"],
            "use_cpu": True,
        },
    )

    assert runtime_config["learning_rate"] == pytest.approx(0.002)
    assert runtime_config["covariate_cols"] == ["iob", "cob"]
    assert runtime_config["use_cpu"] is True
    assert runtime_config["context_length"] == 128
    assert runtime_config["forecast_length"] == 96


def test_model_config_registry_exposes_tsmixer_schema_and_adapter() -> None:
    assert "tsmixer" in get_registered_model_config_types()
    assert get_model_config_schema("tsmixer") is not None


def test_tsmixer_factory_path_uses_schema_adapter_for_lr_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_config_class = _install_fake_tsmixer_module(monkeypatch)

    config = GenericModelConfig(
        model_type="tsmixer",
        model_path="",
        context_length=128,
        forecast_length=96,
        batch_size=16,
        num_epochs=2,
        learning_rate=1e-4,
        extra_config={"lr": 0.003, "covariate_cols": ["iob"]},
    )

    model = ModelFactory.create_model(config)
    assert isinstance(model.config, fake_config_class)
    assert model.config.learning_rate == pytest.approx(0.003)
    assert model.config.covariate_cols == ["iob"]


def test_tsmixer_factory_path_reports_unknown_runtime_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_tsmixer_module(monkeypatch)

    config = GenericModelConfig(
        model_type="tsmixer",
        model_path="",
        context_length=128,
        forecast_length=96,
        extra_config={"unknown_field": True},
    )

    with pytest.raises(ValueError) as exc_info:
        ModelFactory.create_model(config)

    assert "unknown_field" in str(exc_info.value)


def test_runtime_adapter_reports_registered_types_for_unknown_model() -> None:
    with pytest.raises(ValueError) as exc_info:
        build_model_runtime_config("unknown_model", {})

    message = str(exc_info.value)
    assert "unknown_model" in message
    assert "Registered adapter types" in message
    assert "tsmixer" in message
