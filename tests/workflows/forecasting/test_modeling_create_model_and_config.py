"""Unit coverage for schema-routed create_model_and_config helper."""
# pyright: reportMissingImports=false

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from src.workflows.forecasting.modeling import ModelFactory, create_model_and_config


def test_create_model_and_config_uses_zero_shot_path_for_zero_shot_family(
    monkeypatch,
) -> None:
    recorded: dict[str, Any] = {}
    sentinel_config = SimpleNamespace(name="zero-shot-config")
    sentinel_model = SimpleNamespace(config=SimpleNamespace(name="model-config"))

    def fake_create_zero_shot_config(**kwargs):
        recorded["path"] = "zero_shot"
        recorded["kwargs"] = kwargs
        return sentinel_config

    def fake_create_model(config):
        recorded["create_model_config"] = config
        return sentinel_model

    monkeypatch.setattr(
        ModelFactory,
        "create_zero_shot_config",
        staticmethod(fake_create_zero_shot_config),
    )
    monkeypatch.setattr(ModelFactory, "create_model", staticmethod(fake_create_model))

    model, config = create_model_and_config(
        "ttm",
        context_length=128,
        forecast_length=64,
        batch_size=16,
        covariate_cols=["iob"],
    )

    assert recorded["path"] == "zero_shot"
    assert recorded["create_model_config"] is sentinel_config
    assert recorded["kwargs"]["model_type"] == "ttm"
    assert model is sentinel_model
    assert config is sentinel_model.config


def test_create_model_and_config_checkpoint_applies_safe_overrides(monkeypatch) -> None:
    sentinel_generic_config = SimpleNamespace(name="finetune-config")
    loaded_config = SimpleNamespace(
        forecast_length=96,
        context_length=512,
        batch_size=8,
        covariate_cols=[],
    )
    loaded_model = SimpleNamespace(config=loaded_config)

    monkeypatch.setattr(
        ModelFactory,
        "create_finetune_config",
        staticmethod(lambda **kwargs: sentinel_generic_config),
    )
    monkeypatch.setattr(
        ModelFactory, "load_model", staticmethod(lambda *_args, **_kwargs: loaded_model)
    )

    model, config = create_model_and_config(
        "chronos2",
        checkpoint="/tmp/fake-checkpoint",
        forecast_length=72,
        context_length=700,
        batch_size=64,
        covariate_cols=["iob"],
    )

    assert model is loaded_model
    assert config is loaded_config
    assert config.forecast_length == 72
    assert config.context_length == 512  # mismatch ignored for checkpoint safety
    assert config.batch_size == 64
    assert config.covariate_cols == ["iob"]


def test_create_model_and_config_checkpoint_blocks_forecast_length_increase(
    monkeypatch,
) -> None:
    loaded_config = SimpleNamespace(forecast_length=96, context_length=512)
    loaded_model = SimpleNamespace(config=loaded_config)

    monkeypatch.setattr(
        ModelFactory,
        "create_finetune_config",
        staticmethod(lambda **_kwargs: SimpleNamespace()),
    )
    monkeypatch.setattr(
        ModelFactory, "load_model", staticmethod(lambda *_args, **_kwargs: loaded_model)
    )

    _, config = create_model_and_config(
        "moment",
        checkpoint="/tmp/fake-checkpoint",
        forecast_length=120,
    )

    assert config.forecast_length == 96
