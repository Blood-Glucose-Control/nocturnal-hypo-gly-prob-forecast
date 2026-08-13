"""Tests for the ModelAdapter protocol and NaiveBaseline reference port."""

from src.models.adapter import AdapterMetadata, ModelAdapter, assert_model_adapter
from src.models.naive_baseline import (
    NaiveBaselineConfig,
    NaiveBaselineForecaster,
    build_naive_baseline_adapter,
)


def test_naive_baseline_builds_protocol_adapter():
    cfg = NaiveBaselineConfig()
    adapter = build_naive_baseline_adapter(cfg)
    assert isinstance(adapter, NaiveBaselineForecaster)
    assert adapter.config.model_type == "naive_baseline"
    assert adapter.is_fitted is False


def test_naive_baseline_is_runtime_model_adapter():
    model = NaiveBaselineForecaster(NaiveBaselineConfig())
    protocol_typed = assert_model_adapter(model)
    assert protocol_typed is model
    assert isinstance(model, ModelAdapter)


def test_adapter_metadata_from_naive_baseline():
    model = NaiveBaselineForecaster(NaiveBaselineConfig())
    metadata = AdapterMetadata.from_adapter(model)
    assert metadata.model_type == "naive_baseline"
    assert metadata.supports_zero_shot is False
    assert metadata.supports_probabilistic_forecast is True
    assert metadata.is_fitted is False
