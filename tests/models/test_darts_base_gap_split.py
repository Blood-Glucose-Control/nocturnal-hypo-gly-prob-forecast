"""Tests for Darts base segment splitting on timestamp discontinuities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from src.models.darts_base import DartsGlobalModelBase  # noqa: E402
from src.models.tsmixer.config import TSMixerConfig  # noqa: E402


class _DummyDartsModel(DartsGlobalModelBase):
    def _create_darts_model(self):
        return object()

    def _load_darts_model(self, model_path: str):
        return object()


class _FakeForecast:
    def __init__(self, values: np.ndarray, components: list[str]) -> None:
        self._values = values
        self.components = components

    def values(self, copy: bool = False) -> np.ndarray:
        del copy
        return self._values


class _FakePredictor:
    def __init__(self, forecast: _FakeForecast) -> None:
        self._forecast = forecast
        self.last_kwargs: dict[str, object] | None = None

    def predict(self, **kwargs):
        self.last_kwargs = kwargs
        return self._forecast


def test_split_segment_on_time_gaps_breaks_discontinuous_series() -> None:
    idx = pd.to_datetime(
        [
            "2020-01-01 00:00:00",
            "2020-01-01 00:05:00",
            "2020-01-01 00:10:00",
            "2020-01-01 00:15:00",
            "2020-01-02 00:00:00",
            "2020-01-02 00:05:00",
            "2020-01-02 00:10:00",
            "2020-01-02 00:15:00",
        ]
    )
    segment = pd.DataFrame(
        {"bg_mM": [5.0, 5.1, 5.2, 5.3, 6.0, 6.1, 6.2, 6.3]}, index=idx
    )

    model = _DummyDartsModel(
        TSMixerConfig(context_length=4, forecast_length=2, min_segment_length=3)
    )
    chunks = model._split_segment_on_time_gaps(
        segment_df=segment,
        expected_delta=pd.Timedelta(minutes=5),
        min_chunk_length=3,
    )

    assert [len(chunk) for chunk in chunks] == [4, 4]


def test_split_segment_on_time_gaps_drops_short_chunks() -> None:
    idx = pd.to_datetime(
        [
            "2020-01-01 00:00:00",
            "2020-01-01 00:05:00",
            "2020-01-02 00:00:00",
            "2020-01-02 00:05:00",
            "2020-01-02 00:10:00",
            "2020-01-02 00:15:00",
        ]
    )
    segment = pd.DataFrame({"bg_mM": [5.0, 5.1, 6.0, 6.1, 6.2, 6.3]}, index=idx)

    model = _DummyDartsModel(
        TSMixerConfig(context_length=4, forecast_length=2, min_segment_length=3)
    )
    chunks = model._split_segment_on_time_gaps(
        segment_df=segment,
        expected_delta=pd.Timedelta(minutes=5),
        min_chunk_length=3,
    )

    assert len(chunks) == 1
    assert len(chunks[0]) == 4


def test_predict_quantiles_returns_requested_quantiles_in_requested_order() -> None:
    model = _DummyDartsModel(TSMixerConfig(context_length=4, forecast_length=3))
    model._to_target_and_covariates = lambda _df: ("target_series", "past_covariates")
    forecast = _FakeForecast(
        values=np.array(
            [
                [1.0, 10.0, 100.0],
                [2.0, 20.0, 200.0],
                [3.0, 30.0, 300.0],
            ]
        ),
        components=["bg_mM_q0.1", "bg_mM_q0.5", "bg_mM_q0.9"],
    )
    predictor = _FakePredictor(forecast)
    model.model = predictor

    result = model._predict(
        pd.DataFrame({"bg_mM": [1.0, 2.0, 3.0]}),
        quantile_levels=[0.9, 0.1],
    )

    assert predictor.last_kwargs is not None
    assert predictor.last_kwargs["predict_likelihood_parameters"] is True
    assert result.shape == (2, 3)
    np.testing.assert_allclose(result[0], np.array([100.0, 200.0, 300.0]))
    np.testing.assert_allclose(result[1], np.array([1.0, 2.0, 3.0]))


def test_predict_quantiles_raises_when_requested_quantile_is_unavailable() -> None:
    model = _DummyDartsModel(TSMixerConfig(context_length=4, forecast_length=3))
    model._to_target_and_covariates = lambda _df: ("target_series", None)
    forecast = _FakeForecast(
        values=np.array(
            [
                [1.0, 10.0],
                [2.0, 20.0],
                [3.0, 30.0],
            ]
        ),
        components=["bg_mM_q0.1", "bg_mM_q0.5"],
    )
    model.model = _FakePredictor(forecast)

    with pytest.raises(ValueError, match="Requested quantile 0.9 not available"):
        model._predict(
            pd.DataFrame({"bg_mM": [1.0, 2.0, 3.0]}),
            quantile_levels=[0.9],
        )
