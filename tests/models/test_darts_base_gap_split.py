"""Tests for Darts base segment splitting on timestamp discontinuities."""

from __future__ import annotations

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
