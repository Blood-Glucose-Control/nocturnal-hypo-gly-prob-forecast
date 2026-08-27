"""Focused runtime helper regression coverage for Toto WS1 extraction."""
# pyright: reportMissingImports=false

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

pytest.importorskip("toto")

from src.models.toto.config import TotoConfig
from src.models.toto.model import TotoForecaster


def _build_test_model(config: TotoConfig) -> TotoForecaster:
    model = object.__new__(TotoForecaster)
    model.config = config
    model.device = "cpu"
    return model


def _episode_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "episode_id": ["e1", "e1", "e1", "e2", "e2"],
            "bg_mM": [5.0, 5.5, 6.0, 7.0, 7.5],
            "datetime": pd.to_datetime(
                [
                    "2024-01-01T00:00:00",
                    "2024-01-01T00:05:00",
                    "2024-01-01T00:10:00",
                    "2024-01-02T00:00:00",
                    "2024-01-02T00:05:00",
                ]
            ),
        }
    )


def _attach_batch_build_stubs(model: TotoForecaster) -> None:
    model._extract_timestamps = lambda df: df["datetime"].to_numpy()
    model._timestamps_to_seconds = lambda ts: torch.arange(len(ts), dtype=torch.float32)
    model._build_variates = lambda df: [
        torch.tensor(df["bg_mM"].to_numpy(dtype=np.float32), dtype=torch.float32)
    ]


def test_resolve_eval_chunk_size_validates_positive_int() -> None:
    model = _build_test_model(TotoConfig(eval_batch_size=None))
    assert model._resolve_eval_chunk_size(batch_size=3) == 3

    model.config.eval_batch_size = "2"
    assert model._resolve_eval_chunk_size(batch_size=3) == 2

    model.config.eval_batch_size = 0
    with pytest.raises(ValueError, match="positive integer"):
        model._resolve_eval_chunk_size(batch_size=3)


def test_predict_batch_point_maps_chunked_outputs_per_episode() -> None:
    model = _build_test_model(TotoConfig(forecast_length=3, eval_batch_size=1))
    _attach_batch_build_stubs(model)

    call_index = {"value": 0}

    def fake_run_forecast(inputs):
        outputs = []
        for _ in range(inputs.series.shape[0]):
            call_index["value"] += 1
            outputs.append(
                np.full(
                    (model.config.forecast_length,),
                    call_index["value"],
                    dtype=np.float32,
                )
            )
        return np.stack(outputs)

    model._run_forecast = fake_run_forecast
    model.forecaster = SimpleNamespace()

    preds = model._predict_batch(_episode_frame(), episode_col="episode_id")

    assert set(preds.keys()) == {"e1", "e2"}
    np.testing.assert_array_equal(
        preds["e1"], np.array([1.0, 1.0, 1.0], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        preds["e2"], np.array([2.0, 2.0, 2.0], dtype=np.float32)
    )


def test_predict_batch_quantiles_respects_chunking() -> None:
    config = TotoConfig(forecast_length=3, eval_batch_size=1, num_samples=4)
    model = _build_test_model(config)
    _attach_batch_build_stubs(model)

    call_index = {"value": 0}

    def fake_forecast(inputs, prediction_length, num_samples, samples_per_batch):
        _ = (prediction_length, num_samples, samples_per_batch)
        chunk_samples = []
        for _ in range(inputs.series.shape[0]):
            call_index["value"] += 1
            base = call_index["value"] * 10
            sample = torch.tensor(
                np.tile(
                    np.arange(base, base + config.num_samples, dtype=np.float32),
                    (config.forecast_length, 1),
                )
            )
            chunk_samples.append(sample.unsqueeze(0))
        samples = torch.stack(chunk_samples, dim=0)
        return SimpleNamespace(samples=samples)

    model.forecaster = SimpleNamespace(forecast=fake_forecast)

    quantiles = model._predict_batch(
        _episode_frame(),
        episode_col="episode_id",
        quantile_levels=[0.5],
    )

    assert set(quantiles.keys()) == {"e1", "e2"}
    np.testing.assert_allclose(quantiles["e1"], np.array([[11.5, 11.5, 11.5]]))
    np.testing.assert_allclose(quantiles["e2"], np.array([[21.5, 21.5, 21.5]]))
