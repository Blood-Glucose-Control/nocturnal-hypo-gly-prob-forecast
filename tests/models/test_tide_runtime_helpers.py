"""Focused runtime helper coverage for TiDE WS1 extraction."""
# pyright: reportMissingImports=false

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from src.models.tide.config import TiDEConfig
from src.models.tide.model import TiDEForecaster


def _build_test_model(**config_overrides: object) -> TiDEForecaster:
    model = object.__new__(TiDEForecaster)
    model.config = TiDEConfig(**config_overrides)
    model.logger = logging.getLogger("tests.tide.runtime_helpers")
    model.predictor = None
    return model


def _build_prediction_frame() -> pd.DataFrame:
    index = pd.MultiIndex.from_product(
        [["e1", "e2"], pd.date_range("2024-01-01", periods=2, freq="5min")],
        names=["item_id", "timestamp"],
    )
    return pd.DataFrame(
        {
            "mean": [1.0, 2.0, 3.0, 4.0],
            "0.1": [0.5, 1.5, 2.5, 3.5],
            "0.5": [1.1, 2.1, 3.1, 4.1],
        },
        index=index,
    )


def test_build_predictor_kwargs_uses_frequency_and_quantiles() -> None:
    model = _build_test_model(interval_mins=15, quantile_levels=None)
    kwargs = model._build_predictor_kwargs(output_dir="/tmp/tide")

    assert kwargs["freq"] == "15min"
    assert kwargs["path"] == "/tmp/tide"
    assert kwargs["quantile_levels"] == model.DEFAULT_QUANTILE_LEVELS

    model.config.quantile_levels = [0.2, 0.8]
    kwargs = model._build_predictor_kwargs(output_dir="/tmp/tide")
    assert kwargs["quantile_levels"] == [0.2, 0.8]


def test_build_prediction_context_fills_missing_covariates() -> None:
    model = _build_test_model(covariate_cols=["iob"])
    data = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2024-01-01T00:00:00", "2024-01-01T00:05:00"]),
            "bg_mM": [5.1, 5.2],
        }
    )

    context = model._build_prediction_context(data, item_id_column=None)

    assert context.index.names == ["item_id", "timestamp"]
    assert list(context.columns) == ["target", "iob"]
    np.testing.assert_allclose(context["target"].to_numpy(), np.array([5.1, 5.2]))
    np.testing.assert_allclose(context["iob"].to_numpy(), np.array([0.0, 0.0]))
    assert set(context.index.get_level_values("item_id")) == {"ep_0"}


def test_extract_quantile_predictions_checks_missing_levels() -> None:
    model = _build_test_model()
    episode_predictions = pd.DataFrame(
        {"mean": [1.0, 2.0], "0.1": [0.8, 1.8], "0.5": [1.1, 2.1]}
    )

    quantiles = model._extract_quantile_predictions(episode_predictions, [0.1, 0.5])
    assert quantiles.shape == (2, 2)
    np.testing.assert_allclose(quantiles[0], np.array([0.8, 1.8]))
    np.testing.assert_allclose(quantiles[1], np.array([1.1, 2.1]))

    with pytest.raises(ValueError, match="Quantile levels"):
        model._extract_quantile_predictions(episode_predictions, [0.9])


def test_collect_batch_predictions_handles_missing_episode_ids() -> None:
    model = _build_test_model()
    ag_predictions = _build_prediction_frame()

    point_preds = model._collect_batch_predictions(
        ag_predictions=ag_predictions,
        episode_ids=["e1", "missing", "e2"],
        quantile_levels=None,
    )
    assert list(point_preds) == ["e1", "e2"]
    np.testing.assert_allclose(point_preds["e1"], np.array([1.0, 2.0]))
    np.testing.assert_allclose(point_preds["e2"], np.array([3.0, 4.0]))

    quantile_preds = model._collect_batch_predictions(
        ag_predictions=ag_predictions,
        episode_ids=["e2"],
        quantile_levels=[0.5, 0.1],
    )
    assert quantile_preds["e2"].shape == (2, 2)
    np.testing.assert_allclose(quantile_preds["e2"][0], np.array([3.1, 4.1]))
    np.testing.assert_allclose(quantile_preds["e2"][1], np.array([2.5, 3.5]))
