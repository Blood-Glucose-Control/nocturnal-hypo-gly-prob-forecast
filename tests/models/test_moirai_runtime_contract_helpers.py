"""Runtime helper contract tests for Moirai training/prediction preparation."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch

pytest.importorskip("gluonts")
pytest.importorskip("uni2ts")


def _stub_initialize_model(self):
    self.model = SimpleNamespace(module=SimpleNamespace(patch_sizes=[4, 8]))
    return self.model


def _build_forecaster(monkeypatch) -> Any:
    from src.models.moirai.config import (  # pyright: ignore[reportMissingImports]
        MoiraiConfig,
    )
    from src.models.moirai.model import (  # pyright: ignore[reportMissingImports]
        MoiraiForecaster,
    )

    monkeypatch.setattr(MoiraiForecaster, "_initialize_model", _stub_initialize_model)
    return MoiraiForecaster(
        MoiraiConfig(
            context_length=4,
            forecast_length=2,
            batch_size=2,
            patch_size=4,
            target_col="bg_mM",
            covariate_cols=["iob"],
            past_covariate_dim=1,
        )
    )


def _build_known_future_forecaster(monkeypatch) -> Any:
    from src.models.moirai.config import (  # pyright: ignore[reportMissingImports]
        MoiraiConfig,
    )
    from src.models.moirai.model import (  # pyright: ignore[reportMissingImports]
        MoiraiForecaster,
    )

    monkeypatch.setattr(MoiraiForecaster, "_initialize_model", _stub_initialize_model)
    return MoiraiForecaster(
        MoiraiConfig(
            context_length=4,
            forecast_length=2,
            batch_size=2,
            patch_size=4,
            target_col="bg_mM",
            covariate_cols=["iob"],
            past_covariate_dim=1,
            known_covariate_cols=["total_insulin"],
            known_covariate_dim=1,
        )
    )


def test_prepare_training_tensors_builds_full_window_known_future_covariates(
    monkeypatch,
):
    model = _build_known_future_forecaster(monkeypatch)

    index = pd.date_range("2024-01-01", periods=6, freq="5min")
    frame = pd.DataFrame(
        {
            "patient_id": ["p1"] * 6,
            "datetime": index,
            "bg_mM": [5.0, 5.5, 6.0, 6.5, 6.8, 7.1],
            "iob": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "total_insulin": [1.0, 0.0, 0.0, 2.0, 0.0, 0.0],
        }
    )

    tensors = model._prepare_training_tensors(frame)

    past_covariates = tensors[6]
    known_covariates = tensors[8]
    known_observed = tensors[9]

    assert past_covariates is not None
    assert past_covariates.shape == (1, 4, 1)  # context-length past covariate

    assert known_covariates is not None
    # Known-future covariate spans the full context+horizon window.
    assert known_covariates.shape == (1, 6, 1)
    assert known_observed is not None
    assert known_observed.shape == (1, 6, 1)
    torch.testing.assert_close(
        known_covariates[0, :, 0],
        torch.tensor([1.0, 0.0, 0.0, 2.0, 0.0, 0.0]),
    )


def test_prepare_training_tensors_normalizes_dict_and_list_episode_inputs(monkeypatch):
    model = _build_forecaster(monkeypatch)

    episode = {
        "context_df": pd.DataFrame(
            {
                "bg_mM": [5.0, 5.5, 6.0, 6.5],
                "iob": [0.1, 0.2, 0.3, 0.4],
            }
        ),
        "target_bg": np.array([6.8, 7.1], dtype=np.float32),
    }

    list_tensors = model._prepare_training_tensors([episode])
    dict_tensors = model._prepare_training_tensors({"p1": [episode]})

    torch.testing.assert_close(list_tensors[0], dict_tensors[0])
    torch.testing.assert_close(list_tensors[1], dict_tensors[1])
    assert list_tensors[6] is not None and dict_tensors[6] is not None
    torch.testing.assert_close(list_tensors[6], dict_tensors[6])


def test_prepare_training_data_builds_patched_loader_from_tensor_contract(monkeypatch):
    model = _build_forecaster(monkeypatch)

    class _FakeConvertModel:
        module = SimpleNamespace(patch_sizes=[4, 8])

        @staticmethod
        def _convert(
            patch_size,
            *,
            past_target,
            past_observed_target,
            past_is_pad,
            future_target,
            future_observed_target,
            future_is_pad,
            feat_dynamic_real=None,
            observed_feat_dynamic_real=None,
            past_feat_dynamic_real=None,
            past_observed_feat_dynamic_real=None,
        ):
            _ = (
                patch_size,
                past_is_pad,
                future_is_pad,
                feat_dynamic_real,
                observed_feat_dynamic_real,
                past_feat_dynamic_real,
                past_observed_feat_dynamic_real,
            )
            target = torch.cat([past_target, future_target], dim=1)
            observed = torch.cat([past_observed_target, future_observed_target], dim=1)
            seq_len = target.shape[1]
            batch_size = target.shape[0]
            sample_id = torch.zeros((batch_size, seq_len, 1), dtype=torch.long)
            time_id = (
                torch.arange(seq_len, dtype=torch.long)
                .view(1, seq_len, 1)
                .repeat(batch_size, 1, 1)
            )
            variate_id = torch.zeros((batch_size, seq_len, 1), dtype=torch.long)
            prediction_mask = torch.cat(
                [
                    torch.zeros_like(past_observed_target, dtype=torch.bool),
                    torch.ones_like(future_observed_target, dtype=torch.bool),
                ],
                dim=1,
            )
            return target, observed, sample_id, time_id, variate_id, prediction_mask

    model.model = _FakeConvertModel()

    fake_tensors = (
        torch.randn(3, 4, 1),
        torch.randn(3, 2, 1),
        torch.ones(3, 4, 1, dtype=torch.bool),
        torch.ones(3, 2, 1, dtype=torch.bool),
        torch.zeros(3, 4, dtype=torch.long),
        torch.zeros(3, 2, dtype=torch.long),
        None,
        None,
        None,
        None,
    )
    monkeypatch.setattr(model, "_prepare_training_tensors", lambda _: fake_tensors)

    loader, val_loader, test_loader = model._prepare_training_data(train_data=[])

    assert val_loader is None
    assert test_loader is None
    assert len(loader.dataset) == 3
    sample = loader.dataset[0]
    assert "patch_size" in sample
    assert int(sample["patch_size"][0]) == 4


def test_predict_impl_returns_expected_quantile_and_mean_shapes(monkeypatch):
    model = _build_forecaster(monkeypatch)

    class _Forecast:
        def __init__(self):
            self.samples = np.array(
                [
                    [1.0, 2.0],
                    [2.0, 3.0],
                    [3.0, 4.0],
                    [4.0, 5.0],
                ],
                dtype=np.float32,
            )
            self.mean = np.array([2.5, 3.5], dtype=np.float32)

    class _Predictor:
        @staticmethod
        def predict(_dataset):
            return [_Forecast()]

    monkeypatch.setattr(model, "_get_or_create_predictor", lambda **_: _Predictor())
    monkeypatch.setattr(
        model, "_normalize_predict_input", lambda _data: ("dataset", True)
    )

    quantiles = model._predict_impl(data=object(), quantile_levels=[0.5], batch_size=2)
    means = model._predict_impl(data=object(), quantile_levels=None, batch_size=2)

    assert quantiles.shape == (1, 2)
    assert means.shape == (2,)
    np.testing.assert_allclose(means, np.array([2.5, 3.5], dtype=np.float32))
