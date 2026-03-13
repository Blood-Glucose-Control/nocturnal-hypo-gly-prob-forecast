"""Tests for the probabilistic forecasting API (quantile_levels parameter)."""

import numpy as np
import pandas as pd
import pytest

from src.models.base.base_model import (
    BaseTimeSeriesFoundationModel,
    ModelConfig,
    TrainingBackend,
)


# ---------------------------------------------------------------------------
# Minimal concrete subclasses for testing
# ---------------------------------------------------------------------------


class _PointOnlyModel(BaseTimeSeriesFoundationModel):
    """Model that does NOT support probabilistic forecasting."""

    @property
    def supports_zero_shot(self) -> bool:
        return True

    @property
    def supports_lora(self) -> bool:
        return False

    @property
    def training_backend(self) -> TrainingBackend:
        return TrainingBackend.CUSTOM

    def _predict(self, data, quantile_levels=None, **kwargs):
        return np.zeros(self.config.forecast_length)

    def _initialize_model(self):
        pass

    def _prepare_training_data(self, train_data, val_data=None, **kwargs):
        return train_data, val_data

    def _train_model(self, train_data, val_data, output_dir):
        pass

    def _save_checkpoint(self, path):
        pass

    def _load_checkpoint(self, path):
        pass


class _ProbModel(_PointOnlyModel):
    """Model that supports probabilistic forecasting."""

    @property
    def supports_probabilistic_forecast(self) -> bool:
        return True

    def _predict(self, data, quantile_levels=None, **kwargs):
        if quantile_levels is not None:
            n_q = len(quantile_levels)
            return np.ones((n_q, self.config.forecast_length))
        return np.zeros(self.config.forecast_length)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPredictWithQuantileLevels:
    def _make_model(self, cls, **config_kwargs):
        defaults = dict(
            model_type="test",
            model_path="test",
            context_length=10,
            forecast_length=5,
        )
        defaults.update(config_kwargs)
        config = ModelConfig(**defaults)
        return cls(config)

    def test_unsupported_model_raises(self):
        model = self._make_model(_PointOnlyModel)
        with pytest.raises(NotImplementedError, match="does not support"):
            model.predict(
                pd.DataFrame({"bg_mM": [1, 2, 3]}),
                quantile_levels=[0.1, 0.5, 0.9],
            )

    def test_supported_model_returns_array(self):
        model = self._make_model(_ProbModel)
        result = model.predict(
            pd.DataFrame({"bg_mM": range(10)}),
            quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        )
        assert isinstance(result, np.ndarray)
        assert result.shape == (9, 5)

    def test_point_forecast_without_quantile_levels(self):
        model = self._make_model(_ProbModel)
        result = model.predict(pd.DataFrame({"bg_mM": range(10)}))
        assert isinstance(result, np.ndarray)
        assert result.shape == (5,)

    def test_kwarg_levels_override_config(self):
        model = self._make_model(_ProbModel, quantile_levels=[0.1, 0.5, 0.9])
        custom = [0.25, 0.75]
        result = model.predict(
            pd.DataFrame({"bg_mM": range(10)}), quantile_levels=custom
        )
        assert result.shape[0] == 2  # kwarg wins over config

    def test_supports_probabilistic_forecast_default_false(self):
        model = self._make_model(_PointOnlyModel)
        assert model.supports_probabilistic_forecast is False

    def test_supports_probabilistic_forecast_override_true(self):
        model = self._make_model(_ProbModel)
        assert model.supports_probabilistic_forecast is True


class TestPredictBatchWithQuantileLevels:
    def _make_model(self, cls, **config_kwargs):
        defaults = dict(
            model_type="test",
            model_path="test",
            context_length=10,
            forecast_length=5,
        )
        defaults.update(config_kwargs)
        config = ModelConfig(**defaults)
        return cls(config)

    def _make_panel(self, n_episodes=3, n_rows=10):
        dfs = []
        for i in range(n_episodes):
            df = pd.DataFrame({"bg_mM": np.random.rand(n_rows)})
            df["episode_id"] = f"ep_{i}"
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True)

    def test_batch_point_forecast(self):
        model = self._make_model(_PointOnlyModel)
        panel = self._make_panel()
        results = model.predict_batch(panel, episode_col="episode_id")
        assert len(results) == 3
        for v in results.values():
            assert v.shape == (5,)

    def test_batch_quantile_unsupported_raises(self):
        model = self._make_model(_PointOnlyModel)
        panel = self._make_panel()
        with pytest.raises(NotImplementedError, match="does not support"):
            model.predict_batch(
                panel, episode_col="episode_id", quantile_levels=[0.1, 0.5, 0.9]
            )

    def test_batch_quantile_supported(self):
        model = self._make_model(_ProbModel)
        panel = self._make_panel()
        results = model.predict_batch(
            panel, episode_col="episode_id", quantile_levels=[0.1, 0.5, 0.9]
        )
        assert len(results) == 3
        for v in results.values():
            assert v.shape == (3, 5)
