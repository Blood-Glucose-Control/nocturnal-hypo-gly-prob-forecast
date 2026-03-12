"""Tests for the predict_quantiles() base class dispatch logic."""


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

    def _predict(self, data, **kwargs):
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

    def _predict_quantiles(self, data, quantile_levels, **kwargs):
        n_q = len(quantile_levels)
        return np.ones((n_q, self.config.forecast_length))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPredictQuantilesDispatch:
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
            model.predict_quantiles(pd.DataFrame({"bg_mM": [1, 2, 3]}))

    def test_supported_model_returns_array(self):
        model = self._make_model(_ProbModel)
        result = model.predict_quantiles(pd.DataFrame({"bg_mM": range(10)}))
        assert isinstance(result, np.ndarray)
        # Default levels: [0.1, 0.2, ..., 0.9] = 9 quantiles
        assert result.shape == (9, 5)

    def test_kwarg_levels_override_config(self):
        model = self._make_model(_ProbModel, quantile_levels=[0.1, 0.5, 0.9])
        custom = [0.25, 0.75]
        result = model.predict_quantiles(
            pd.DataFrame({"bg_mM": range(10)}), quantile_levels=custom
        )
        assert result.shape[0] == 2  # kwarg wins over config

    def test_config_levels_override_default(self):
        model = self._make_model(_ProbModel, quantile_levels=[0.1, 0.5, 0.9])
        result = model.predict_quantiles(pd.DataFrame({"bg_mM": range(10)}))
        assert result.shape[0] == 3  # config wins over DEFAULT_QUANTILE_LEVELS

    def test_default_levels_used_when_no_override(self):
        model = self._make_model(_ProbModel)
        assert model.config.quantile_levels is None  # no config override
        result = model.predict_quantiles(pd.DataFrame({"bg_mM": range(10)}))
        assert result.shape[0] == len(model.DEFAULT_QUANTILE_LEVELS)

    def test_supports_probabilistic_forecast_default_false(self):
        model = self._make_model(_PointOnlyModel)
        assert model.supports_probabilistic_forecast is False

    def test_supports_probabilistic_forecast_override_true(self):
        model = self._make_model(_ProbModel)
        assert model.supports_probabilistic_forecast is True
