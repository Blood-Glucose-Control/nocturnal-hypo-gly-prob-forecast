"""Tests for the base class predict() guard logic.

Verifies the four states of (is_fitted, supports_zero_shot):
  1. not fitted + not supports_zero_shot → RuntimeError
  2. fitted + not supports_zero_shot → delegates to _predict()
  3. not fitted + supports_zero_shot → delegates to _predict()
  4. fitted + supports_zero_shot → delegates to _predict()
"""

import numpy as np
import pandas as pd
import pytest

from src.models.base import BaseTimeSeriesFoundationModel, ModelConfig, TrainingBackend


class _StubConfig(ModelConfig):
    """Minimal config for testing."""

    def __init__(self, **kwargs):
        super().__init__(
            model_path="stub", context_length=10, forecast_length=5, **kwargs
        )


class _StubModel(BaseTimeSeriesFoundationModel):
    """Concrete stub with configurable supports_zero_shot."""

    def __init__(self, zero_shot: bool):
        self._zero_shot = zero_shot
        super().__init__(_StubConfig())

    @property
    def training_backend(self) -> TrainingBackend:
        return TrainingBackend.PYTORCH

    @property
    def supports_lora(self) -> bool:
        return False

    @property
    def supports_zero_shot(self) -> bool:
        return self._zero_shot

    def _initialize_model(self) -> None:
        pass

    def _prepare_training_data(self, train_data, **kwargs):
        return (None, None, None)

    def _train_model(self, train_data, output_dir, **kwargs):
        return {}

    def _predict(self, data, **kwargs):
        return np.ones(5)

    def _save_checkpoint(self, output_dir):
        pass

    def _load_checkpoint(self, model_dir):
        pass


_DUMMY_DF = pd.DataFrame({"bg_mM": np.random.rand(10)})


class TestPredictGuard:
    """Test the four (is_fitted, supports_zero_shot) states."""

    def test_not_fitted_not_zero_shot_raises(self):
        model = _StubModel(zero_shot=False)
        assert not model.is_fitted
        with pytest.raises(RuntimeError, match="Call fit\\(\\) or load\\(\\) first"):
            model.predict(_DUMMY_DF)

    def test_fitted_not_zero_shot_delegates(self):
        model = _StubModel(zero_shot=False)
        model.is_fitted = True
        result = model.predict(_DUMMY_DF)
        np.testing.assert_array_equal(result, np.ones(5))

    def test_not_fitted_zero_shot_delegates(self):
        model = _StubModel(zero_shot=True)
        assert not model.is_fitted
        result = model.predict(_DUMMY_DF)
        np.testing.assert_array_equal(result, np.ones(5))

    def test_fitted_zero_shot_delegates(self):
        model = _StubModel(zero_shot=True)
        model.is_fitted = True
        result = model.predict(_DUMMY_DF)
        np.testing.assert_array_equal(result, np.ones(5))
