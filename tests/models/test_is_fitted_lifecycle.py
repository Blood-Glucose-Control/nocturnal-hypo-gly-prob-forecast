"""Tests for is_fitted state transitions through the base class lifecycle.

Verifies that is_fitted is correctly set across the five fundamental
scenarios:

  1. Fresh init (no ZS)      → is_fitted=False, predict() blocked
  2. Fresh init (supports ZS) → is_fitted=False, predict() allowed
  3. After fit() succeeds     → is_fitted=True
  4. After fit() fails        → is_fitted stays False
  5. save() → load() round-trip → is_fitted=True preserved

Uses lightweight stubs — no GPU, no ML dependencies.
"""

import unittest.mock as mock

import numpy as np
import pandas as pd
import pytest
import torch  # noqa: F401  # pre-import so mock.patch.dict(sys.modules) doesn't evict the C extension

# Patch heavy optional deps before importing the base module.
_mocks = {
    "src.utils.logging_helper": mock.MagicMock(),
}
with mock.patch.dict("sys.modules", _mocks):
    from src.models.base.base_model import (  # noqa: E402
        BaseTimeSeriesFoundationModel,
        ModelConfig,
        TrainingBackend,
    )


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _StubModel(BaseTimeSeriesFoundationModel):
    """Minimal concrete subclass with configurable supports_zero_shot."""

    def __init__(self, config=None, *, zero_shot: bool = True):
        self._zero_shot = zero_shot
        super().__init__(config or ModelConfig())

    @property
    def training_backend(self) -> TrainingBackend:
        return TrainingBackend.CUSTOM

    @property
    def supports_lora(self) -> bool:
        return False

    @property
    def supports_zero_shot(self) -> bool:
        return self._zero_shot

    def _initialize_model(self) -> None:
        pass

    def _prepare_training_data(self, train_data):
        return None, None, None

    def _train_model(self, train_data, output_dir, **kwargs):
        return {}

    def _predict(self, data, **kwargs):
        return np.ones(5)

    def _save_checkpoint(self, output_dir):
        pass

    def _load_checkpoint(self, model_dir):
        pass


class _FailingTrainModel(_StubModel):
    """Stub whose _train_model always raises."""

    def _train_model(self, train_data, output_dir, **kwargs):
        raise RuntimeError("training exploded")


_DUMMY_DF = pd.DataFrame({"bg_mM": np.random.default_rng(42).uniform(4, 12, size=10)})


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestIsFittedLifecycle:
    """Verify is_fitted state across the five fundamental scenarios."""

    def test_init_no_zero_shot_is_false_and_blocked(self):
        """Scenario 1: Fresh model without ZS → is_fitted=False, predict blocked."""
        model = _StubModel(zero_shot=False)
        assert not model.is_fitted
        with pytest.raises(RuntimeError, match="requires training"):
            model.predict(_DUMMY_DF)

    def test_init_with_zero_shot_is_false_and_allowed(self):
        """Scenario 2: Fresh model with ZS → is_fitted=False, predict works."""
        model = _StubModel(zero_shot=True)
        assert not model.is_fitted
        result = model.predict(_DUMMY_DF)
        assert isinstance(result, np.ndarray)

    def test_fit_sets_is_fitted_true(self, tmp_path):
        """Scenario 3: fit() succeeds → is_fitted=True."""
        model = _StubModel(zero_shot=False)
        assert not model.is_fitted
        model.fit(train_data=_DUMMY_DF, output_dir=str(tmp_path))
        assert model.is_fitted

    def test_fit_failure_leaves_is_fitted_false(self, tmp_path):
        """Scenario 4: fit() fails → is_fitted stays False."""
        model = _FailingTrainModel(zero_shot=False)
        assert not model.is_fitted
        with pytest.raises(RuntimeError, match="training exploded"):
            model.fit(train_data=_DUMMY_DF, output_dir=str(tmp_path))
        assert not model.is_fitted

    def test_save_load_round_trip_preserves_is_fitted_true(self, tmp_path):
        """Scenario 5a: fit() → save() → load() → is_fitted still True."""
        model = _StubModel(zero_shot=False)
        model.fit(train_data=_DUMMY_DF, output_dir=str(tmp_path))
        assert model.is_fitted

        save_dir = str(tmp_path / "saved_fitted")
        model.save(save_dir)

        loaded = _StubModel.load(save_dir)
        assert loaded.is_fitted

    def test_save_load_round_trip_preserves_is_fitted_false(self, tmp_path):
        """Scenario 5b: save() without fit() → load() → is_fitted still False."""
        model = _StubModel(zero_shot=True)
        assert not model.is_fitted

        save_dir = str(tmp_path / "saved_unfitted")
        model.save(save_dir)

        loaded = _StubModel.load(save_dir)
        assert not loaded.is_fitted
