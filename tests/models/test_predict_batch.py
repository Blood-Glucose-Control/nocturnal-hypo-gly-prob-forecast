"""
Tests for predict_batch() / _predict_batch() on BaseTimeSeriesFoundationModel.

These tests use a lightweight stub model so they run in any environment
without model-specific virtual environments (no AutoGluon, tsfm_public, etc.).
"""

import logging
import unittest.mock as mock

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Minimal stub model (no GPU / ML deps needed)
# ---------------------------------------------------------------------------

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


class _StubModel(BaseTimeSeriesFoundationModel):
    """Minimal concrete subclass — delegates all abstract methods as no-ops."""

    @property
    def training_backend(self) -> TrainingBackend:
        return TrainingBackend.CUSTOM

    @property
    def supports_lora(self) -> bool:
        return False

    @property
    def supports_zero_shot(self) -> bool:
        return True

    def _predict(self, data: pd.DataFrame, **kwargs) -> np.ndarray:
        # Return a fixed forecast of length 3 for easy assertion.
        return np.array([1.0, 2.0, 3.0])

    def _initialize_model(self) -> None:
        pass

    def _prepare_training_data(self, train_data):
        return None, None, None

    def _save_checkpoint(self, output_dir: str) -> None:
        pass

    def _load_checkpoint(self, model_dir: str) -> None:
        pass

    def _train_model(self, train_data, output_dir, **kwargs):
        return {}


class _BatchStubModel(_StubModel):
    """Stub that overrides _predict_batch() to verify override path is taken."""

    def _predict_batch(self, data: pd.DataFrame, episode_col: str):
        # Return a known sentinel value to distinguish from default loop.
        return {
            str(ep_id): np.array([10.0, 20.0]) for ep_id in data[episode_col].unique()
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_panel(episode_ids=("ep_a", "ep_b"), n_rows=5):
    """Build a minimal panel DataFrame with episode_id column."""
    rows = []
    for eid in episode_ids:
        ts = pd.date_range("2024-01-01", periods=n_rows, freq="5min")
        for t in ts:
            rows.append(
                {
                    "episode_id": eid,
                    "datetime": t,
                    "bg_mM": np.random.default_rng(42).uniform(4, 12),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests: default sequential loop
# ---------------------------------------------------------------------------


class TestDefaultPredictBatch:
    def setup_method(self):
        config = ModelConfig()
        self.model = _StubModel(config)

    def test_returns_dict(self):
        panel = _make_panel(episode_ids=["ep_0", "ep_1"])
        result = self.model.predict_batch(panel)
        assert isinstance(result, dict)

    def test_keys_match_episode_ids(self):
        panel = _make_panel(episode_ids=["ep_0", "ep_1", "ep_2"])
        result = self.model.predict_batch(panel)
        assert set(result.keys()) == {"ep_0", "ep_1", "ep_2"}

    def test_keys_are_strings(self):
        """Episode IDs must be coerced to str (int IDs are common)."""
        panel = _make_panel(episode_ids=["ep_0", "ep_1"])
        # Replace string IDs with integers
        panel["episode_id"] = panel["episode_id"].map({"ep_0": 0, "ep_1": 1})
        result = self.model.predict_batch(panel)
        assert set(result.keys()) == {"0", "1"}

    def test_values_are_numpy_arrays(self):
        panel = _make_panel(episode_ids=["ep_x"])
        result = self.model.predict_batch(panel)
        assert isinstance(result["ep_x"], np.ndarray)

    def test_custom_episode_col(self):
        """predict_batch() must respect a custom episode column name."""
        panel = _make_panel(episode_ids=["a", "b"])
        panel = panel.rename(columns={"episode_id": "night_id"})
        result = self.model.predict_batch(panel, episode_col="night_id")
        assert set(result.keys()) == {"a", "b"}

    def test_single_episode(self):
        panel = _make_panel(episode_ids=["solo"])
        result = self.model.predict_batch(panel)
        assert list(result.keys()) == ["solo"]
        np.testing.assert_array_equal(result["solo"], np.array([1.0, 2.0, 3.0]))

    def test_empty_panel_returns_empty_dict(self):
        panel = pd.DataFrame(columns=["episode_id", "datetime", "bg_mM"])
        result = self.model.predict_batch(panel)
        assert result == {}


# ---------------------------------------------------------------------------
# Tests: _predict_batch() override
# ---------------------------------------------------------------------------


class TestBatchOverride:
    def setup_method(self):
        config = ModelConfig()
        self.model = _BatchStubModel(config)

    def test_override_is_called(self):
        """predict_batch() must dispatch to the overridden _predict_batch()."""
        panel = _make_panel(episode_ids=["ep_0", "ep_1"])
        result = self.model.predict_batch(panel)
        # _BatchStubModel returns [10., 20.] — not [1., 2., 3.] from predict()
        np.testing.assert_array_equal(result["ep_0"], np.array([10.0, 20.0]))

    def test_all_episodes_present(self):
        panel = _make_panel(episode_ids=["a", "b", "c"])
        result = self.model.predict_batch(panel)
        assert set(result.keys()) == {"a", "b", "c"}

    def test_empty_panel_returns_empty_dict_with_override(self):
        """Even with an overridden _predict_batch(), empty input must yield {}."""
        panel = pd.DataFrame(columns=["episode_id", "datetime", "bg_mM"])
        result = self.model.predict_batch(panel)
        assert result == {}


# ---------------------------------------------------------------------------
# Tests: validation and warnings
# ---------------------------------------------------------------------------


class TestPredictBatchValidation:
    def setup_method(self):
        config = ModelConfig()
        self.model = _StubModel(config)

    def test_missing_episode_col_raises(self):
        panel = _make_panel(episode_ids=["ep_0"])
        panel = panel.drop(columns=["episode_id"])
        with pytest.raises(ValueError, match="Column 'episode_id' not found"):
            self.model.predict_batch(panel)

    def test_missing_episode_col_custom_name(self):
        panel = _make_panel(episode_ids=["ep_0"])
        with pytest.raises(ValueError, match="Column 'night_id' not found"):
            self.model.predict_batch(panel, episode_col="night_id")

    def test_missing_episode_warning(self, caplog):
        """Warn when _predict_batch drops an episode."""

        class _DroppingModel(_StubModel):
            def _predict_batch(self, data, episode_col):
                # Only return results for the first episode
                ids = list(data[episode_col].unique())
                return {str(ids[0]): np.array([1.0])}

        model = _DroppingModel(ModelConfig())
        panel = _make_panel(episode_ids=["ep_0", "ep_1"])
        with caplog.at_level(logging.WARNING):
            result = model.predict_batch(panel)
        assert "ep_1" not in result
        assert "1 episode(s) produced no predictions" in caplog.text
