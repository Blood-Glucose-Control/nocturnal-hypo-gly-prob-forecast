"""Unit tests for TTM preprocessor schema validation."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("tsfm_public")


class _UnsupportedSchemaPreprocessor:
    """Minimal stand-in for a preprocessor missing required schema fields."""

    def __init__(self):
        self.target_columns = ["bg_mM"]


def test_unsupported_schema_preprocessor_is_rejected_with_actionable_error():
    from src.models.ttm.model import _validate_preprocessor_schema

    preprocessor = _UnsupportedSchemaPreprocessor()
    assert not hasattr(preprocessor, "other_columns_to_scale")

    with pytest.raises(ValueError, match="unsupported by the current runtime"):
        _validate_preprocessor_schema(preprocessor)  # type: ignore[arg-type]


def test_zero_shot_predict_batch_includes_episode_col_in_pipeline_ids(monkeypatch):
    import src.models.ttm.model as ttm_model_module
    from src.models.ttm.config import TTMConfig
    from src.models.ttm.model import TTMForecaster

    captured: dict[str, object] = {}

    class _DummyPipeline:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
            target_col = str(captured["target_columns"][0])  # type: ignore[index]
            out = data[["episode_id"]].copy()
            out[target_col] = data["bg_mM"].to_numpy() + 1.0
            return out

    def _stub_initialize_model(self: TTMForecaster) -> None:
        self.model = object()

    monkeypatch.setattr(TTMForecaster, "_initialize_model", _stub_initialize_model)
    monkeypatch.setattr(
        ttm_model_module, "TimeSeriesForecastingPipeline", _DummyPipeline
    )

    model = TTMForecaster(
        TTMConfig(
            training_mode="zero_shot",
            num_epochs=0,
            input_features=["iob"],
            target_features=["bg_mM"],
        )
    )
    panel = pd.DataFrame(
        {
            "episode_id": ["ep1", "ep1", "ep2", "ep2"],
            "patient_id": ["p1", "p1", "p2", "p2"],
            "datetime": pd.date_range("2024-01-01", periods=4, freq="5min"),
            "bg_mM": [5.0, 6.0, 7.5, 8.5],
            "iob": [0.1, 0.2, 0.3, 0.4],
        }
    )

    results = model._predict_batch(panel, episode_col="episode_id")

    assert captured["id_columns"] == ["episode_id", "patient_id"]
    assert captured["target_columns"] == ["bg_mM"]
    assert set(results) == {"ep1", "ep2"}
    np.testing.assert_array_equal(results["ep1"], np.array([6.0, 7.0]))
    np.testing.assert_array_equal(results["ep2"], np.array([8.5, 9.5]))
