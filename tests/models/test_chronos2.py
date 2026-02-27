"""
Chronos-2 model class tests.

Focuses on core integration requirements and key edge cases.
Real-data validation lives in scripts/test_chronos2_parity.py (watgpu).

Requires the chronos2 virtual environment (.venvs/chronos2) which includes
AutoGluon. Tests are automatically skipped in other environments via conftest.py.

Run:
    make test-chronos2                               # recommended
    .venvs/chronos2/bin/python -m pytest tests/models/ -v -k chronos2
    pytest tests/models/test_chronos2.py -v -m slow  # GPU-only slow tests
"""

import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

# All Chronos-2 tests require AutoGluon — skip the whole module outside the
# chronos2 venv (conftest.py also enforces this at collection time).
pytest.importorskip("autogluon.timeseries")

from src.models.chronos2.config import Chronos2Config  # noqa: E402
from src.models.chronos2.model import Chronos2Forecaster  # noqa: E402
from src.models.chronos2.utils import (  # noqa: E402
    build_midnight_episodes,
    convert_to_patient_dict,
    format_segments_for_autogluon,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_patient_df(n_days=5, include_iob=True, start="2024-01-01"):
    """Single-patient DataFrame with DatetimeIndex, 5-min intervals."""
    n = n_days * 288  # 288 points per day at 5-min
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {"bg_mM": rng.normal(8.0, 2.0, n).clip(2.2, 22.0)},
        index=pd.date_range(start, periods=n, freq="5min", name="datetime"),
    )
    if include_iob:
        df["iob"] = rng.exponential(0.5, n).clip(0, 5)
    return df


def _make_flat_df(n_patients=3, n_days=5, include_iob=True):
    """Flat DataFrame mimicking registry output (concatenated patients)."""
    frames = []
    for i in range(1, n_patients + 1):
        pdf = _make_patient_df(
            n_days=n_days, include_iob=include_iob, start=f"2024-{i:02d}-01"
        )
        pdf = pdf.reset_index()
        pdf["p_num"] = float(i)  # Brown 2019 uses float64 patient IDs
        frames.append(pdf)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Core integration tests
# ---------------------------------------------------------------------------


class TestChronos2:
    """Each test maps to a pipeline requirement or real failure mode."""

    def test_config_and_factory(self):
        """Config produces valid AutoGluon hyperparams; factory creates model."""
        # Hyperparameters must be correct or training is silently wrong
        cfg = Chronos2Config(training_mode="fine_tune", fine_tune_steps=5000)
        hp = cfg.get_autogluon_hyperparameters()
        assert hp["Chronos2"]["fine_tune"] is True
        assert hp["Chronos2"]["fine_tune_steps"] == 5000
        assert hp["Chronos2"]["context_length"] == 512

        # Zero-shot must disable fine-tuning
        zs_cfg = Chronos2Config(training_mode="zero_shot", fine_tune_steps=0)
        hp_zs = zs_cfg.get_autogluon_hyperparameters()
        assert hp_zs["Chronos2"]["fine_tune"] is False

        # Factory routing — pipeline depends on this
        from src.models.base.base_model import create_model_from_config

        for model_type in ("chronos2", "chronos"):
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as f:
                json.dump({"model_type": model_type}, f)
                tmp = f.name
            try:
                assert isinstance(create_model_from_config(tmp), Chronos2Forecaster)
            finally:
                os.unlink(tmp)

    def test_flat_df_to_patient_dict(self):
        """Registry flat df → patient dict. Regression: float 1.0 → key "1" not "1.0"."""
        flat = _make_flat_df(n_patients=3, n_days=2)
        flat = flat.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
        result = convert_to_patient_dict(flat, "p_num", "datetime")

        assert set(result.keys()) == {"1", "2", "3"}  # clean strings, not "1.0"
        for pdf in result.values():
            assert isinstance(pdf.index, pd.DatetimeIndex)
            assert pdf.index.is_monotonic_increasing
            assert "p_num" not in pdf.columns

    def test_midnight_episodes(self):
        """Episodes are midnight-anchored with correct dimensions."""
        episodes = build_midnight_episodes(
            _make_patient_df(n_days=10),
            "bg_mM",
            ["iob"],
            interval_mins=5,
            context_len=512,
            horizon=72,
        )
        assert len(episodes) > 0

        ep = episodes[0]
        assert ep["anchor"].hour == 0 and ep["anchor"].minute == 0
        assert ep["context_df"].shape[0] == 512
        assert ep["target_bg"].shape == (72,)
        assert ep["future_covariates"]["iob"].shape == (72,)
        assert not np.isnan(ep["target_bg"]).any()
        # Context ends exactly one interval before midnight anchor
        assert ep["context_df"].index[-1] + pd.Timedelta(minutes=5) == ep["anchor"]

    def test_insufficient_data(self):
        """Short data and missing covariates → empty results, not crashes."""
        # Too short for context + horizon (288 < 512+72)
        assert (
            build_midnight_episodes(
                _make_patient_df(n_days=1),
                "bg_mM",
                ["iob"],
                interval_mins=5,
                context_len=512,
                horizon=72,
            )
            == []
        )

        # No covariate columns at all
        assert (
            build_midnight_episodes(
                _make_patient_df(n_days=10, include_iob=False),
                "bg_mM",
                ["iob"],
                interval_mins=5,
                context_len=512,
                horizon=72,
            )
            == []
        )

    def test_predict_before_fit(self):
        """predict/evaluate before fit → clear ValueError, not AttributeError."""
        model = Chronos2Forecaster(Chronos2Config())
        flat = _make_flat_df(n_patients=1, n_days=5)

        with pytest.raises(ValueError, match="fitted or loaded"):
            model.predict(flat)
        with pytest.raises(ValueError, match="fitted or loaded"):
            model.evaluate(flat)

    def test_data_pipeline(self):
        """Core pipeline: flat df → segments → TimeSeriesDataFrame with covariates."""
        from src.data.preprocessing.gap_handling import segment_all_patients

        flat = _make_flat_df(n_patients=2, n_days=5)
        patient_dict = convert_to_patient_dict(flat, "p_num", "datetime")
        segments = segment_all_patients(
            patient_dict,
            imputation_threshold_mins=45,
            min_segment_length=584,
        )
        ts = format_segments_for_autogluon(segments, "bg_mM", ["iob"])

        assert ts.num_items == len(segments)
        assert ts["target"].isna().sum() == 0
        assert ts["iob"].isna().sum() == 0

    def test_covariate_robustness(self):
        """Missing covariate column → zeros; NaN → forward-filled."""
        seg_no_iob = _make_patient_df(n_days=2, include_iob=False)
        ts1 = format_segments_for_autogluon({"seg": seg_no_iob}, "bg_mM", ["iob"])
        assert (ts1["iob"] == 0.0).all()

        seg_nan = _make_patient_df(n_days=2, include_iob=True)
        seg_nan.iloc[10:20, seg_nan.columns.get_loc("iob")] = np.nan
        ts2 = format_segments_for_autogluon({"seg": seg_nan}, "bg_mM", ["iob"])
        assert ts2["iob"].isna().sum() == 0


# ---------------------------------------------------------------------------
# GPU-only end-to-end tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestChronos2GPU:
    """Run with: pytest tests/models/test_chronos2.py -m slow"""

    def test_fit_predict_evaluate(self):
        """Full pipeline: config → fit → predict → evaluate."""
        config = Chronos2Config(fine_tune_steps=1, min_segment_length=100)
        model = Chronos2Forecaster(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.fit(_make_flat_df(n_patients=2, n_days=10), output_dir=tmpdir)
            assert model.is_fitted and model.predictor is not None

            results = model.evaluate(_make_flat_df(n_patients=1, n_days=10))
            assert "rmse" in results and results["n_episodes"] >= 0

    def test_save_and_load(self):
        """Trained model persists and reloads correctly."""
        config = Chronos2Config(fine_tune_steps=1, min_segment_length=100)
        model = Chronos2Forecaster(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.fit(_make_flat_df(n_patients=2, n_days=10), output_dir=tmpdir)
            model.save(tmpdir)

            loaded = Chronos2Forecaster.load(tmpdir, config=config)
            assert loaded.is_fitted and loaded.predictor is not None
