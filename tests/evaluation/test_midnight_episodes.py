# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)
# For commercial licensing, contact: christopher/cjrisi AT gluroo/uwaterloo DOT com/ca

"""Tests for midnight-anchored episode building."""

import numpy as np
import pandas as pd

from src.evaluation.episode_builders import build_midnight_episodes


def _make_patient_df(
    n_days: int = 5,
    interval_mins: int = 5,
    start: str = "2024-01-01",
    bg_value: float = 7.0,
    include_iob: bool = False,
    iob_value: float = 1.5,
) -> pd.DataFrame:
    """Create a synthetic patient DataFrame with DatetimeIndex."""
    freq = f"{interval_mins}min"
    index = pd.date_range(start, periods=n_days * 24 * 60 // interval_mins, freq=freq)
    data = {"bg_mM": np.full(len(index), bg_value)}
    if include_iob:
        data["iob"] = np.full(len(index), iob_value)
    return pd.DataFrame(data, index=index)


class TestBuildMidnightEpisodes:
    """Test build_midnight_episodes function."""

    def test_episodes_anchored_at_midnight_with_correct_lengths(self):
        """Core behavior: episodes anchor at midnight, context/target have correct sizes."""
        context_len, forecast_len = 144, 72  # 12h context, 6h forecast
        df = _make_patient_df(n_days=5, bg_value=8.0)

        episodes = build_midnight_episodes(
            df, context_length=context_len, forecast_length=forecast_len
        )

        assert len(episodes) == 4  # Jan 2, 3, 4, 5 midnights
        for ep in episodes:
            assert ep["anchor"].hour == 0 and ep["anchor"].minute == 0
            assert len(ep["context_df"]) == context_len
            assert len(ep["target_bg"]) == forecast_len
            assert "bg_mM" in ep["context_df"].columns
            assert np.allclose(ep["target_bg"], 8.0)

    def test_no_episodes_when_data_too_short(self):
        """Boundary: returns empty list if data can't fit context + forecast."""
        df = _make_patient_df(n_days=1)  # 24h can't fit 512-step (~42.7h) context
        assert build_midnight_episodes(df, context_length=512, forecast_length=72) == []

    def test_nan_bg_episodes_skipped(self):
        """Safety: episodes with NaN BG are excluded (missing CGM data)."""
        df = _make_patient_df(n_days=5)

        # Punch a NaN hole in Jan 3's forecast window (00:30-01:00)
        midnight_jan3 = pd.Timestamp("2024-01-03 00:00")
        mask = (df.index >= midnight_jan3 + pd.Timedelta(minutes=30)) & (
            df.index < midnight_jan3 + pd.Timedelta(minutes=60)
        )
        df.loc[mask, "bg_mM"] = np.nan

        anchors = [
            ep["anchor"]
            for ep in build_midnight_episodes(
                df, context_length=144, forecast_length=72
            )
        ]
        assert midnight_jan3 not in anchors

    def test_covariates_returned_when_available(self):
        """Covariate data (IOB) is included in future_covariates dict."""
        df = _make_patient_df(n_days=5, include_iob=True, iob_value=2.0)
        episodes = build_midnight_episodes(
            df, context_length=144, forecast_length=72, covariate_cols=["iob"]
        )

        assert len(episodes) > 0
        for ep in episodes:
            assert "iob" in ep["future_covariates"]
            assert len(ep["future_covariates"]["iob"]) == 72
            assert np.allclose(ep["future_covariates"]["iob"], 2.0)

    def test_works_without_covariates(self):
        """BG-only models: episodes built with empty future_covariates."""
        df = _make_patient_df(n_days=5)  # No IOB column
        episodes = build_midnight_episodes(
            df, context_length=144, forecast_length=72, covariate_cols=["iob"]
        )

        assert len(episodes) > 0
        for ep in episodes:
            assert ep["future_covariates"] == {}

    def test_low_covariate_coverage_skips_episode(self):
        """Clinical threshold: episodes with <50% covariate coverage are excluded."""
        df = _make_patient_df(n_days=5, include_iob=True, iob_value=1.0)

        # Set IOB to NaN for 80% of context before Jan 3 midnight
        midnight_jan3 = pd.Timestamp("2024-01-03 00:00")
        context_start = midnight_jan3 - pd.Timedelta(minutes=144 * 5)
        nan_end = context_start + pd.Timedelta(minutes=int(144 * 5 * 0.8))
        df.loc[context_start:nan_end, "iob"] = np.nan

        anchors = [
            ep["anchor"]
            for ep in build_midnight_episodes(
                df, context_length=144, forecast_length=72, covariate_cols=["iob"]
            )
        ]
        assert midnight_jan3 not in anchors
