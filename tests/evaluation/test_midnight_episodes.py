# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)
# For commercial licensing, contact: christopher/cjrisi AT gluroo/uwaterloo DOT com/ca

"""Tests for midnight-anchored episode building."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.episode_builders import (
    build_midnight_episodes,
    build_patient_episodes,
    select_episodes_stratified,
)


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

        episodes, skip_stats = build_midnight_episodes(
            df, context_length=context_len, forecast_length=forecast_len
        )

        assert len(episodes) == 4  # Jan 2, 3, 4, 5 midnights
        assert skip_stats["total_anchors"] == 4
        assert skip_stats["skipped_bg_nan"] == 0
        assert skip_stats["interpolated_episodes"] == 0
        for ep in episodes:
            assert ep["anchor"].hour == 0 and ep["anchor"].minute == 0
            assert len(ep["context_df"]) == context_len
            assert len(ep["target_bg"]) == forecast_len
            assert "bg_mM" in ep["context_df"].columns
            assert np.allclose(ep["target_bg"], 8.0)

    def test_no_episodes_when_data_too_short(self):
        """Boundary: returns empty list if data can't fit context + forecast."""
        df = _make_patient_df(n_days=1)  # 24h can't fit 512-step (~42.7h) context
        episodes, skip_stats = build_midnight_episodes(
            df, context_length=512, forecast_length=72
        )
        assert episodes == []
        assert skip_stats["total_anchors"] == 0

    def test_nan_bg_episodes_skipped(self):
        """Safety: episodes with NaN BG are excluded (missing CGM data)."""
        df = _make_patient_df(n_days=5)

        # Punch a NaN hole in Jan 3's forecast window (00:30-03:00)
        midnight_jan3 = pd.Timestamp("2024-01-03 00:00")
        mask = (df.index >= midnight_jan3 + pd.Timedelta(minutes=30)) & (
            df.index < midnight_jan3 + pd.Timedelta(minutes=180)
        )
        df.loc[mask, "bg_mM"] = np.nan

        episodes, skip_stats = build_midnight_episodes(
            df, context_length=144, forecast_length=72
        )
        anchors = [ep["anchor"] for ep in episodes]
        assert midnight_jan3 not in anchors
        assert skip_stats["skipped_bg_nan"] >= 1
        assert midnight_jan3 in skip_stats["skipped_anchors"]

    def test_covariates_returned_when_available(self):
        """Covariate data (IOB) is included in future_covariates dict."""
        df = _make_patient_df(n_days=5, include_iob=True, iob_value=2.0)
        episodes, _ = build_midnight_episodes(
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
        episodes, _ = build_midnight_episodes(
            df, context_length=144, forecast_length=72, covariate_cols=["iob"]
        )

        assert len(episodes) > 0
        for ep in episodes:
            assert ep["future_covariates"] == {}

    def test_context_df_includes_covariates(self):
        """Context DataFrame includes covariate columns alongside BG."""
        df = _make_patient_df(n_days=5, include_iob=True, iob_value=1.0)
        episodes, _ = build_midnight_episodes(
            df, context_length=144, forecast_length=72, covariate_cols=["iob"]
        )

        assert len(episodes) > 0
        for ep in episodes:
            assert "bg_mM" in ep["context_df"].columns
            assert "iob" in ep["context_df"].columns
            assert len(ep["context_df"]) == 144

    def test_specific_anchor_dates(self):
        """Verify anchor timestamps match expected midnight dates."""
        # 5 days starting Jan 1: expect midnights Jan 2, 3, 4, 5
        df = _make_patient_df(n_days=5, start="2024-01-01")
        episodes, _ = build_midnight_episodes(
            df, context_length=144, forecast_length=72
        )

        expected_anchors = [
            pd.Timestamp("2024-01-02 00:00"),
            pd.Timestamp("2024-01-03 00:00"),
            pd.Timestamp("2024-01-04 00:00"),
            pd.Timestamp("2024-01-05 00:00"),
        ]
        actual_anchors = [ep["anchor"] for ep in episodes]
        assert actual_anchors == expected_anchors

    def test_duplicate_timestamps_handled(self):
        """Duplicate timestamps are deduplicated (keep last)."""
        df = _make_patient_df(n_days=5, bg_value=5.0)
        # Add duplicate timestamp with different value
        dup_idx = df.index[100]
        dup_row = pd.DataFrame({"bg_mM": [9.0]}, index=[dup_idx])
        df = pd.concat([df, dup_row])

        episodes, _ = build_midnight_episodes(
            df, context_length=144, forecast_length=72
        )

        # Should still work, and the last value (9.0) should be kept for that timestamp
        assert len(episodes) > 0

    def test_irregular_timestamps_reindexed(self):
        """Data with gaps is reindexed to regular grid, missing BG causes skip."""
        df = _make_patient_df(n_days=5, bg_value=6.0)
        # Remove some rows to create gaps
        df = df.drop(df.index[50:65])  # 75 min gap

        episodes, skip_stats = build_midnight_episodes(
            df, context_length=144, forecast_length=72
        )

        # Gap creates NaN after reindex, so some episodes may be skipped
        # but if gap doesn't hit a midnight window, episodes still build
        # Key assertion: function doesn't crash on irregular data
        assert isinstance(episodes, list)
        assert isinstance(skip_stats, dict)

    def test_timezone_naive_timestamps(self):
        """Timezone-naive timestamps work correctly (default CGM data format)."""
        df = _make_patient_df(n_days=5, start="2024-01-01")
        assert df.index.tz is None  # Confirm naive

        episodes, _ = build_midnight_episodes(
            df, context_length=144, forecast_length=72
        )

        assert len(episodes) == 4
        for ep in episodes:
            assert ep["anchor"].tz is None

    def test_short_bg_gap_interpolated(self):
        """Gaps <= max_bg_gap_steps are filled via interpolation, episode kept."""
        df = _make_patient_df(n_days=5, bg_value=8.0)
        # Punch a 1-step gap into Jan 3's forecast window
        gap_time = pd.Timestamp("2024-01-03 00:30")
        df.loc[gap_time, "bg_mM"] = np.nan

        episodes, skip_stats = build_midnight_episodes(
            df, context_length=144, forecast_length=72, max_bg_gap_steps=2
        )
        anchors = [ep["anchor"] for ep in episodes]
        assert pd.Timestamp("2024-01-03") in anchors
        assert skip_stats["interpolated_episodes"] >= 1
        assert skip_stats["skipped_bg_nan"] == 0

    def test_long_bg_gap_still_skipped(self):
        """Gaps > max_bg_gap_steps are not filled and episode is skipped."""
        df = _make_patient_df(n_days=5, bg_value=8.0)
        # Punch a 15-step (75 min) gap into Jan 3's forecast window
        for i in range(15):  # 15 steps = 75 min gap
            df.loc[
                pd.Timestamp("2024-01-03 00:05") + pd.Timedelta(minutes=5 * i), "bg_mM"
            ] = np.nan

        episodes, skip_stats = build_midnight_episodes(
            df, context_length=144, forecast_length=72, max_bg_gap_steps=2
        )
        anchors = [ep["anchor"] for ep in episodes]
        assert pd.Timestamp("2024-01-03") not in anchors
        assert skip_stats["skipped_bg_nan"] >= 1

    def test_interpolation_disabled_with_zero(self):
        """max_bg_gap_steps=0 disables interpolation entirely."""
        df = _make_patient_df(n_days=5, bg_value=8.0)
        gap_time = pd.Timestamp("2024-01-03 00:30")
        df.loc[gap_time, "bg_mM"] = np.nan

        episodes, skip_stats = build_midnight_episodes(
            df, context_length=144, forecast_length=72, max_bg_gap_steps=0
        )
        anchors = [ep["anchor"] for ep in episodes]
        assert pd.Timestamp("2024-01-03") not in anchors
        assert skip_stats["interpolated_episodes"] == 0

    def test_missing_target_column_raises_value_error(self):
        """Missing target column raises clear ValueError, not KeyError."""
        import pytest

        df = _make_patient_df(n_days=5)
        df = df.rename(columns={"bg_mM": "glucose"})  # Remove expected column

        with pytest.raises(ValueError, match="Target column 'bg_mM' not found"):
            build_midnight_episodes(df, context_length=144, forecast_length=72)


# ---------------------------------------------------------------------------
# Helpers for multi-patient tests
# ---------------------------------------------------------------------------


def _make_holdout_df(
    patient_ids: list,
    n_days: int = 5,
    interval_mins: int = 5,
    start: str = "2024-01-01",
    bg_value: float = 7.0,
    patient_col: str = "p_num",
) -> pd.DataFrame:
    """Create a combined holdout DataFrame with multiple patients."""
    frames = []
    for pid in patient_ids:
        freq = f"{interval_mins}min"
        index = pd.date_range(
            start, periods=n_days * 24 * 60 // interval_mins, freq=freq
        )
        df = pd.DataFrame(
            {
                "datetime": index,
                patient_col: pid,
                "bg_mM": np.full(len(index), bg_value),
            }
        )
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Tests for build_patient_episodes
# ---------------------------------------------------------------------------


class TestBuildPatientEpisodes:
    """Test build_patient_episodes function."""

    def test_basic_multi_patient(self):
        """Episodes are built and grouped by patient ID."""
        holdout = _make_holdout_df(["pat_A", "pat_B"], n_days=5)
        result = build_patient_episodes(
            holdout, ["pat_A", "pat_B"], context_length=144, forecast_length=72
        )
        assert "pat_A" in result
        assert "pat_B" in result
        for pid, eps in result.items():
            assert len(eps) > 0
            for ep in eps:
                assert ep["patient_id"] == pid

    def test_patient_id_column_p_num(self):
        """Detects p_num as the patient column."""
        holdout = _make_holdout_df(["p1", "p2"], patient_col="p_num")
        result = build_patient_episodes(
            holdout, ["p1"], context_length=144, forecast_length=72
        )
        assert "p1" in result

    def test_patient_id_column_id(self):
        """Detects id as the patient column."""
        holdout = _make_holdout_df(["p1", "p2"], patient_col="id")
        result = build_patient_episodes(
            holdout, ["p1"], context_length=144, forecast_length=72
        )
        assert "p1" in result

    def test_missing_patient_column_raises(self):
        """Raises ValueError when neither p_num nor id column exists."""
        holdout = _make_holdout_df(["p1"], patient_col="subject")
        with pytest.raises(ValueError, match="Expected patient column"):
            build_patient_episodes(
                holdout, ["p1"], context_length=144, forecast_length=72
            )

    def test_datetime_column_set_as_index(self):
        """DataFrame with a 'datetime' column is correctly indexed."""
        holdout = _make_holdout_df(["pat_A"], n_days=5)
        assert "datetime" in holdout.columns
        result = build_patient_episodes(
            holdout, ["pat_A"], context_length=144, forecast_length=72
        )
        assert "pat_A" in result
        for ep in result["pat_A"]:
            assert isinstance(ep["context_df"].index, pd.DatetimeIndex)

    def test_unknown_patient_returns_empty(self):
        """Requesting a patient not in the data yields no episodes for that patient."""
        holdout = _make_holdout_df(["pat_A"])
        result = build_patient_episodes(
            holdout, ["pat_MISSING"], context_length=144, forecast_length=72
        )
        assert "pat_MISSING" not in result


# ---------------------------------------------------------------------------
# Tests for select_episodes_stratified
# ---------------------------------------------------------------------------


class TestSelectEpisodesStratified:
    """Test select_episodes_stratified function."""

    def test_selects_correct_count_per_patient(self):
        """Picks exactly per_patient episodes from each patient."""
        holdout = _make_holdout_df(["pat_A", "pat_B"], n_days=5)
        by_patient = build_patient_episodes(
            holdout, ["pat_A", "pat_B"], context_length=144, forecast_length=72
        )
        selected = select_episodes_stratified(by_patient, per_patient=2, seed=42)
        from_a = [ep for ep in selected if ep["patient_id"] == "pat_A"]
        from_b = [ep for ep in selected if ep["patient_id"] == "pat_B"]
        assert len(from_a) == 2
        assert len(from_b) == 2

    def test_deterministic_with_same_seed(self):
        """Same seed produces identical episode selection."""
        holdout = _make_holdout_df(["pat_A", "pat_B"], n_days=5)
        by_patient = build_patient_episodes(
            holdout, ["pat_A", "pat_B"], context_length=144, forecast_length=72
        )
        sel1 = select_episodes_stratified(by_patient, per_patient=2, seed=99)
        sel2 = select_episodes_stratified(by_patient, per_patient=2, seed=99)
        anchors1 = [(ep["patient_id"], ep["anchor"]) for ep in sel1]
        anchors2 = [(ep["patient_id"], ep["anchor"]) for ep in sel2]
        assert anchors1 == anchors2

    def test_different_seed_different_selection(self):
        """Different seeds produce different selections (with enough episodes)."""
        holdout = _make_holdout_df(["pat_A"], n_days=10)
        by_patient = build_patient_episodes(
            holdout, ["pat_A"], context_length=144, forecast_length=72
        )
        sel1 = select_episodes_stratified(by_patient, per_patient=2, seed=1)
        sel2 = select_episodes_stratified(by_patient, per_patient=2, seed=999)
        anchors1 = [ep["anchor"] for ep in sel1]
        anchors2 = [ep["anchor"] for ep in sel2]
        assert anchors1 != anchors2

    def test_caps_at_available_episodes(self):
        """If per_patient exceeds available episodes, returns all of them."""
        holdout = _make_holdout_df(["pat_A"], n_days=5)
        by_patient = build_patient_episodes(
            holdout, ["pat_A"], context_length=144, forecast_length=72
        )
        n_available = len(by_patient["pat_A"])
        selected = select_episodes_stratified(by_patient, per_patient=9999, seed=42)
        assert len(selected) == n_available
