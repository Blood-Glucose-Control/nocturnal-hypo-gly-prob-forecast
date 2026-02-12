"""
Tests for gap handling utilities (interpolation + segmentation).

Run:
pytest tests/data/data_cleaning/test_gap_handling.py -v -s
"""

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessing.gap_handling import (
    segment_all_patients,
    _detect_interval,
    _find_nan_runs,
    _interpolate_small_gaps,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_series_df(values: list, freq_min: int = 5) -> pd.DataFrame:
    """Create a DataFrame from explicit bg_mM values with DatetimeIndex."""
    idx = pd.date_range("2024-01-01", periods=len(values), freq=f"{freq_min}min")
    df = pd.DataFrame({"bg_mM": values}, index=idx)
    df.index.name = "datetime"
    return df


# ---------------------------------------------------------------------------
# Core behavior — hand-constructed, every value traceable
# ---------------------------------------------------------------------------


class TestHandConstructedData:
    """Tests with explicit values so you can verify every number by hand."""

    def test_small_gap_filled_with_correct_values(self):
        """
        [4.0, NaN, NaN, NaN, 8.0]  threshold=3
        → [4.0, 5.0, 6.0, 7.0, 8.0]  (linear interpolation)
        """
        df = _make_series_df([4.0, np.nan, np.nan, np.nan, 8.0])
        result = _interpolate_small_gaps(df, max_gap_rows=3)
        np.testing.assert_array_almost_equal(
            result["bg_mM"].values, [4.0, 5.0, 6.0, 7.0, 8.0]
        )

    def test_small_gap_next_to_large_gap(self):
        """
        [1.0, NaN, 3.0, NaN, NaN, NaN, NaN, NaN, 9.0]  threshold=2
        Small gap (1 row): filled → 2.0
        Large gap (5 rows): untouched → stays NaN
        Segments: [1.0, 2.0, 3.0] and [9.0]
        """
        df = _make_series_df(
            [1.0, np.nan, 3.0, np.nan, np.nan, np.nan, np.nan, np.nan, 9.0]
        )
        # Check interpolation values
        interp = _interpolate_small_gaps(df, max_gap_rows=2)
        assert interp["bg_mM"].iloc[1] == pytest.approx(2.0)
        assert interp["bg_mM"].iloc[3:8].isna().all()

        # Check full pipeline segments
        result = segment_all_patients(
            {"p1": df}, imputation_threshold_mins=10, min_segment_length=1
        )
        assert len(result) == 2
        np.testing.assert_array_almost_equal(
            result["p1_seg_0"]["bg_mM"].values, [1.0, 2.0, 3.0]
        )
        np.testing.assert_array_almost_equal(result["p1_seg_1"]["bg_mM"].values, [9.0])

    def test_leading_and_trailing_nan_dropped(self):
        """
        [NaN, NaN, 5.0, 6.0, 7.0, NaN, NaN]  threshold=3
        Leading/trailing NaN have no anchor → can't interpolate → dropped
        Segment: [5.0, 6.0, 7.0]
        """
        df = _make_series_df([np.nan, np.nan, 5.0, 6.0, 7.0, np.nan, np.nan])
        result = segment_all_patients(
            {"p1": df}, imputation_threshold_mins=15, min_segment_length=1
        )
        assert len(result) == 1
        np.testing.assert_array_almost_equal(
            result["p1_seg_0"]["bg_mM"].values, [5.0, 6.0, 7.0]
        )

    def test_threshold_boundary_inclusive(self):
        """
        9-row gap at 5-min intervals = 45 min = exactly at threshold → interpolated
        10-row gap = 50 min = one above → segmented
        """
        # Exactly at threshold: interpolated, single segment
        df9 = _make_series_df([1.0] + [np.nan] * 9 + [11.0])
        result9 = segment_all_patients(
            {"p1": df9}, imputation_threshold_mins=45, min_segment_length=1
        )
        assert len(result9) == 1
        assert result9["p1_seg_0"]["bg_mM"].isna().sum() == 0

        # One above threshold: segmented into two
        df10 = _make_series_df([1.0] + [np.nan] * 10 + [12.0])
        result10 = segment_all_patients(
            {"p1": df10}, imputation_threshold_mins=45, min_segment_length=1
        )
        assert len(result10) == 2


# ---------------------------------------------------------------------------
# Edge cases & contracts
# ---------------------------------------------------------------------------


class TestEdgeCasesAndContracts:
    def test_empty_input_returns_empty(self):
        assert segment_all_patients({}) == {}

    def test_no_datetime_index_raises_value_error(self):
        df = pd.DataFrame({"bg_mM": [5.0, 5.5, 6.0]}, index=pd.RangeIndex(3))
        with pytest.raises(ValueError, match="DatetimeIndex"):
            segment_all_patients({"p1": df})

    def test_missing_bg_col_raises_value_error(self):
        idx = pd.date_range("2024-01-01", periods=5, freq="5min")
        df = pd.DataFrame({"other_col": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=idx)
        with pytest.raises(ValueError, match="missing required column"):
            segment_all_patients({"p1": df})

    def test_all_nan_patient_returns_no_segments(self):
        df = _make_series_df([np.nan] * 20)
        result = segment_all_patients({"p1": df}, min_segment_length=1)
        assert len(result) == 0

    def test_multiple_patients_independent(self):
        """
        p1: [1, 2, 3, 4, 5] — no gaps → 1 segment
        p2: [1, 2, 3, NaN×5, 9, 10, 11] — large gap → 2 segments
        """
        p1 = _make_series_df([1.0, 2.0, 3.0, 4.0, 5.0])
        p2 = _make_series_df(
            [1.0, 2.0, 3.0, np.nan, np.nan, np.nan, np.nan, np.nan, 9.0, 10.0, 11.0]
        )
        result = segment_all_patients(
            {"p1": p1, "p2": p2}, imputation_threshold_mins=10, min_segment_length=1
        )
        assert "p1_seg_0" in result
        assert "p2_seg_0" in result
        assert "p2_seg_1" in result
        assert len(result) == 3

    def test_non_bg_numeric_columns_not_interpolated(self):
        """
        bg_mM: [4.0, NaN, NaN, NaN, 8.0] — small gap, interpolated
        bolus: [1.0, NaN, NaN, NaN, 2.0] — should NOT be interpolated
        """
        idx = pd.date_range("2024-01-01", periods=5, freq="5min")
        df = pd.DataFrame(
            {
                "bg_mM": [4.0, np.nan, np.nan, np.nan, 8.0],
                "bolus": [1.0, np.nan, np.nan, np.nan, 2.0],
            },
            index=idx,
        )
        df.index.name = "datetime"
        result = _interpolate_small_gaps(df, max_gap_rows=3)
        # bg_mM filled
        assert result["bg_mM"].isna().sum() == 0
        # bolus untouched — still has 3 NaN
        assert result["bolus"].isna().sum() == 3

    def test_gaps_in_non_bg_columns_do_not_trigger_segmentation(self):
        """
        bg_mM: [5.0, 5.5, 6.0, 6.5, 7.0] — complete, no gaps
        dose_units: [1.0, NaN, NaN, NaN, 2.0] — large gap in event column
        food_g: [10.0, NaN, NaN, NaN, NaN] — trailing NaN in event column

        Should produce 1 segment (not 2+), since bg_mM is continuous.
        Event columns retain their NaN (no fractional interpolation).
        """
        idx = pd.date_range("2024-01-01", periods=20, freq="5min")
        df = pd.DataFrame(
            {
                "bg_mM": [
                    5.0,
                    5.5,
                    6.0,
                    6.5,
                    7.0,
                    8.0,
                    9.0,
                    10.0,
                    11.0,
                    12.0,
                    11.0,
                    10.0,
                    9.0,
                    8.0,
                    7.0,
                    6.0,
                    5.0,
                    4.0,
                    3.0,
                    2.0,
                ],
                "dose_units": [
                    1.0,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    2.0,
                ],
                "food_g": [
                    10.0,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                ],
            },
            index=idx,
        )
        df.index.name = "datetime"
        result = segment_all_patients(
            {"p1": df}, imputation_threshold_mins=10, min_segment_length=1
        )
        # Single segment — gaps in non-BG columns don't split
        assert len(result) == 1
        assert "p1_seg_0" in result
        # BG values preserved
        np.testing.assert_array_almost_equal(
            result["p1_seg_0"]["bg_mM"].values,
            [
                5.0,
                5.5,
                6.0,
                6.5,
                7.0,
                8.0,
                9.0,
                10.0,
                11.0,
                12.0,
                11.0,
                10.0,
                9.0,
                8.0,
                7.0,
                6.0,
                5.0,
                4.0,
                3.0,
                2.0,
            ],
        )
        # Event columns still have NaN (not interpolated to fractional values)
        assert result["p1_seg_0"]["dose_units"].isna().sum() == 18
        assert result["p1_seg_0"]["food_g"].isna().sum() == 19

    def test_segmentation_only_at_bg_gaps_not_event_gaps(self):
        """
        bg_mM: [1, 2, 3, NaN×5, 9, 10] — large gap triggers segmentation
        steps: [100, NaN×8, 1000] — gap spans both segments

        Should produce 2 segments based on bg_mM gap only.
        steps column NaN preserved in both segments.
        """
        idx = pd.date_range("2024-01-01", periods=10, freq="5min")
        df = pd.DataFrame(
            {
                "bg_mM": [
                    1.0,
                    2.0,
                    3.0,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    9.0,
                    10.0,
                ],
                "steps": [
                    100.0,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    1000.0,
                ],
            },
            index=idx,
        )
        df.index.name = "datetime"
        result = segment_all_patients(
            {"p1": df}, imputation_threshold_mins=10, min_segment_length=1
        )
        # Two segments from bg_mM gap
        assert len(result) == 2
        # First segment: bg values [1, 2, 3], steps has 2 NaN
        np.testing.assert_array_almost_equal(
            result["p1_seg_0"]["bg_mM"].values, [1.0, 2.0, 3.0]
        )
        assert result["p1_seg_0"]["steps"].iloc[0] == 100.0
        assert result["p1_seg_0"]["steps"].isna().sum() == 2
        # Second segment: bg values [9, 10], steps has 1 NaN
        np.testing.assert_array_almost_equal(
            result["p1_seg_1"]["bg_mM"].values, [9.0, 10.0]
        )
        assert result["p1_seg_1"]["steps"].iloc[-1] == 1000.0
        assert result["p1_seg_1"]["steps"].isna().sum() == 1

    def test_input_data_not_mutated(self):
        df = _make_series_df([1.0, np.nan, np.nan, 4.0, 5.0])
        df_copy = df.copy()
        input_dict = {"p1": df}

        segment_all_patients(input_dict, min_segment_length=1)

        assert "p1" in input_dict
        pd.testing.assert_frame_equal(input_dict["p1"], df_copy)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class TestDetectInterval:
    def test_detects_5_min_interval(self):
        idx = pd.date_range("2024-01-01", periods=10, freq="5min")
        df = pd.DataFrame({"bg_mM": range(10)}, index=idx)
        assert _detect_interval(df) == 5

    def test_detects_15_min_interval(self):
        idx = pd.date_range("2024-01-01", periods=100, freq="15min")
        df = pd.DataFrame({"bg_mM": range(100)}, index=idx)
        assert _detect_interval(df) == 15

    def test_single_row_falls_back_to_default(self):
        idx = pd.date_range("2024-01-01", periods=1, freq="5min")
        df = pd.DataFrame({"bg_mM": [5.0]}, index=idx)
        assert _detect_interval(df) == 5


class TestFindNanRuns:
    def test_no_nans(self):
        assert _find_nan_runs(pd.Series([1.0, 2.0, 3.0])) == []

    def test_single_nan_run(self):
        runs = _find_nan_runs(pd.Series([1.0, np.nan, np.nan, np.nan, 2.0]))
        assert len(runs) == 1
        start, end, length = runs[0]
        assert (start, end, length) == (1, 4, 3)

    def test_multiple_nan_runs(self):
        runs = _find_nan_runs(pd.Series([1.0, np.nan, 2.0, np.nan, np.nan, 3.0]))
        assert len(runs) == 2
        assert runs[0][2] == 1  # first run length
        assert runs[1][2] == 2  # second run length

    def test_all_nan(self):
        runs = _find_nan_runs(pd.Series([np.nan, np.nan, np.nan]))
        assert len(runs) == 1
        assert runs[0][2] == 3
