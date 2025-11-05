import pandas as pd
import numpy as np
import pytest

from src.data.preprocessing.sampling import (
    ensure_regular_time_intervals_with_aggregation,
)


class TestEnsureRegularTimeIntervalsWithAggregation:
    @pytest.fixture
    def sample_data_within_bins(self):
        # Regular 5-min grid intended: 00:00, 00:05, 00:10, 00:15
        # Provide multiple rows within the 00:05 window to trigger aggregation,
        # and no rows near 00:10 to create an empty bin after reindex.
        dt = pd.to_datetime(
            [
                "2024-01-01 00:00:00",  # -> rounds to 00:00
                "2024-01-01 00:05:00",  # -> rounds to 00:05
                "2024-01-01 00:06:00",  # -> rounds to 00:05 (aggregation target)
                # no time around 00:10 to leave an empty bin
                "2024-01-01 00:15:00",  # -> rounds to 00:15
            ]
        )

        df = pd.DataFrame(
            {
                "p_num": ["patient_01", "patient_01", "patient_01", "patient_01"],
                "bg_mM": [5.0, 6.0, 4.0, 7.0],  # BG should be averaged within bin
                "rate": [0.8, 0.6, 0.4, 1.0],  # rate should be averaged within bin
                "hr_bpm": [70, 72, 74, 69],  # other numeric should be summed within bin
                "food_g": [5, 10, 2, 0],  # summed
                "msg_type": ["A", "B", "C", "D"],  # categorical => first within bin
            },
            index=dt,
        )
        df.index.name = "datetime"
        return df

    def test_aggregation_rules_and_reindexing(self, sample_data_within_bins):
        result, freq = ensure_regular_time_intervals_with_aggregation(
            sample_data_within_bins
        )

        # Frequency detected should be 5 minutes
        assert freq == 5

        # Index should be regular from min to max at 5-min intervals
        expected_index = pd.date_range(
            "2024-01-01 00:00:00", "2024-01-01 00:15:00", freq="5min"
        )
        assert list(result.index) == list(expected_index)

        # Index name preserved as 'datetime'
        assert result.index.name == "datetime"

        # p_num should be filled for empty bins (e.g., 00:10) and match original
        assert result.loc["2024-01-01 00:10:00", "p_num"] == "patient_01"

        # Check aggregation at 00:05:00 (two rows mapped here: 00:05 and 00:06)
        bg_mean = np.mean([6.0, 4.0])
        rate_mean = np.mean([0.6, 0.4])
        hr_sum = np.mean([72, 74])
        food_sum = 10 + 2
        # msg_type takes the first within that bin, which should be the first occurrence among rows that map there (00:05 -> "B")
        assert np.isclose(result.loc["2024-01-01 00:05:00", "bg_mM"], bg_mean)
        assert np.isclose(result.loc["2024-01-01 00:05:00", "rate"], rate_mean)
        assert result.loc["2024-01-01 00:05:00", "hr_bpm"] == hr_sum
        assert result.loc["2024-01-01 00:05:00", "food_g"] == food_sum
        assert result.loc["2024-01-01 00:05:00", "msg_type"] == "B"

        # Check non-aggregated bins (single-row bins) are unchanged except dtype coercions
        assert result.loc["2024-01-01 00:00:00", "bg_mM"] == 5.0
        assert result.loc["2024-01-01 00:15:00", "bg_mM"] == 7.0

        # Empty bin (00:10) should exist, with NaNs in numeric/categorical except p_num
        row_0010 = result.loc["2024-01-01 00:10:00"]
        assert row_0010["p_num"] == "patient_01"
        assert row_0010["bg_mM"] != row_0010["bg_mM"]  # NaN
        assert row_0010["hr_bpm"] != row_0010["hr_bpm"]  # NaN
        assert row_0010["food_g"] != row_0010["food_g"]  # NaN
        assert row_0010["rate"] != row_0010["rate"]  # NaN
        assert row_0010["msg_type"] != row_0010["msg_type"]  # NaN

    def test_empty_dataframe(self):
        empty = pd.DataFrame(columns=["p_num", "bg_mM"])
        empty.index = pd.DatetimeIndex([])
        empty.index.name = "datetime"

        result, freq = ensure_regular_time_intervals_with_aggregation(empty)
        assert freq == 0
        assert result.empty
        assert list(result.columns) == ["p_num", "bg_mM"]

    def test_validation_errors(self):
        # Non-DatetimeIndex
        df = pd.DataFrame({"p_num": ["p1", "p1"], "bg_mM": [5.0, 6.0]})
        with pytest.raises(ValueError, match="datetime index"):
            ensure_regular_time_intervals_with_aggregation(df)

        # Missing p_num
        dt = pd.date_range("2024-01-01", periods=2, freq="5min")
        df = pd.DataFrame({"bg_mM": [5.0, 6.0]}, index=dt)
        df.index.name = "datetime"
        with pytest.raises(ValueError, match="p_num"):
            ensure_regular_time_intervals_with_aggregation(df)

        # Too few rows (<=1)
        df = pd.DataFrame({"p_num": ["p1"], "bg_mM": [5.0]}, index=dt[:1])
        df.index.name = "datetime"
        with pytest.raises(ValueError, match="more than 1 row"):
            ensure_regular_time_intervals_with_aggregation(df)

    def test_unaligned_timestamps_date_range_alignment(self):
        """
        Test that handles timestamps not aligned to the standard grid.

        This tests the bug fix where timestamps that round to a standard grid
        (e.g., 00:02:30 -> 00:00 for 5min freq) need the date_range to also
        be aligned to the rounded grid, otherwise reindexing produces all NaNs.
        """
        # Timestamps offset from standard 5-min grid
        # Intentionally not create middle timestamp like 02:30 which is between 00:00 and 00:05
        # Harder to debug due to rounding to the nearest and likely to happen irl anyway
        dt = pd.to_datetime(
            [
                "2024-01-01 00:02:25",  # -> rounds to 00:00
                "2024-01-01 00:07:25",  # -> rounds to 00:05
                "2024-01-01 00:08:00",  # -> rounds to 00:10
                "2024-01-01 00:13:00",  # -> rounds to 00:15
                "2024-01-01 00:17:25",  # -> rounds to 00:15
            ]
        )

        df = pd.DataFrame(
            {
                "p_num": [
                    "patient_02",
                    "patient_02",
                    "patient_02",
                    "patient_02",
                    "patient_02",
                ],
                "bg_mM": [5.0, 6.0, 4.0, 7.0, 8.0],
                "rate": [0.8, 0.6, 0.4, 1.0, 1.2],
                "hr_bpm": [70, 72, 74, 69, 71],
                "food_g": [5, 10, 2, 0, 3],
                "msg_type": ["A", "B", "C", "D", "E"],
            },
            index=dt,
        )
        df.index.name = "datetime"

        result, freq = ensure_regular_time_intervals_with_aggregation(df)

        # Frequency detected should be 5 minutes
        assert freq == 5

        # Verify we don't get all NaNs (the bug that was fixed)
        # At least some rows should have valid data
        assert (
            not result["bg_mM"].isna().all()
        ), "All bg_mM values are NaN - date range alignment bug!"

        # Index should be regular from rounded min to rounded max at 5-min intervals
        # Original min: 00:02:30 -> rounds to 00:00:00
        # Original max: 00:17:30 -> rounds to 00:15:00
        expected_index = pd.date_range(
            "2024-01-01 00:00:00", "2024-01-01 00:15:00", freq="5min"
        )
        assert list(result.index) == list(expected_index)

        # Verify data at rounded bins
        # 00:00:00 bin should have data from 00:02:30
        assert np.isclose(result.loc["2024-01-01 00:00:00", "bg_mM"], 5.0)
        assert result.loc["2024-01-01 00:00:00", "p_num"] == "patient_02"

        # 00:05:00 bin should have data from 00:07:25 only
        assert np.isclose(result.loc["2024-01-01 00:05:00", "bg_mM"], 6.0)
        assert result.loc["2024-01-01 00:05:00", "p_num"] == "patient_02"

        # 00:10:00 bin should have data from 00:08:00
        assert np.isclose(result.loc["2024-01-01 00:10:00", "bg_mM"], 4.0)
        assert result.loc["2024-01-01 00:10:00", "p_num"] == "patient_02"

        # 00:15:00 bin should have data aggregated from 00:13:00 and 00:17:25
        bg_mean_15 = np.mean([7.0, 8.0])
        rate_mean_15 = np.mean([1.0, 1.2])
        assert np.isclose(result.loc["2024-01-01 00:15:00", "bg_mM"], bg_mean_15)
        assert np.isclose(result.loc["2024-01-01 00:15:00", "rate"], rate_mean_15)
        assert result.loc["2024-01-01 00:15:00", "p_num"] == "patient_02"

    def test_edge_case_large_time_offset(self):
        """
        Test edge case with very large offset from standard grid.
        """
        # Timestamps with significant offset that will round to different grid points
        # This tests various edge cases: near boundaries, large gaps, multiple aggregations
        dt = pd.to_datetime(
            [
                "2024-01-01 00:03:59",  # -> floors to 04 (just under 4min)
                "2024-01-01 00:04:30",  # -> floors to 04 (aggregation target)
                "2024-01-01 00:07:15",  # -> floors to 08
                "2024-01-01 00:08:45",  # -> floors to 08 (aggregation target)
                "2024-01-01 00:09:30",  # -> floors to 10
                "2024-01-01 00:12:30",  # -> floors to 12
                "2024-01-01 00:14:01",  # -> floors to 14
                "2024-01-01 00:17:45",  # -> floors to 18
                "2024-01-01 00:19:30",  # -> floors to 20
                "2024-01-01 00:22:15",  # -> floors to 22
            ]
        )

        df = pd.DataFrame(
            {
                "p_num": ["patient_03"] * 10,
                "bg_mM": [5.0, 5.5, 6.0, 6.5, 4.0, 7.0, 7.5, 8.0, 8.5, 9.0],
                "rate": [0.8, 0.9, 0.6, 0.7, 0.4, 1.0, 1.1, 1.2, 1.3, 1.4],
                "hr_bpm": [70, 71, 72, 73, 74, 69, 70, 71, 72, 73],
                "food_g": [5, 2, 10, 3, 2, 0, 5, 3, 4, 2],
                "msg_type": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
            },
            index=dt,
        )
        df.index.name = "datetime"

        result, freq = ensure_regular_time_intervals_with_aggregation(df)

        assert freq == 2

        # Verify no all-NaN issue
        assert not result["bg_mM"].isna().all()

        # Index should span from rounded min (00:00) to rounded max (00:22)
        # Min: 00:03:59 -> rounds to 00:04:00
        # Max: 00:22:15 -> rounds to 00:22:00
        expected_index = pd.date_range(
            "2024-01-01 00:04:00", "2024-01-01 00:22:00", freq="2min"
        )
        assert list(result.index) == list(expected_index)

        # Verify data is correctly placed
        # 00:04:00 bin should have aggregated data from 00:03:59 and 00:04:30
        bg_mean_04 = np.mean([5.0, 5.5])
        assert np.isclose(result.loc["2024-01-01 00:04:00", "bg_mM"], bg_mean_04)

        # 00:08:00 bin should have aggregated data from 00:07:15 and 00:08:45
        bg_mean_08 = np.mean([6.0, 6.5])
        assert np.isclose(result.loc["2024-01-01 00:08:00", "bg_mM"], bg_mean_08)

        # 00:10:00 bin should have data from 00:09:30
        assert np.isclose(result.loc["2024-01-01 00:10:00", "bg_mM"], 4.0)

        # 00:12:00 bin should have data from 00:12:30
        assert np.isclose(result.loc["2024-01-01 00:12:00", "bg_mM"], 7.0)

        # 00:14:00 bin should have data from 00:14:01
        assert np.isclose(result.loc["2024-01-01 00:14:00", "bg_mM"], 7.5)

        # 00:18:00 bin should have data from 00:17:45
        assert np.isclose(result.loc["2024-01-01 00:18:00", "bg_mM"], 8.0)

        # 00:20:00 bin should have data from 00:19:30
        assert np.isclose(result.loc["2024-01-01 00:20:00", "bg_mM"], 8.5)

        # 00:22:00 bin should have data from 00:22:15
        assert np.isclose(result.loc["2024-01-01 00:22:00", "bg_mM"], 9.0)
