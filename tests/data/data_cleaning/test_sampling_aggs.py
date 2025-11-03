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
