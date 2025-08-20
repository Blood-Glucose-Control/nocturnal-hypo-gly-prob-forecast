"""
Run:
pytest tests/data/data_cleaning/test_sampling.py -v -s
"""

import pandas as pd
import pytest
from src.data.preprocessing.sampling import ensure_regular_time_intervals


class TestEnsureRegularTimeIntervals:
    """Test class for ensure_regular_time_intervals function."""

    @pytest.fixture
    def sample_regular_data(self):
        """Create sample data with regular 5-minute intervals."""
        # Create regular 5-minute intervals
        datetime_index = pd.date_range(
            start="2024-01-01 00:00:00", periods=12, freq="5min"
        )

        data = {
            "p_num": ["patient_01"] * 12,
            "bg_mM": [5.0, 5.2, 5.1, 4.9, 5.3, 5.0, 4.8, 5.1, 5.2, 4.9, 5.0, 5.1],
            "hr_bpm": [70, 72, 71, 69, 73, 70, 68, 71, 72, 69, 70, 71],
        }

        df = pd.DataFrame(data, index=datetime_index)
        df.index.name = "datetime"
        return df

    @pytest.fixture
    def sample_irregular_data(self):
        """Create sample data with missing time intervals."""
        # Create irregular intervals (missing some 5-minute periods)
        datetime_list = [
            "2024-01-01 00:00:00",
            "2024-01-01 00:05:00",
            # Missing 00:10:00
            "2024-01-01 00:15:00",
            "2024-01-01 00:20:00",
            # Missing 00:25:00 and 00:30:00
            "2024-01-01 00:35:00",
            "2024-01-01 00:40:00",
        ]

        datetime_index = pd.to_datetime(datetime_list)

        data = {
            "p_num": ["patient_01"] * 6,
            "bg_mM": [5.0, 5.2, 5.1, 4.9, 5.3, 5.0],
            "hr_bpm": [70, 72, 71, 69, 73, 70],
        }

        df = pd.DataFrame(data, index=datetime_index)
        df.index.name = "datetime"
        return df

    @pytest.fixture
    def shifted_sample_irregular_data(self):
        """Create sample data with missing time intervals."""
        # Create irregular intervals (missing some 15-minute periods)
        datetime_list = [
            "2024-01-01 00:00:00",
            "2024-01-01 00:15:00",
            # Shifted 00:30:00 to 00:40:00
            "2024-01-01 00:40:00",
            "2024-01-01 00:55:00",
            # Missing 00:55:00 to 04:55:00
            "2024-01-01 04:55:00",
            "2024-01-01 05:10:00",
        ]

        datetime_index = pd.to_datetime(datetime_list)

        data = {
            "p_num": ["patient_01"] * 6,
            "bg_mM": [5.0, 5.2, 5.1, 4.9, 5.3, 5.0],
            "hr_bpm": [70, 72, 71, 69, 73, 70],
        }

        df = pd.DataFrame(data, index=datetime_index)
        df.index.name = "datetime"
        return df

    # @pytest.fixture
    # def multi_patient_data(self):
    #     """Create sample data with multiple patients and irregular intervals."""
    #     # Patient 1 data - missing some intervals
    #     p1_times = [
    #         "2024-01-01 00:00:00",
    #         "2024-01-01 00:05:00",
    #         # Missing 00:10:00
    #         "2024-01-01 00:15:00",
    #         "2024-01-01 00:20:00",
    #     ]

    #     # Patient 2 data - make 5min clearly most common
    #     p2_times = [
    #         "2024-01-01 00:00:00",
    #         "2024-01-01 00:05:00",
    #         # Missing 00:10:00
    #         # Missing 00:15:00
    #         "2024-01-01 00:20:00",
    #         "2024-01-01 00:25:00",
    #     ]

    #     p1_data = {
    #         "p_num": ["patient_01"] * 4,
    #         "id": [f"patient_01_{i}" for i in range(4)],
    #         "bg_mM": [5.0, 5.2, 5.1, 4.9],
    #         "hr_bpm": [70, 72, 71, 69],
    #     }

    #     p2_data = {
    #         "p_num": ["patient_02"] * 4,  # Updated count
    #         "id": [f"patient_02_{i}" for i in range(4)],
    #         "bg_mM": [4.8, 5.3, 5.0, 4.9],  # Added one more value
    #         "hr_bpm": [68, 73, 70, 69],  # Added one more value
    #     }

    #     p1_df = pd.DataFrame(p1_data, index=pd.to_datetime(p1_times))
    #     p2_df = pd.DataFrame(p2_data, index=pd.to_datetime(p2_times))

    #     combined_df = pd.concat([p1_df, p2_df])
    #     combined_df.index.name = "datetime"
    #     return combined_df

    def test_basic_functionality_regular_data(self, sample_regular_data):
        """Test function works correctly with already regular data."""
        result = ensure_regular_time_intervals(sample_regular_data)

        # Should return data with same number of rows for regular data
        assert len(result) == len(sample_regular_data)
        assert isinstance(result, pd.DataFrame)
        assert isinstance(result.index, pd.DatetimeIndex)

        # Check that all original data is preserved
        assert all(result["p_num"] == "patient_01")
        assert len(result["bg_mM"].dropna()) == len(sample_regular_data)

    def test_basic_functionality_irregular_data(self, sample_irregular_data):
        """Test function correctly fills missing time intervals."""
        result = ensure_regular_time_intervals(sample_irregular_data)

        # Should have more rows due to filled intervals
        assert len(result) > len(sample_irregular_data)

        # Check that regular 5-minute intervals are created
        time_diffs = result.index.to_series().diff().dropna()
        expected_interval = pd.Timedelta(minutes=5)
        assert all(diff == expected_interval for diff in time_diffs)

        # Check that original non-NaN values are preserved
        non_nan_bg = result["bg_mM"].dropna()
        original_bg = sample_irregular_data["bg_mM"]
        assert len(non_nan_bg) == len(original_bg)

        # Patient ID should be filled for all rows
        assert all(result["p_num"] == "patient_01")
        assert result["p_num"].isna().sum() == 0

    def test_basic_functionality_shifted_irregular_data(
        self, shifted_sample_irregular_data
    ):
        """Test function correctly fills missing time intervals."""
        result = ensure_regular_time_intervals(shifted_sample_irregular_data)

        # Should have more rows due to filled intervals
        assert len(result) > len(shifted_sample_irregular_data)

        # Check that regular 15-minute intervals are created
        time_diffs = result.index.to_series().diff().dropna()
        expected_interval = pd.Timedelta(minutes=15)
        assert all(diff == expected_interval for diff in time_diffs)

        # Check that original non-NaN values are preserved
        non_nan_bg = result["bg_mM"].dropna()
        original_bg = shifted_sample_irregular_data["bg_mM"]
        assert len(non_nan_bg) == len(original_bg)

        # Patient ID should be filled for all rows
        assert all(result["p_num"] == "patient_01")
        assert result["p_num"].isna().sum() == 0

    def test_missing_intervals_filled_with_nan(self, sample_irregular_data):
        """Test that missing time intervals are filled with NaN values."""
        result = ensure_regular_time_intervals(sample_irregular_data)

        # Count NaN values in the result
        nan_count_bg = result["bg_mM"].isna().sum()
        nan_count_hr = result["hr_bpm"].isna().sum()

        # Should have NaN values for missing intervals
        assert nan_count_bg > 0
        assert nan_count_hr > 0

        # NaN counts should be equal across numeric columns
        assert nan_count_bg == nan_count_hr

    # def test_multiple_patients(self, multi_patient_data):
    #     """Test function works correctly with multiple patients."""
    #     result = ensure_regular_time_intervals(multi_patient_data)

    #     # Check that both patients are present
    #     patients = result["p_num"].unique()
    #     assert "patient_01" in patients
    #     assert "patient_02" in patients

    #     # Check each patient separately
    #     for patient_id in patients:
    #         patient_data = result[result["p_num"] == patient_id].sort_index()

    #         # Check regular intervals for each patient
    #         time_diffs = patient_data.index.to_series().diff().dropna()
    #         expected_interval = pd.Timedelta(minutes=5)
    #         assert all(diff == expected_interval for diff in time_diffs)

    def test_empty_dataframe(self):
        """Test function handles empty DataFrame gracefully."""
        # Create empty DataFrame with proper structure
        empty_df = pd.DataFrame(columns=["p_num", "id", "bg_mM", "hr_bpm"])
        empty_df.index = pd.DatetimeIndex([])
        empty_df.index.name = "datetime"

        result = ensure_regular_time_intervals(empty_df)

        # Should return empty DataFrame with same structure
        assert len(result) == 0
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["p_num", "id", "bg_mM", "hr_bpm"]

    def test_data_types_preserved(self, sample_irregular_data):
        """Test that data types are preserved in the result where possible."""
        result = ensure_regular_time_intervals(sample_irregular_data)

        # Check that string/object data types match original
        assert result["p_num"].dtype == sample_irregular_data["p_num"].dtype

        # Check that float columns preserve their type
        assert result["bg_mM"].dtype == sample_irregular_data["bg_mM"].dtype

        # For integer columns that may contain NaN after filling intervals,
        # pandas converts them to float64 - this is expected behavior
        if result["hr_bpm"].isna().any():
            # If there are NaN values, expect conversion to float64
            assert result["hr_bpm"].dtype == "float64"
        else:
            # If no NaN values, original type should be preserved
            assert result["hr_bpm"].dtype == sample_irregular_data["hr_bpm"].dtype

        assert isinstance(result.index, pd.DatetimeIndex)

    def test_original_dataframe_not_modified(self, sample_irregular_data):
        """Test that the original DataFrame is not modified."""
        original_length = len(sample_irregular_data)
        original_bg_values = sample_irregular_data["bg_mM"].copy()

        result = ensure_regular_time_intervals(sample_irregular_data)

        # Original DataFrame should be unchanged
        assert len(sample_irregular_data) == original_length
        assert sample_irregular_data["bg_mM"].equals(original_bg_values)

        # Result should be different
        assert len(result) != len(sample_irregular_data)

    def test_datetime_index_name_preserved(self, sample_irregular_data):
        """Test that datetime index name is preserved."""
        result = ensure_regular_time_intervals(sample_irregular_data)

        assert result.index.name == "datetime"
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_different_time_intervals(self):
        """Test function works with different time intervals (not just 5 minutes)."""
        # Create data with 10-minute intervals
        datetime_list = [
            "2024-01-01 00:00:00",
            "2024-01-01 00:10:00",
            # Missing 00:20:00
            "2024-01-01 00:30:00",
        ]

        datetime_index = pd.to_datetime(datetime_list)

        data = {
            "p_num": ["patient_01"] * 3,
            "id": [f"patient_01_{i}" for i in range(3)],
            "bg_mM": [5.0, 5.2, 5.1],
            "hr_bpm": [70, 72, 71],
        }

        df = pd.DataFrame(data, index=datetime_index)
        df.index.name = "datetime"

        result = ensure_regular_time_intervals(df)

        # Should fill missing 10-minute interval
        assert len(result) == 4  # Original 3 + 1 missing

        # Check 10-minute intervals
        time_diffs = result.index.to_series().diff().dropna()
        expected_interval = pd.Timedelta(minutes=10)
        assert all(diff == expected_interval for diff in time_diffs)
