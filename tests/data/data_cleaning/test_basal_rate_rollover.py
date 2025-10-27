"""
Run:
pytest tests/data/data_cleaning/test_basal_rate_rollover.py -v -s
"""

import pandas as pd
import pytest
import numpy as np

from src.data.preprocessing.feature_engineering import rollover_basal_rate
from src.data.models import ColumnNames


# TODO: Test this
class TestRolloverBasalRate:
    """Test class for rollover_basal_rate function."""

    @pytest.fixture
    def sample_data_with_rate(self):
        """Create sample data with basal rate set at certain intervals."""
        # Create regular 5-minute intervals
        datetime_index = pd.date_range(
            start="2024-01-01 00:00:00", periods=18, freq="5min"
        )

        data = {
            "p_num": ["patient_01"] * 18,
            "bg_mM": [5.0] * 18,
            "dose_units": [0.0] * 18,
            "food_g": [0.0] * 18,
        }

        df = pd.DataFrame(data, index=datetime_index)
        df.index.name = "datetime"

        # Set basal rate at specific times
        df.loc["2024-01-01 00:00:00", ColumnNames.RATE.value] = 1.2  # 1.2 units/hr
        df.loc["2024-01-01 01:00:00", ColumnNames.RATE.value] = 0.8  # 0.8 units/hr

        return df

    @pytest.fixture
    def sample_data_with_bolus(self):
        """Create sample data with existing bolus doses."""
        datetime_index = pd.date_range(
            start="2024-01-01 00:00:00", periods=15, freq="5min"
        )

        data = {
            "p_num": ["patient_01"] * 15,
            "bg_mM": [5.0] * 15,
            "dose_units": [0.0] * 15,
            "food_g": [0.0] * 15,
        }

        df = pd.DataFrame(data, index=datetime_index)
        df.index.name = "datetime"

        # Set existing bolus doses
        df.loc["2024-01-01 00:00:00", "dose_units"] = 5.0
        df.loc["2024-01-01 00:05:00", "dose_units"] = 0.0
        df.loc["2024-01-01 00:10:00", "dose_units"] = 3.0

        # Set basal rate
        df.loc["2024-01-01 00:00:00", ColumnNames.RATE.value] = 1.2  # 1.2 units/hr

        return df

    @pytest.fixture
    def sample_data_15min_intervals(self):
        """Create sample data with 15-minute intervals."""
        datetime_index = pd.date_range(
            start="2024-01-01 00:00:00", periods=8, freq="15min"
        )

        data = {
            "p_num": ["patient_01"] * 8,
            "bg_mM": [5.0] * 8,
            "dose_units": [0.0] * 8,
            "food_g": [0.0] * 8,
        }

        df = pd.DataFrame(data, index=datetime_index)
        df.index.name = "datetime"

        # Set basal rate (with 15-min intervals, rows_per_hour = 4)
        df.loc["2024-01-01 00:00:00", ColumnNames.RATE.value] = 1.2  # 1.2 units/hr

        return df

    def test_basic_rollover_functionality(self, sample_data_with_rate):
        """Test that basal rate gets rolled over to next rows."""
        result = rollover_basal_rate(sample_data_with_rate.copy())

        # Check that dose_units has been added
        assert (result["dose_units"] > 0).any(), "Should have added dose_units"

        # With 5-minute intervals, rate of 1.2 units/hr should add 1.2/12 = 0.1 units per row
        # For 12 rows (1 hour), total should be 1.2 units
        nonzero_doses = result[result["dose_units"] > 0]
        assert len(nonzero_doses) >= 12, "Should have added dose to at least 12 rows"

    def test_correct_dose_per_row(self, sample_data_with_rate):
        """Test that dose per row is calculated correctly."""
        result = rollover_basal_rate(sample_data_with_rate.copy())

        # With 5-minute intervals and rate of 1.2 units/hr
        # Expected dose per row: 1.2 / 12 = 0.1 units
        expected_dose_per_row = 1.2 / 12  # 0.1

        # Check rows that should have received the dose
        affected_rows = result["2024-01-01 00:05:00":"2024-01-01 00:55:00"]

        # All these rows should have received the basal dose
        for idx, row in affected_rows.iterrows():
            assert row["dose_units"] == pytest.approx(
                expected_dose_per_row, rel=1e-9
            ), f"Row {idx} should have dose of {expected_dose_per_row} units"

    def test_rate_change_handling(self, sample_data_with_rate):
        """Test that rate changes are handled correctly."""
        result = rollover_basal_rate(sample_data_with_rate.copy())

        # First rate: 1.2 units/hr at 00:00, applies to next 12 rows (00:05 to 00:55)
        # Second rate: 0.8 units/hr at 01:00, applies to next 12 rows (01:05 onwards)

        # Check first period
        first_period = result["2024-01-01 00:05:00":"2024-01-01 00:55:00"]
        expected_dose_1 = 1.2 / 12  # 0.1
        for idx, row in first_period.iterrows():
            assert row["dose_units"] == pytest.approx(expected_dose_1, rel=1e-9)

        # Check second period
        second_period = result["2024-01-01 01:05:00":]
        expected_dose_2 = 0.8 / 12  # ~0.067
        for idx, row in second_period.iterrows():
            assert row["dose_units"] == pytest.approx(expected_dose_2, rel=1e-9)

    def test_adds_to_existing_dose_units(self, sample_data_with_bolus):
        """Test that basal dose is added to existing bolus doses."""
        result = rollover_basal_rate(sample_data_with_bolus.copy())

        # Row with existing bolus dose
        row_00_00 = result.loc["2024-01-01 00:00:00"]
        assert row_00_00["dose_units"] == 5.0, "Original bolus should remain"

        # Row without existing dose should get basal dose
        expected_dose = 1.2 / 12  # 0.1
        row_00_05 = result.loc["2024-01-01 00:05:00"]
        assert row_00_05["dose_units"] == pytest.approx(expected_dose, rel=1e-9)

        # Row with existing dose should have both added
        row_00_10 = result.loc["2024-01-01 00:10:00"]
        assert row_00_10["dose_units"] == pytest.approx(3.0 + expected_dose, rel=1e-9)

    def test_different_time_intervals(self, sample_data_15min_intervals):
        """Test that function works with different time intervals (15 minutes)."""
        result = rollover_basal_rate(sample_data_15min_intervals.copy())

        # With 15-minute intervals, rows_per_hour = 4
        # Rate of 1.2 units/hr should add 1.2/4 = 0.3 units per row
        expected_dose_per_row = 1.2 / 4  # 0.3

        # Check that dose is applied correctly
        affected_rows = result["2024-01-01 00:15:00":"2024-01-01 00:45:00"]
        for idx, row in affected_rows.iterrows():
            assert row["dose_units"] == pytest.approx(expected_dose_per_row, rel=1e-9)

    def test_no_rate_column(self):
        """Test that function handles missing RATE column gracefully."""
        df = pd.DataFrame(
            {
                "p_num": ["patient_01"],
                "bg_mM": [5.0],
                "dose_units": [0.0],
            },
            index=pd.date_range("2024-01-01", periods=1),
        )

        with pytest.warns(UserWarning, match="No.*RATE.*column"):
            result = rollover_basal_rate(df)

        # Should return original dataframe unchanged
        assert len(result) == len(df)
        assert result.equals(df)

    def test_no_dose_units_column(self):
        """Test that function raises error when DOSE_UNITS column is missing."""
        df = pd.DataFrame(
            {
                "p_num": ["patient_01"],
                "bg_mM": [5.0],
                ColumnNames.RATE.value: [1.2],
            },
            index=pd.date_range("2024-01-01", periods=1),
        )

        with pytest.raises(ValueError, match="DOSE_UNITS"):
            rollover_basal_rate(df)

    def test_rate_with_nan_values(self):
        """Test that function handles NaN values in rate column."""
        datetime_index = pd.date_range(
            start="2024-01-01 00:00:00", periods=6, freq="5min"
        )

        data = {
            "p_num": ["patient_01"] * 6,
            "bg_mM": [5.0] * 6,
            "dose_units": [0.0] * 6,
            ColumnNames.RATE.value: [1.2, np.nan, np.nan, np.nan, np.nan, np.nan],
        }

        df = pd.DataFrame(data, index=datetime_index)
        df.index.name = "datetime"

        result = rollover_basal_rate(df)

        # NaN should be treated as no rate change
        # Only rows after the non-nan rate should get dose
        assert result.loc["2024-01-01 00:00:00", "dose_units"] == 0.0
        assert result.loc["2024-01-01 00:05:00", "dose_units"] == pytest.approx(
            0.1, rel=1e-9
        )

    def test_original_dataframe_not_modified(self, sample_data_with_rate):
        """Test that the original DataFrame is not modified."""
        original_dose_units = sample_data_with_rate["dose_units"].copy()

        result = rollover_basal_rate(sample_data_with_rate)

        # Original DataFrame should be unchanged
        assert sample_data_with_rate["dose_units"].equals(original_dose_units)

        # Result should be different
        assert not result["dose_units"].equals(original_dose_units)

    def test_empty_dataframe(self):
        """Test that function handles empty DataFrame gracefully."""
        # Create empty DataFrame with proper structure
        empty_df = pd.DataFrame(
            columns=["p_num", "bg_mM", "dose_units", ColumnNames.RATE.value]
        )
        empty_df.index = pd.DatetimeIndex([])
        empty_df.index.name = "datetime"

        # Should not raise error
        result = rollover_basal_rate(empty_df)

        # Should return empty DataFrame
        assert len(result) == 0

    def test_rate_zero(self):
        """Test that rate of 0 is handled correctly."""
        datetime_index = pd.date_range(
            start="2024-01-01 00:00:00", periods=15, freq="5min"
        )

        data = {
            "p_num": ["patient_01"] * 15,
            "bg_mM": [5.0] * 15,
            "dose_units": [0.0] * 15,
            ColumnNames.RATE.value: [0.0] * 15,
        }

        df = pd.DataFrame(data, index=datetime_index)
        df.index.name = "datetime"

        result = rollover_basal_rate(df)

        # All dose_units should remain 0
        assert all(result["dose_units"] == 0.0)

    def test_multiple_rate_changes(self):
        """Test handling of multiple rate changes in sequence."""
        datetime_index = pd.date_range(
            start="2024-01-01 00:00:00", periods=20, freq="5min"
        )

        data = {
            "p_num": ["patient_01"] * 20,
            "bg_mM": [5.0] * 20,
            "dose_units": [0.0] * 20,
        }

        df = pd.DataFrame(data, index=datetime_index)
        df.index.name = "datetime"

        # Set rates that change every 3 rows
        df.loc["2024-01-01 00:00:00", ColumnNames.RATE.value] = 1.2
        df.loc["2024-01-01 00:15:00", ColumnNames.RATE.value] = 0.8
        df.loc["2024-01-01 00:30:00", ColumnNames.RATE.value] = 1.0

        result = rollover_basal_rate(df)

        # Check that each rate applies to subsequent rows
        # First rate (1.2): applies to rows 1-12
        assert result.loc["2024-01-01 00:05:00", "dose_units"] == pytest.approx(
            0.1, rel=1e-9
        )
        assert result.loc["2024-01-01 00:10:00", "dose_units"] == pytest.approx(
            0.1, rel=1e-9
        )

        # Should not be affected yet by second rate
        assert result.loc["2024-01-01 00:20:00", "dose_units"] == pytest.approx(
            0.2 / 12, rel=1e-9
        )  # 0.8/12
