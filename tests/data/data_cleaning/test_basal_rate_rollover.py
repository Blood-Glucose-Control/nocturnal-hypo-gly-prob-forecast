"""
Run:
pytest tests/data/data_cleaning/test_basal_rate_rollover.py -v -s
"""

import pandas as pd
import pytest

from src.data.preprocessing.feature_engineering import rollover_basal_rate
from src.data.models import ColumnNames


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
            ColumnNames.BASAL_DURATION_MINS.value: [None] * 18,
        }

        df = pd.DataFrame(data, index=datetime_index)
        df.index.name = "datetime"

        # Set basal rate at specific times with duration
        df.loc["2024-01-01 00:00:00", ColumnNames.RATE.value] = 1.2  # 1.2 units/hr
        df.loc["2024-01-01 00:00:00", ColumnNames.BASAL_DURATION_MINS.value] = (
            60  # 1 hour
        )

        df.loc["2024-01-01 01:00:00", ColumnNames.RATE.value] = 0.8  # 0.8 units/hr
        df.loc["2024-01-01 01:00:00", ColumnNames.BASAL_DURATION_MINS.value] = (
            30  # 30 mins
        )

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
            ColumnNames.BASAL_DURATION_MINS.value: [None] * 15,
        }

        df = pd.DataFrame(data, index=datetime_index)
        df.index.name = "datetime"

        # Set existing bolus doses
        df.loc["2024-01-01 00:00:00", "dose_units"] = 5.0
        df.loc["2024-01-01 00:05:00", "dose_units"] = 0.0
        df.loc["2024-01-01 00:10:00", "dose_units"] = 3.0

        # Set basal rate
        df.loc["2024-01-01 00:00:00", ColumnNames.RATE.value] = 1.2  # 1.2 units/hr
        df.loc["2024-01-01 00:00:00", ColumnNames.BASAL_DURATION_MINS.value] = (
            60  # 1 hour
        )

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
            ColumnNames.BASAL_DURATION_MINS.value: [None] * 8,
        }

        df = pd.DataFrame(data, index=datetime_index)
        df.index.name = "datetime"

        # Set basal rate (with 15-min intervals, rows_per_hour = 4)
        df.loc["2024-01-01 00:00:00", ColumnNames.RATE.value] = 1.2  # 1.2 units/hr
        df.loc["2024-01-01 00:00:00", ColumnNames.BASAL_DURATION_MINS.value] = (
            60  # 1 hour
        )

        return df

    def test_basic_rollover_functionality(self, sample_data_with_rate):
        """Test that basal rate gets rolled over to next rows."""
        result = rollover_basal_rate(sample_data_with_rate.copy(), delivery_type="temp")

        # Check that dose_units has been added
        assert (result["dose_units"] > 0).any(), "Should have added dose_units"

        # With 5-minute intervals, rate of 1.2 units/hr should add 1.2/12 = 0.1 units per row
        # For 12 rows (1 hour), total should be 1.2 units
        nonzero_doses = result[result["dose_units"] > 0]
        assert len(nonzero_doses) >= 12, "Should have added dose to at least 12 rows"

    def test_correct_dose_per_row(self, sample_data_with_rate):
        """Test that dose per row is calculated correctly."""
        result = rollover_basal_rate(sample_data_with_rate.copy(), delivery_type="temp")

        # With 5-minute intervals and rate of 1.2 units/hr
        # Expected dose per row: 1.2 / 12 = 0.1 units
        expected_dose_per_row = 1.2 / 12  # 0.1

        # Check rows that should have received the dose (first 60 mins / 5-min intervals = 12 rows)
        affected_rows = result["2024-01-01 00:05:00":"2024-01-01 00:55:00"]

        # All these rows should have received the basal dose
        for idx, row in affected_rows.iterrows():
            assert row["dose_units"] == pytest.approx(
                expected_dose_per_row, rel=1e-9
            ), f"Row {idx} should have dose of {expected_dose_per_row} units"

    def test_rate_change_handling(self, sample_data_with_rate):
        """Test that rate changes are handled correctly."""
        result = rollover_basal_rate(sample_data_with_rate.copy(), delivery_type="temp")

        # First rate: 1.2 units/hr at 00:00 with 60 mins duration, applies to next 12 rows (00:05 to 00:55)
        # Second rate: 0.8 units/hr at 01:00 with 30 mins duration, applies to next 6 rows (01:05 onwards)

        # Check first period
        first_period = result["2024-01-01 00:05:00":"2024-01-01 00:55:00"]
        expected_dose_1 = 1.2 / 12  # 0.1
        for idx, row in first_period.iterrows():
            assert row["dose_units"] == pytest.approx(expected_dose_1, rel=1e-9)

        # Check second period (6 rows for 30 mins)
        second_period = result["2024-01-01 01:05:00":"2024-01-01 01:25:00"]
        expected_dose_2 = 0.8 / 12  # ~0.067
        for idx, row in second_period.iterrows():
            assert row["dose_units"] == pytest.approx(expected_dose_2, rel=1e-9)

    def test_adds_to_existing_dose_units(self, sample_data_with_bolus):
        """Test that basal dose is added to existing bolus doses."""
        result = rollover_basal_rate(
            sample_data_with_bolus.copy(), delivery_type="temp"
        )

        # Row with existing bolus dose
        row_00_00 = result.loc["2024-01-01 00:00:00"]
        assert row_00_00["dose_units"] == 5.1  # Original + 0.1

        # Row without existing dose should get basal dose
        expected_dose = 1.2 / 12  # 0.1
        row_00_05 = result.loc["2024-01-01 00:05:00"]
        assert row_00_05["dose_units"] == pytest.approx(expected_dose, rel=1e-9)

        # Row with existing dose should have both added
        row_00_10 = result.loc["2024-01-01 00:10:00"]
        assert row_00_10["dose_units"] == pytest.approx(3.0 + expected_dose, rel=1e-9)

    def test_different_time_intervals(self, sample_data_15min_intervals):
        """Test that function works with different time intervals (15 minutes)."""
        result = rollover_basal_rate(
            sample_data_15min_intervals.copy(), delivery_type="temp"
        )

        # With 15-minute intervals, rows_per_hour = 4
        # Rate of 1.2 units/hr should add 1.2/4 = 0.3 units per row
        # For 60 mins duration, that's 4 rows
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

        # When there's no rate column, delivery_type doesn't matter - function returns early
        result = rollover_basal_rate(df, delivery_type="temp")

        # Should return original dataframe unchanged
        assert len(result) == len(df)
        assert result.equals(df)

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
            ColumnNames.BASAL_DURATION_MINS.value: [0.0] * 15,
        }

        df = pd.DataFrame(data, index=datetime_index)
        df.index.name = "datetime"

        result = rollover_basal_rate(df, delivery_type="temp")

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
            ColumnNames.BASAL_DURATION_MINS.value: [None] * 20,
        }

        df = pd.DataFrame(data, index=datetime_index)
        df.index.name = "datetime"

        # Set rates that change every 3 rows
        df.loc["2024-01-01 00:00:00", ColumnNames.RATE.value] = 1.2
        df.loc["2024-01-01 00:00:00", ColumnNames.BASAL_DURATION_MINS.value] = 60

        df.loc["2024-01-01 00:15:00", ColumnNames.RATE.value] = 0.8
        df.loc["2024-01-01 00:15:00", ColumnNames.BASAL_DURATION_MINS.value] = 15

        df.loc["2024-01-01 00:30:00", ColumnNames.RATE.value] = 1.0
        df.loc["2024-01-01 00:30:00", ColumnNames.BASAL_DURATION_MINS.value] = 30

        result = rollover_basal_rate(df, delivery_type="temp")

        # Check that each rate applies to subsequent rows based on duration
        # First rate (1.2): applies to rows starting at 00:00 with 60 min duration
        assert result.loc["2024-01-01 00:00:00", "dose_units"] == pytest.approx(
            1.2 / 12, rel=1e-9
        )

        # Second rate (0.8): with 15 min duration (3 rows)
        assert result.loc["2024-01-01 00:15:00", "dose_units"] == pytest.approx(
            0.8 / 12, rel=1e-9
        )

        # Third rate (1.0): with 30 min duration (6 rows)
        assert result.loc["2024-01-01 00:30:00", "dose_units"] == pytest.approx(
            1.0 / 12, rel=1e-9
        )


class TestRolloverBasalAutomated:
    """Test class for automated basal delivery type (Control-IQ, Loop, OpenAPS)."""

    @pytest.fixture
    def sample_data_automated(self):
        """Create sample data for automated basal (no duration column needed)."""
        datetime_index = pd.date_range(
            start="2024-01-01 00:00:00", periods=18, freq="5min"
        )

        data = {
            "p_num": ["patient_01"] * 18,
            "bg_mM": [5.0] * 18,
            "dose_units": [0.0] * 18,
            "food_g": [0.0] * 18,
            ColumnNames.RATE.value: [None] * 18,  # Will set specific rates
        }

        df = pd.DataFrame(data, index=datetime_index)
        df.index.name = "datetime"
        return df

    def test_forward_fill_behavior(self, sample_data_automated):
        """Test that rate persists (forward-fills) until next rate change."""
        df = sample_data_automated.copy()

        # Set rate at row 0, should persist until row 6 where new rate is set
        df.loc["2024-01-01 00:00:00", ColumnNames.RATE.value] = 1.2  # 1.2 units/hr
        df.loc["2024-01-01 00:30:00", ColumnNames.RATE.value] = 0.8  # Change at row 6

        result = rollover_basal_rate(df, delivery_type="automated")

        # With 5-minute intervals, dose per row = rate / 12
        expected_dose_1 = 1.2 / 12  # 0.1 units per row
        expected_dose_2 = 0.8 / 12  # ~0.067 units per row

        # Rows 0-5 should have first rate applied
        for i in range(6):
            assert result.iloc[i]["dose_units"] == pytest.approx(
                expected_dose_1, rel=1e-9
            ), f"Row {i} should have dose {expected_dose_1}"

        # Rows 6-17 should have second rate applied (forward-filled)
        for i in range(6, 18):
            assert result.iloc[i]["dose_units"] == pytest.approx(
                expected_dose_2, rel=1e-9
            ), f"Row {i} should have dose {expected_dose_2}"

    def test_correct_dose_calculation_automated(self, sample_data_automated):
        """Test correct dose calculation per interval for automated delivery."""
        df = sample_data_automated.copy()

        # Set a single rate that should apply to all rows via forward-fill
        df.loc["2024-01-01 00:00:00", ColumnNames.RATE.value] = 2.4  # 2.4 units/hr

        result = rollover_basal_rate(df, delivery_type="automated")

        # With 5-minute intervals, rows_per_hour = 12
        # Dose per row = 2.4 / 12 = 0.2 units
        expected_dose = 2.4 / 12

        # All rows should have this dose (forward-filled from first row)
        for i in range(len(result)):
            assert result.iloc[i]["dose_units"] == pytest.approx(
                expected_dose, rel=1e-9
            ), f"Row {i} should have dose {expected_dose}"

    def test_leading_nan_rates(self, sample_data_automated):
        """Test handling of leading NaN rates before first rate event."""
        df = sample_data_automated.copy()

        # First rate doesn't appear until row 6 (30 minutes in)
        df.loc["2024-01-01 00:30:00", ColumnNames.RATE.value] = 1.0

        result = rollover_basal_rate(df, delivery_type="automated")

        expected_dose = 1.0 / 12

        # Rows 0-5 should have 0 dose (no rate to forward-fill from)
        for i in range(6):
            assert result.iloc[i]["dose_units"] == pytest.approx(
                0.0, rel=1e-9
            ), f"Row {i} should have 0 dose (before first rate)"

        # Rows 6-17 should have the rate applied
        for i in range(6, 18):
            assert result.iloc[i]["dose_units"] == pytest.approx(
                expected_dose, rel=1e-9
            ), f"Row {i} should have dose {expected_dose}"

    def test_trailing_rates_to_end(self, sample_data_automated):
        """Test that trailing rates persist to end of data."""
        df = sample_data_automated.copy()

        # Set rate early, should persist to end
        df.loc["2024-01-01 00:00:00", ColumnNames.RATE.value] = 0.6

        result = rollover_basal_rate(df, delivery_type="automated")

        expected_dose = 0.6 / 12

        # All 18 rows should have the dose applied
        for i in range(len(result)):
            assert result.iloc[i]["dose_units"] == pytest.approx(
                expected_dose, rel=1e-9
            ), f"Row {i} should have dose {expected_dose} via forward-fill"

    def test_multiple_rate_changes_automated(self, sample_data_automated):
        """Test multiple rate changes throughout the data for automated delivery."""
        df = sample_data_automated.copy()

        # Set multiple rate changes
        df.loc["2024-01-01 00:00:00", ColumnNames.RATE.value] = 1.2  # rows 0-3
        df.loc["2024-01-01 00:20:00", ColumnNames.RATE.value] = 0.6  # rows 4-9
        df.loc["2024-01-01 00:50:00", ColumnNames.RATE.value] = 1.8  # rows 10-17

        result = rollover_basal_rate(df, delivery_type="automated")

        # Check each segment
        expected_1 = 1.2 / 12
        expected_2 = 0.6 / 12
        expected_3 = 1.8 / 12

        # Rows 0-3 (00:00 to 00:15)
        for i in range(4):
            assert result.iloc[i]["dose_units"] == pytest.approx(
                expected_1, rel=1e-9
            ), f"Row {i} should have dose {expected_1}"

        # Rows 4-9 (00:20 to 00:45)
        for i in range(4, 10):
            assert result.iloc[i]["dose_units"] == pytest.approx(
                expected_2, rel=1e-9
            ), f"Row {i} should have dose {expected_2}"

        # Rows 10-17 (00:50 to end)
        for i in range(10, 18):
            assert result.iloc[i]["dose_units"] == pytest.approx(
                expected_3, rel=1e-9
            ), f"Row {i} should have dose {expected_3}"

    def test_all_nan_rate_column(self, sample_data_automated):
        """Test edge case: rate column exists but is all NaN."""
        df = sample_data_automated.copy()
        # Rate column already initialized with all None/NaN values

        result = rollover_basal_rate(df, delivery_type="automated")

        # All dose_units should remain 0 (no rates to apply)
        assert all(
            result["dose_units"] == 0.0
        ), "All dose_units should be 0 when rate column is all NaN"

    def test_adds_to_existing_bolus_automated(self, sample_data_automated):
        """Test that automated basal adds to existing bolus doses."""
        df = sample_data_automated.copy()

        # Set existing bolus
        df.loc["2024-01-01 00:10:00", "dose_units"] = 5.0

        # Set rate
        df.loc["2024-01-01 00:00:00", ColumnNames.RATE.value] = 1.2

        result = rollover_basal_rate(df, delivery_type="automated")

        expected_basal = 1.2 / 12

        # Row with bolus should have bolus + basal
        assert result.loc["2024-01-01 00:10:00", "dose_units"] == pytest.approx(
            5.0 + expected_basal, rel=1e-9
        )

        # Other rows should just have basal
        assert result.loc["2024-01-01 00:00:00", "dose_units"] == pytest.approx(
            expected_basal, rel=1e-9
        )


class TestRolloverBasalTempEdgeCases:
    """Additional edge case tests for temp basal delivery type."""

    def test_all_nan_rate_column_temp(self):
        """Test edge case: rate column exists but is all NaN for temp delivery."""
        datetime_index = pd.date_range(
            start="2024-01-01 00:00:00", periods=10, freq="5min"
        )

        data = {
            "p_num": ["patient_01"] * 10,
            "bg_mM": [5.0] * 10,
            "dose_units": [0.0] * 10,
            ColumnNames.RATE.value: [None] * 10,  # All NaN
            ColumnNames.BASAL_DURATION_MINS.value: [None] * 10,
        }

        df = pd.DataFrame(data, index=datetime_index)
        df.index.name = "datetime"

        result = rollover_basal_rate(df, delivery_type="temp")

        # All dose_units should remain 0
        assert all(
            result["dose_units"] == 0.0
        ), "All dose_units should be 0 when rate column is all NaN"

    def test_leading_nan_rates_temp(self):
        """Test handling of leading NaN rates before first rate event for temp delivery."""
        datetime_index = pd.date_range(
            start="2024-01-01 00:00:00", periods=12, freq="5min"
        )

        data = {
            "p_num": ["patient_01"] * 12,
            "bg_mM": [5.0] * 12,
            "dose_units": [0.0] * 12,
            ColumnNames.RATE.value: [None] * 12,
            ColumnNames.BASAL_DURATION_MINS.value: [None] * 12,
        }

        df = pd.DataFrame(data, index=datetime_index)
        df.index.name = "datetime"

        # First rate appears at row 4 (20 minutes in)
        df.loc["2024-01-01 00:20:00", ColumnNames.RATE.value] = 1.2
        df.loc["2024-01-01 00:20:00", ColumnNames.BASAL_DURATION_MINS.value] = 30

        result = rollover_basal_rate(df, delivery_type="temp")

        expected_dose = 1.2 / 12

        # Rows 0-3 should have 0 dose (before first rate)
        for i in range(4):
            assert result.iloc[i]["dose_units"] == pytest.approx(
                0.0, rel=1e-9
            ), f"Row {i} should have 0 dose (before first rate)"

        # Rows 4-9 should have dose (30 min duration = 6 rows)
        for i in range(4, 10):
            assert result.iloc[i]["dose_units"] == pytest.approx(
                expected_dose, rel=1e-9
            ), f"Row {i} should have dose {expected_dose}"

        # Rows 10-11 should have 0 (after duration ends)
        for i in range(10, 12):
            assert result.iloc[i]["dose_units"] == pytest.approx(
                0.0, rel=1e-9
            ), f"Row {i} should have 0 dose (after duration ends)"

    def test_nan_duration_with_valid_rate(self):
        """Test that NaN duration_mins is handled when rate is valid."""
        datetime_index = pd.date_range(
            start="2024-01-01 00:00:00", periods=10, freq="5min"
        )

        data = {
            "p_num": ["patient_01"] * 10,
            "bg_mM": [5.0] * 10,
            "dose_units": [0.0] * 10,
            ColumnNames.RATE.value: [None] * 10,
            ColumnNames.BASAL_DURATION_MINS.value: [None] * 10,
        }

        df = pd.DataFrame(data, index=datetime_index)
        df.index.name = "datetime"

        # Set rate but leave duration as NaN
        df.loc["2024-01-01 00:00:00", ColumnNames.RATE.value] = 1.2
        # duration_mins remains NaN

        result = rollover_basal_rate(df, delivery_type="temp")

        # Should skip this rate since duration is NaN - all doses should be 0
        assert all(
            result["dose_units"] == 0.0
        ), "All dose_units should be 0 when duration_mins is NaN"
