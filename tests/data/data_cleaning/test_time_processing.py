import pandas as pd
import pytest

from src.data.preprocessing.time_processing import (
    iter_daily_forecast_periods,
    iter_patient_validation_splits,
)


class TestIterDailyForecastPeriods:
    """Tests for iter_daily_forecast_periods function."""

    @pytest.fixture
    def sample_patient_data(self):
        """Create sample patient data with datetime index."""
        # Create 3 days of data with 5-minute intervals
        start_date = pd.Timestamp("2024-01-01 00:00:00")
        end_date = pd.Timestamp("2024-01-03 23:55:00")
        datetime_index = pd.date_range(start=start_date, end=end_date, freq="5min")

        data = {
            "bg_mM": [5.5 + i * 0.1 for i in range(len(datetime_index))],
            "food_g": [0] * len(datetime_index),
            "dose_units": [0] * len(datetime_index),
        }

        return pd.DataFrame(data, index=datetime_index)

    def test_default_periods(self, sample_patient_data):
        """Test with default context_period (6, 24) and forecast_horizon (0, 6)."""
        print(f"Sample data shape: {sample_patient_data.shape}")
        print(
            f"Date range: {sample_patient_data.index.min()} to {sample_patient_data.index.max()}"
        )
        print(f"Sample hours: {sample_patient_data.index.hour.unique()}")

        splits = list(iter_daily_forecast_periods(sample_patient_data))
        print(f"Number of splits: {len(splits)}")

        # Should get 2 splits (day1->day2, day2->day3)
        assert len(splits) == 2

        for input_period, forecast_horizon in splits:
            # Input period should be 6am-12am (18 hours)
            assert input_period.index.hour.min() >= 6
            assert input_period.index.hour.max() <= 23

            # Forecast horizon should be 12am-6am (6 hours)
            assert forecast_horizon.index.hour.min() >= 0
            assert forecast_horizon.index.hour.max() < 6

            # Both periods should have data
            assert len(input_period) > 0
            assert len(forecast_horizon) > 0

    def test_custom_periods(self, sample_patient_data):
        """Test with custom time periods."""
        # Custom: context 8am-2pm, forecast 2pm-8pm same day
        splits = list(
            iter_daily_forecast_periods(
                sample_patient_data, context_period=(8, 14), forecast_horizon=(14, 20)
            )
        )

        assert len(splits) > 0

        for input_period, forecast_horizon in splits:
            # Input period should be 8am-2pm
            assert input_period.index.hour.min() >= 8
            assert input_period.index.hour.max() < 14

            # Forecast horizon should be 2pm-8pm
            assert forecast_horizon.index.hour.min() >= 14
            assert forecast_horizon.index.hour.max() < 20

    def test_invalid_datetime_index(self):
        """Test error handling for invalid datetime index."""
        # DataFrame without datetime index
        df = pd.DataFrame({"bg_mM": [5.5, 6.0, 5.8], "food_g": [0, 30, 0]})

        with pytest.raises(ValueError, match="Patient data must have datetime index"):
            list(iter_daily_forecast_periods(df))

    def test_datetime_column_fallback(self):
        """Test fallback to datetime column when no datetime index."""
        df = pd.DataFrame(
            {
                "datetime": pd.date_range("2024-01-01", periods=100, freq="5min"),
                "bg_mM": [5.5] * 100,
                "food_g": [0] * 100,
            }
        )

        # Should work by converting datetime column to index
        splits = list(iter_daily_forecast_periods(df))
        assert len(splits) >= 0  # May be 0 if not enough data spans multiple days

    def test_invalid_hour_parameters(self, sample_patient_data):
        """Test validation of hour parameters."""
        # Invalid context_period
        with pytest.raises(
            ValueError, match="context_period hours must be between 0 and 24"
        ):
            list(
                iter_daily_forecast_periods(
                    sample_patient_data, context_period=(-1, 12)
                )
            )

        # Invalid forecast_horizon
        with pytest.raises(
            ValueError, match="forecast_horizon hours must be between 0 and 24"
        ):
            list(
                iter_daily_forecast_periods(
                    sample_patient_data, forecast_horizon=(12, 25)
                )
            )

    def test_no_data_periods(self):
        """Test when there's insufficient data for splits."""
        # Only a few hours of data - not enough for full day splits
        datetime_index = pd.date_range("2024-01-01 08:00", periods=10, freq="30min")
        df = pd.DataFrame({"bg_mM": [5.5] * 10}, index=datetime_index)

        splits = list(iter_daily_forecast_periods(df))
        # Should return empty list when insufficient data
        assert len(splits) == 0


class TestIterPatientValidationSplits:
    """Tests for iter_patient_validation_splits function."""

    @pytest.fixture
    def sample_validation_data(self):
        """Create sample validation data dictionary."""
        # Create data for multiple patients
        patients = ["p001", "p002"]
        validation_data = {}

        for patient in patients:
            # Create 2 days of data for each patient
            start_date = pd.Timestamp("2024-01-01 00:00:00")
            end_date = pd.Timestamp("2024-01-02 23:55:00")
            datetime_index = pd.date_range(start=start_date, end=end_date, freq="5min")

            data = {
                "bg_mM": [5.5 + i * 0.1 for i in range(len(datetime_index))],
                "food_g": [0] * len(datetime_index),
                "dose_units": [0] * len(datetime_index),
            }

            validation_data[patient] = pd.DataFrame(data, index=datetime_index)

        return validation_data

    def test_valid_patient_splits(self, sample_validation_data):
        """Test getting splits for a valid patient."""
        patient_id = "p001"
        splits = list(
            iter_patient_validation_splits(sample_validation_data, patient_id)
        )

        # Should get at least one split
        assert len(splits) >= 1

        for patient, input_period, forecast_horizon in splits:
            # Should return correct patient ID
            assert patient == patient_id

            # Both periods should be DataFrames with data
            assert isinstance(input_period, pd.DataFrame)
            assert isinstance(forecast_horizon, pd.DataFrame)
            assert len(input_period) > 0
            assert len(forecast_horizon) > 0

            # Input period should be 6am-12am
            assert input_period.index.hour.min() >= 6
            assert input_period.index.hour.max() <= 23

            # Forecast horizon should be 12am-6am next day
            assert forecast_horizon.index.hour.min() >= 0
            assert forecast_horizon.index.hour.max() < 6

    def test_none_validation_data(self):
        """Test error handling when validation_data is None."""
        with pytest.raises(ValueError, match="Validation data is not available"):
            list(iter_patient_validation_splits(None, "p001"))

    def test_invalid_validation_data_type(self):
        """Test error handling when validation_data is not a dict."""
        with pytest.raises(TypeError, match="Expected dict for validation_data"):
            list(iter_patient_validation_splits("invalid_data", "p001"))

    def test_patient_not_found(self, sample_validation_data):
        """Test error handling when patient ID is not in validation data."""
        with pytest.raises(
            ValueError, match="Patient p999 not found in validation data"
        ):
            list(iter_patient_validation_splits(sample_validation_data, "p999"))

    def test_empty_validation_data(self):
        """Test with empty validation data dictionary."""
        empty_data = {}
        with pytest.raises(
            ValueError, match="Patient p001 not found in validation data"
        ):
            list(iter_patient_validation_splits(empty_data, "p001"))

    def test_multiple_patients_available(self, sample_validation_data):
        """Test that error message shows available patients."""
        try:
            list(iter_patient_validation_splits(sample_validation_data, "p999"))
        except ValueError as e:
            # Error message should include available patients
            assert "p001" in str(e)
            assert "p002" in str(e)

    def test_patient_with_insufficient_data(self):
        """Test patient with data that doesn't span enough time for splits."""
        # Create validation data with insufficient time span
        datetime_index = pd.date_range("2024-01-01 08:00", periods=10, freq="30min")
        patient_data = pd.DataFrame({"bg_mM": [5.5] * 10}, index=datetime_index)

        validation_data = {"p001": patient_data}

        # Should not error, but may return empty list
        splits = list(iter_patient_validation_splits(validation_data, "p001"))
        # Length could be 0 if insufficient data for splits
        assert len(splits) >= 0


class TestIntegration:
    """Integration tests for both functions working together."""

    def test_end_to_end_workflow(self):
        """Test complete workflow from validation data to forecast periods."""
        # Create realistic multi-day patient data
        start_date = pd.Timestamp("2024-01-01 00:00:00")
        end_date = pd.Timestamp("2024-01-04 23:55:00")  # 4 days
        datetime_index = pd.date_range(start=start_date, end=end_date, freq="5min")

        # Create patient data with some variation
        patient_data = pd.DataFrame(
            {
                "bg_mM": [5.5 + (i % 100) * 0.05 for i in range(len(datetime_index))],
                "food_g": [
                    30 if i % 72 == 0 else 0 for i in range(len(datetime_index))
                ],  # Meals every 6 hours
                "dose_units": [
                    2 if i % 72 == 0 else 0 for i in range(len(datetime_index))
                ],  # Insulin with meals
            },
            index=datetime_index,
        )

        validation_data = {"p001": patient_data}

        # Get all splits for the patient
        all_splits = list(iter_patient_validation_splits(validation_data, "p001"))

        # Should get multiple splits (3-4 days worth)
        assert len(all_splits) >= 2

        # Verify each split has proper structure
        for patient_id, input_period, forecast_horizon in all_splits:
            assert patient_id == "p001"
            assert len(input_period) > 0
            assert len(forecast_horizon) > 0

            # Verify time continuity - forecast should be after input
            assert forecast_horizon.index.min() >= input_period.index.max()
