import pytest
import numpy as np
import pandas as pd
from src.data.preprocessing.signal_processing import apply_moving_average


class TestMovingAverage:
    def setUp(self):
        self.window_length = 5

    @pytest.fixture
    def sample_data(self):
        # Create sample time series data
        np.random.seed(42)
        dates = pd.date_range(start="2023-01-01", periods=100, freq="h")
        values = np.sin(np.linspace(0, 4 * np.pi, 100)) + np.random.normal(0, 0.1, 100)
        return pd.DataFrame({"bg_mM": values}, index=dates)

    @pytest.fixture
    def normalizer(self):
        return apply_moving_average  # 24-hour window

    def test_on_sample_data(self, normalizer):
        # tests on a sample, custom data
        # construct a sample data with 24 hours of data
        dates = pd.date_range(start="2023-01-01", periods=24, freq="h")
        values = np.linspace(0, 2.3, 24)  # Simple linear sequence from 0 to 2.3
        sample_data = pd.DataFrame({"bg_mM": values}, index=dates)
        normalized_data = normalizer(sample_data, window_size=5)
        assert not normalized_data.empty
        # check moving average is applied correctly (i.e rolling average)
        # hardcode the expected values based on linear sequence with window=5
        expected_values = [
            0.0,  # First value is just itself
            0.05,  # Average of first 2 values
            0.1,  # Average of first 3 values
            0.15,  # Average of first 4 values
            0.2,  # Average of first 5 values
            0.3,  # Moving window of 5 values
            0.4,  # Moving window of 5 values
            0.5,  # Moving window of 5 values
            0.6,  # Moving window of 5 values
            0.7,  # Moving window of 5 values
            0.8,  # Moving window of 5 values
            0.9,  # Moving window of 5 values
            1.0,  # Moving window of 5 values
            1.1,  # Moving window of 5 values
            1.2,  # Moving window of 5 values
            1.3,  # Moving window of 5 values
            1.4,  # Moving window of 5 values
            1.5,  # Moving window of 5 values
            1.6,  # Moving window of 5 values
            1.7,  # Moving window of 5 values
            1.8,  # Moving window of 5 values
            1.9,  # Moving window of 5 values
            2.0,  # Moving window of 5 values
            2.1,  # Moving window of 5 values
        ]
        np.testing.assert_array_almost_equal(
            normalized_data["bg_mM"].values, expected_values
        )

    def test_moving_average_normalization(self, sample_data, normalizer):
        """Test that moving average normalization works as expected"""
        normalized_data = normalizer(sample_data)

        # check that output is the same length as input
        assert len(normalized_data) == len(sample_data), (
            "Output is not the same length as input"
        )

        # ensure no NaNs (the function assumes no NaNs in the data, but anyhow)
        assert not normalized_data["bg_mM"][23:].isna().any(), (
            "There are NaNs in the data"
        )

    def test_invalid_window_size(self, sample_data, normalizer):
        """
        Test that invalid window sizes raise appropriate errors
        """
        with pytest.raises(ValueError):
            normalizer(sample_data, window_size=0)

        with pytest.raises(ValueError):
            normalizer(sample_data, window_size=-1)

    def test_empty_series(self, normalizer):
        """
        Test behavior with empty series
        """
        empty_series = pd.DataFrame({"bg_mM": []})
        res = normalizer(empty_series)
        assert res.empty

    def test_series_shorter_than_window(self, normalizer):
        """Test behavior when series is shorter than window size"""
        short_series = pd.DataFrame({"bg_mM": np.random.randn(10)})
        normalized = normalizer(short_series)

        # check non-empty and non-nan
        assert not normalized.empty
        assert not np.any(np.isnan(normalized))
