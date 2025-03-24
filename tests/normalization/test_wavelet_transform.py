import pytest
import numpy as np
import pandas as pd
from src.data.data_transforms import apply_wavelet_transform


class TestWaveletTransform:
    def setUp(self):
        self.window_length = 5

    @pytest.fixture
    def sample_data(self):
        # Create sample time series data with multiple patients
        np.random.seed(42)
        dates = pd.date_range(
            start="2023-01-01", periods=128, freq="H"
        )  # Power of 2 length

        # Create two patients' worth of data
        patients_data = []
        for p_num in [1, 2]:
            # Create a signal with known frequencies for better testing
            # NOTE: just combining diff frequency components for testing
            # maybe use bgl data itself here?
            t = np.linspace(0, 4 * np.pi, 128)
            values = (
                np.sin(t)  # Low frequency component
                + 0.5 * np.sin(4 * t)  # Medium frequency component
                + 0.25 * np.sin(8 * t)  # High frequency component
                + np.random.normal(0, 0.1, 128)
            )  # Some noise

            # Create patient dataframe with required columns
            patient_df = pd.DataFrame(
                {
                    "datetime": dates,
                    "bg-0:00": values,
                    "p_num": p_num,
                    "id": [f"{p_num}_{i}" for i in range(len(dates))],
                }
            )
            patients_data.append(patient_df)

        # Combine patient data
        return pd.concat(patients_data, axis=0).reset_index(drop=True)

    @pytest.fixture
    def normalizer(self):
        return apply_wavelet_transform  # 24-hour window

    def test_empty_series(self, normalizer):
        """
        Test behavior with empty series
        """
        empty_series = pd.DataFrame({"bg-0:00": [], "p_num": 1})
        # ensure error raised
        with pytest.raises(ValueError):
            normalizer(empty_series)

    def test_series_shorter_than_window(self, normalizer):
        """Test behavior when series is shorter than window size"""
        short_series = pd.DataFrame({"bg-0:00": np.random.randn(10), "p_num": 1})
        normalized = normalizer(short_series)

        # check non-empty and non-nan
        assert not normalized.empty
        assert not np.any(np.isnan(normalized))

    def test_basic_transform(self, sample_data, normalizer):
        """Test basic wavelet transform functionality"""
        transformed_data = normalizer(sample_data, wavelet="db4", level=3)

        # Check output structure
        assert isinstance(transformed_data, pd.DataFrame)
        assert len(transformed_data) == len(sample_data)
        assert not transformed_data.isna().any().any()

    def test_different_wavelets(self, sample_data, normalizer):
        """Test transform with different wavelet types"""
        wavelet_types = ["db4", "haar", "sym4"]
        for wavelet in wavelet_types:
            transformed = normalizer(sample_data, wavelet=wavelet, level=2)
            assert not transformed.empty
            assert not transformed.isna().any().any()

    def test_different_levels(self, sample_data, normalizer):
        """Test transform with different decomposition levels"""
        for level in [1, 2, 3, 4]:
            transformed = normalizer(sample_data, wavelet="db4", level=level)
            assert len(transformed) == len(sample_data)
            assert not transformed.isna().any().any()

    def test_invalid_inputs(self, sample_data, normalizer):
        """Test error handling for invalid inputs"""
        # Test invalid wavelet
        with pytest.raises(ValueError):
            normalizer(sample_data, wavelet="invalid_wavelet")

    def test_empty_data(self, normalizer):
        """Test behavior with empty DataFrame"""
        empty_df = pd.DataFrame({"bg-0:00": [], "p_num": []})
        with pytest.raises(ValueError):
            normalizer(empty_df)

    def test_non_power_of_two(self, normalizer):
        """Test handling of data with length not power of 2"""
        dates = pd.date_range(start="2023-01-01", periods=100, freq="H")
        values = np.random.randn(100)
        data = pd.DataFrame({"bg-0:00": values, "p_num": 1}, index=dates)

        transformed = normalizer(data)
        # Should either pad to next power of 2 or handle non-power-2 length appropriately
        assert len(transformed) == len(data)
        assert not transformed.isna().any().any()

    def test_column_handling(self, sample_data, normalizer):
        """Test handling of multiple columns"""
        dates = pd.date_range(start="2023-01-01", periods=128, freq="H")
        df = pd.DataFrame(
            {
                "bg-0:00": np.random.randn(128),
                "bg-1:00": np.random.randn(128),
                "p_num": [1, 2] * 64,
            },
            index=dates,
        )

        transformed = normalizer(df)
        assert set(transformed.columns) == set(df.columns)
        assert not transformed.isna().any().any()
