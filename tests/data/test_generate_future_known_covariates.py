import numpy as np
import pandas as pd
import pytest

from src.data.preprocessing.feature_engineering import generate_future_known_covariates


class TestGenerateFutureKnownCovariates:
    """Tests for generate_future_known_covariates()."""

    def test_shape_and_timestamps(self):
        """Output has correct shape, columns, and timestamps start after last."""
        last_ts = pd.Timestamp("2020-06-15 00:00:00")
        result = generate_future_known_covariates(
            last_timestamp=last_ts,
            forecast_length=72,
            known_covariate_cols=["hour_sin", "hour_cos"],
            interval_mins=5,
        )
        assert result.shape == (72, 2)
        assert list(result.columns) == ["hour_sin", "hour_cos"]
        assert result.index[0] == pd.Timestamp("2020-06-15 00:05:00")
        assert result.isna().sum().sum() == 0

    def test_sin_cos_unit_circle(self):
        """sin^2 + cos^2 == 1 for all rows."""
        result = generate_future_known_covariates(
            last_timestamp=pd.Timestamp("2020-06-15 23:00:00"),
            forecast_length=72,
            known_covariate_cols=["hour_sin", "hour_cos"],
        )
        norm = np.sqrt(result["hour_sin"] ** 2 + result["hour_cos"] ** 2)
        np.testing.assert_allclose(norm.values, 1.0, atol=1e-10)

    def test_midnight_crossing(self):
        """At midnight, sin=0 and cos=1."""
        result = generate_future_known_covariates(
            last_timestamp=pd.Timestamp("2020-06-15 23:50:00"),
            forecast_length=6,
            known_covariate_cols=["hour_sin", "hour_cos"],
            interval_mins=5,
        )
        midnight_row = result.loc[pd.Timestamp("2020-06-16 00:00:00")]
        assert abs(midnight_row["hour_sin"]) < 1e-10
        assert abs(midnight_row["hour_cos"] - 1.0) < 1e-10

    def test_empty_cols_and_unknown_raises(self):
        """Empty list returns 0-column DataFrame; unknown name raises."""
        result = generate_future_known_covariates(
            last_timestamp=pd.Timestamp("2020-01-01"),
            forecast_length=12,
            known_covariate_cols=[],
        )
        assert result.shape == (12, 0)

        with pytest.raises(ValueError, match="Unsupported known covariate"):
            generate_future_known_covariates(
                last_timestamp=pd.Timestamp("2020-01-01"),
                forecast_length=12,
                known_covariate_cols=["unknown_feature"],
            )
