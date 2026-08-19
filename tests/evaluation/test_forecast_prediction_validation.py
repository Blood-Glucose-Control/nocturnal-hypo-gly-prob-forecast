"""Tests for forecast value validation in forecasting workflow evaluation."""

import numpy as np
import pytest

from src.workflows.forecasting.evaluation import (
    ForecastValidationError,
    _validate_forecast_values,
)


def test_validate_forecast_values_raises_on_all_nan() -> None:
    with pytest.raises(ForecastValidationError, match="All predictions are non-finite"):
        _validate_forecast_values(
            np.array([np.nan, np.nan, np.nan]),
            phase_name="1_after_training",
            dataset_name="aleppo_2017",
            patient_id="ale_134",
        )


def test_validate_forecast_values_raises_on_all_inf() -> None:
    with pytest.raises(ForecastValidationError, match="All predictions are non-finite"):
        _validate_forecast_values(
            np.array([np.inf, -np.inf, np.inf]),
            phase_name="2_after_loading",
            dataset_name="aleppo_2017",
            patient_id="ale_134",
        )


def test_validate_forecast_values_allows_mixed_finite_and_nan() -> None:
    validated = _validate_forecast_values(
        np.array([5.1, np.nan, 6.3]),
        phase_name="1_after_training",
        dataset_name="aleppo_2017",
        patient_id="ale_134",
    )
    np.testing.assert_allclose(validated[[0, 2]], np.array([5.1, 6.3]))
    assert np.isnan(validated[1])
