"""Regression metrics for time series forecasting evaluation."""

from numpy.typing import ArrayLike
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    root_mean_squared_error,
)


def compute_metrics(predictions: ArrayLike, targets: ArrayLike) -> dict[str, float]:
    """
    Compute standard regression metrics for forecast evaluation.

    Args:
        predictions: Predicted values (array-like)
        targets: Actual/ground truth values (array-like)

    Returns:
        Dictionary with:
            - rmse: Root Mean Squared Error
            - mae: Mean Absolute Error
            - mape: Mean Absolute Percentage Error (%)
    """
    return {
        "rmse": float(root_mean_squared_error(targets, predictions)),
        "mae": float(mean_absolute_error(targets, predictions)),
        "mape": float(mean_absolute_percentage_error(targets, predictions) * 100),
    }

