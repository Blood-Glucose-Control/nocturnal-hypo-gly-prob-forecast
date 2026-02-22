"""Standard regression metrics for time series forecasting evaluation."""

from typing import Dict

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)


def compute_regression_metrics(
    y_pred: np.ndarray, y_true: np.ndarray
) -> Dict[str, float]:
    """Compute standard regression metrics.

    Calculates MSE, RMSE, MAE, and MAPE for comparing predictions to ground truth.

    Args:
        y_pred: Predicted values as numpy array.
        y_true: Ground truth values as numpy array.

    Returns:
        Dictionary containing:
            - mse: Mean Squared Error
            - rmse: Root Mean Squared Error
            - mae: Mean Absolute Error
            - mape: Mean Absolute Percentage Error (as percentage)
    """
    # Flatten if needed for comparison
    y_pred = np.asarray(y_pred).flatten()
    y_true = np.asarray(y_true).flatten()

    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))  # Backwards compatible with older sklearn versions
    mae = float(mean_absolute_error(y_true, y_pred))
    mape = float(mean_absolute_percentage_error(y_true, y_pred) * 100)

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
    }
