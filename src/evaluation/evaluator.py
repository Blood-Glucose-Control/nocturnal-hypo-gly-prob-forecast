"""Model evaluation and comparison utilities.

This module provides utilities for evaluating and comparing time series
forecasting models using the standardized predict() interface and shared
metrics from src.evaluation.metrics.

For full holdout evaluation with episode-based benchmarking, use the
dedicated holdout_eval.py script instead.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from src.models.base import BaseTimeSeriesFoundationModel
from src.evaluation.metrics import compute_regression_metrics
from src.utils.logging_helper import error_print


class ModelEvaluator:
    """Handles evaluation and comparison of TSFM models.

    Uses model.predict() and shared metrics for evaluation.
    For episode-based holdout evaluation, use holdout_eval.py instead.
    """

    def __init__(self, metrics: Optional[List[str]] = None):
        """Initialize evaluator with metric configuration."""
        self.metrics = metrics or ["mse", "mae", "rmse"]

    def evaluate(
        self,
        model: BaseTimeSeriesFoundationModel,
        context_data: pd.DataFrame,
        target_data: np.ndarray,
    ) -> Dict[str, float]:
        """Evaluate a single model on test data.

        Args:
            model: Fitted model instance with predict() method.
            context_data: Input DataFrame for prediction (with 'bg_mM' column).
            target_data: Ground truth values as numpy array.

        Returns:
            Dictionary of metric names to values.
        """
        predictions = model.predict(context_data)
        return compute_regression_metrics(predictions, target_data)

    def compare_models(
        self,
        models: Dict[str, BaseTimeSeriesFoundationModel],
        context_data: pd.DataFrame,
        target_data: np.ndarray,
        metrics: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Compare multiple models on the same test data.

        Args:
            models: Dictionary mapping model names to model instances.
            context_data: Input DataFrame for prediction.
            target_data: Ground truth values.
            metrics: Specific metrics to filter (uses all if None).

        Returns:
            DataFrame with models as rows and metrics as columns.
        """
        results = {}
        for name, model in models.items():
            try:
                predictions = model.predict(context_data)
                results[name] = compute_regression_metrics(predictions, target_data)
            except Exception as e:
                error_print(f"Model {name} failed: {e}")
                results[name] = {}

        df = pd.DataFrame(results).T
        if metrics:
            df = df[[m for m in metrics if m in df.columns]]
        return df


def compare_models(
    models: List[BaseTimeSeriesFoundationModel],
    context_data: pd.DataFrame,
    target_data: np.ndarray,
    metrics: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """Compare multiple models on the same test dataset.

    Evaluates each fitted model using predict() and computes standard
    regression metrics for comparison.

    Args:
        models: List of fitted BaseTimeSeriesFoundationModel instances.
        context_data: Input DataFrame for prediction (with 'bg_mM' column).
        target_data: Ground truth values as numpy array.
        metrics: List of metric names to include in results. If None,
            all computed metrics are returned.

    Returns:
        Dict mapping model class names to metric dictionaries.

    Note:
        Models that are not fitted will be skipped with a warning.
    """
    results = {}

    for model in models:
        if not model.is_fitted:
            error_print(f"Model {model.__class__.__name__} is not fitted, skipping")
            continue

        try:
            predictions = model.predict(context_data)
            model_metrics = compute_regression_metrics(predictions, target_data)

            # Filter metrics if specified
            if metrics:
                model_metrics = {k: v for k, v in model_metrics.items() if k in metrics}

            results[model.__class__.__name__] = model_metrics
        except Exception as e:
            error_print(f"Model {model.__class__.__name__} evaluation failed: {e}")

    return results
