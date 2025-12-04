"""Model evaluation and comparison utilities."""

from typing import Dict, List, Any, Optional
import pandas as pd

from src.models.base import BaseTSFM
from src.utils.logging import error_print

class ModelEvaluator:
    """Handles evaluation and comparison of TSFM models."""
    
    def __init__(self, metrics: Optional[List[str]] = None):
        """Initialize evaluator with metric configuration."""
        self.metrics = metrics or ["mse", "mae", "rmse"]
    
    def evaluate(self, model: BaseTSFM, test_data: pd.DataFrame) -> Dict[str, float]:
        """Evaluate a single model on test data."""
        return model.evaluate(test_data)
    
    def compare_models(
        self,
        models: Dict[str, BaseTSFM],
        test_data: pd.DataFrame,
        metrics: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Compare multiple models on the same test data.
        
        Args:
            models: Dictionary mapping model names to model instances.
            test_data: Test dataset for evaluation.
            metrics: Specific metrics to compare (uses self.metrics if None).
            
        Returns:
            DataFrame with models as rows and metrics as columns.
        """
        results = {}
        for name, model in models.items():
            results[name] = model.evaluate(test_data)
        return pd.DataFrame(results).T

def compare_models(
    models: List[BaseTSFM], test_data: Any, metrics: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """Compare multiple models on the same test dataset.

    Evaluates each fitted model on the provided test data and collects
    their performance metrics for comparison.

    Args:
        models: List of fitted BaseTSFM instances to compare.
        test_data: Test dataset compatible with all models' _prepare_data methods.
        metrics: List of metric names to include in results. Currently unused;
            all computed metrics are returned.

    Returns:
        Dict[str, Dict[str, float]]: Dictionary mapping model class names to
            their evaluation metrics dictionaries.

    Note:
        Models that are not fitted will be skipped with a warning message.
    """
    results = {}

    for model in models:
        if not model.is_fitted:
            error_print(f"Model {model.__class__.__name__} is not fitted, skipping")
            continue

        # Prepare test data
        _, _, test_loader = model._prepare_data(None, None, test_data)

        # Evaluate model
        model_metrics = model.evaluate(test_loader)
        results[model.__class__.__name__] = model_metrics

    return results
