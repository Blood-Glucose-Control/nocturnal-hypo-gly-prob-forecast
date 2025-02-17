# Load all sktime forecasters to be accessible by class name
from sktime.registry import all_estimators


def load_all_forecasters() -> dict:
    """Load all sktime forecasters to be accessible by class name"""
    return {name: est for name, est in all_estimators(estimator_types="forecaster")}


def get_estimator(forecasters: dict, model_name: str) -> object:
    """Returns the sktime estimator class given the class name"""
    if model_name not in forecasters:
        raise ValueError(f"Model {model_name} not found in sktime")
    return forecasters[model_name]
