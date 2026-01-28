# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)
# For commercial licensing, contact: christopher/cjrisi AT gluroo/uwaterloo DOT com/ca

from sktime.registry import all_estimators as sktime_estimators
from sklearn.utils.discovery import all_estimators as sklearn_estimators


def load_all_forecasters() -> dict:
    """Load all sktime forecasters to be accessible by class name"""
    return {name: est for name, est in sktime_estimators(estimator_types="forecaster")}


def load_all_regressors() -> dict:
    """Load all sklearn regressors to be accessible by class name"""
    return {name: est for name, est in sklearn_estimators(type_filter="regressor")}


def get_estimator(estimators: dict, model_name: str) -> object:
    """Returns the sktime estimator class given the class name"""
    if model_name not in estimators:
        raise ValueError(f"Model {model_name} not found in sktime")
    return estimators[model_name]
