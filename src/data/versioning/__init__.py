"""
Data versioning and registry module.

Provides dataset registry with holdout management for reproducible experiments.
"""

from .dataset_registry import (
    DatasetRegistry,
    get_dataset_registry,
    load_holdout_data,
    load_split_data,
    load_training_data,
)

__all__ = [
    "DatasetRegistry",
    "get_dataset_registry",
    "load_training_data",
    "load_holdout_data",
    "load_split_data",
]
