# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)
# For commercial licensing, contact: [Add your contact information]

"""
Dataset registry for tracking dataset versions and holdout configurations.

This module provides utilities to load datasets with their associated holdout
configurations, ensuring consistent train/test splits across all experiments.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import pandas as pd

from src.data.diabetes_datasets.data_loader import get_loader
from src.data.versioning.holdout_config import HoldoutConfig
from src.data.versioning.holdout_manager import HoldoutManager

logger = logging.getLogger(__name__)


class DatasetRegistry:
    """Registry for managing datasets with holdout configurations."""

    def __init__(self, holdout_config_dir: Union[str, Path] = "configs/data/holdout"):
        """Initialize dataset registry.

        Args:
            holdout_config_dir: Directory containing holdout configuration files
        """
        self.holdout_config_dir = Path(holdout_config_dir)
        self._loaded_configs: Dict[str, HoldoutConfig] = {}

    def get_holdout_config(self, dataset_name: str) -> Optional[HoldoutConfig]:
        """Load holdout configuration for a dataset.

        Args:
            dataset_name: Name of the dataset

        Returns:
            HoldoutConfig if found, None otherwise
        """
        # Check cache
        if dataset_name in self._loaded_configs:
            return self._loaded_configs[dataset_name]

        # Try to load from file
        config_path = self.holdout_config_dir / f"{dataset_name}.yaml"

        if not config_path.exists():
            logger.warning(
                f"No holdout config found for {dataset_name} at {config_path}. "
                f"Please run scripts/data_processing_scripts/generate_holdout_configs.py"
            )
            return None

        try:
            config = HoldoutConfig.load(config_path)
            self._loaded_configs[dataset_name] = config
            return config
        except Exception as e:
            logger.error(f"Error loading holdout config for {dataset_name}: {e}")
            return None

    def load_dataset_with_split(
        self,
        dataset_name: str,
        patient_col: str = "p_num",
        time_col: str = "datetime",
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load dataset and apply holdout split.

        Args:
            dataset_name: Name of dataset to load
            patient_col: Column name for patient IDs
            time_col: Column name for timestamps

        Returns:
            Tuple of (train_data, holdout_data)
        """
        logger.info(" ")
        logger.info(f"Loading dataset with holdout split: {dataset_name}")

        # Load full dataset (already loaded in __init__)
        loader = get_loader(dataset_name, use_cached=True)  # type: ignore[arg-type]
        full_data = loader.processed_data

        # Load holdout configuration
        config = self.get_holdout_config(dataset_name)

        if config is None:
            raise ValueError(
                f"No holdout configuration found for {dataset_name}. "
                f"Run scripts/data_processing_scripts/generate_holdout_configs.py first."
            )

        # Apply split
        manager = HoldoutManager(config)
        train_data, holdout_data = manager.split_data(
            full_data, patient_col=patient_col, time_col=time_col
        )

        # Validate split
        validation = manager.validate_split(train_data, holdout_data, patient_col)
        if not all(validation.values()):
            logger.error(f"Split validation failed: {validation}")
            raise ValueError("Invalid data split detected")

        logger.info(
            f"Loaded {dataset_name}: {len(train_data):,} train samples, "
            f"{len(holdout_data):,} holdout samples"
        )

        return train_data, holdout_data

    def load_training_data_only(
        self,
        dataset_name: str,
        patient_col: str = "p_num",
        time_col: str = "datetime",
    ) -> pd.DataFrame:
        """Load only the training portion of a dataset.

        This is the primary method to use when training models to ensure
        holdout data is never accidentally used.

        Args:
            dataset_name: Name of dataset to load
            patient_col: Column name for patient IDs
            time_col: Column name for timestamps

        Returns:
            Training data only
        """
        train_data, _ = self.load_dataset_with_split(
            dataset_name, patient_col=patient_col, time_col=time_col
        )
        return train_data

    def load_holdout_data_only(
        self,
        dataset_name: str,
        patient_col: str = "p_num",
        time_col: str = "datetime",
    ) -> pd.DataFrame:
        """Load only the holdout portion of a dataset.

        Use this for final model evaluation only.

        Args:
            dataset_name: Name of dataset to load
            patient_col: Column name for patient IDs
            time_col: Column name for timestamps

        Returns:
            Holdout data only
        """
        _, holdout_data = self.load_dataset_with_split(
            dataset_name, patient_col=patient_col, time_col=time_col
        )
        return holdout_data

    def get_split_info(self, dataset_name: str) -> Dict:
        """Get information about the data split for a dataset.

        Args:
            dataset_name: Name of the dataset

        Returns:
            Dictionary with split information
        """
        config = self.get_holdout_config(dataset_name)

        if config is None:
            return {"error": "No holdout configuration found"}

        info = {
            "dataset_name": config.dataset_name,
            "holdout_type": config.holdout_type.value,
            "description": config.description,
            "version": config.version,
            "created_date": config.created_date,
        }

        if config.temporal_config:
            info["temporal_split"] = {
                "holdout_percentage": config.temporal_config.holdout_percentage,
                "min_train_samples": config.temporal_config.min_train_samples,
                "min_holdout_samples": config.temporal_config.min_holdout_samples,
            }

        if config.patient_config:
            info["patient_split"] = {
                "holdout_patients": config.patient_config.holdout_patients,
                "num_holdout_patients": len(config.patient_config.holdout_patients),
                "min_train_patients": config.patient_config.min_train_patients,
                "random_seed": config.patient_config.random_seed,
            }

        return info

    def list_available_datasets(self) -> list[str]:
        """List all datasets with holdout configurations.

        Returns:
            List of dataset names
        """
        if not self.holdout_config_dir.exists():
            logger.warning(
                f"Holdout config directory not found: {self.holdout_config_dir}"
            )
            return []

        config_files = list(self.holdout_config_dir.glob("*.yaml"))
        dataset_names = [f.stem for f in config_files]

        return sorted(dataset_names)


# Global registry instance
_registry = None


def get_dataset_registry(
    holdout_config_dir: Union[str, Path] = "configs/data/holdout",
) -> DatasetRegistry:
    """Get or create the global dataset registry instance.

    Args:
        holdout_config_dir: Directory containing holdout configuration files

    Returns:
        DatasetRegistry instance
    """
    global _registry
    if _registry is None:
        _registry = DatasetRegistry(holdout_config_dir)
    return _registry


# Convenience functions using global registry
def load_training_data(dataset_name: str, **kwargs) -> pd.DataFrame:
    """Load training data for a dataset (convenience function).

    Args:
        dataset_name: Name of dataset to load
        **kwargs: Additional arguments passed to load_training_data_only

    Returns:
        Training data only
    """
    registry = get_dataset_registry()
    return registry.load_training_data_only(dataset_name, **kwargs)


def load_holdout_data(dataset_name: str, **kwargs) -> pd.DataFrame:
    """Load holdout data for a dataset (convenience function).

    Args:
        dataset_name: Name of dataset to load
        **kwargs: Additional arguments passed to load_holdout_data_only

    Returns:
        Holdout data only
    """
    registry = get_dataset_registry()
    return registry.load_holdout_data_only(dataset_name, **kwargs)


def load_split_data(dataset_name: str, **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load both training and holdout data (convenience function).

    Args:
        dataset_name: Name of dataset to load
        **kwargs: Additional arguments passed to load_dataset_with_split

    Returns:
        Tuple of (train_data, holdout_data)
    """
    registry = get_dataset_registry()
    return registry.load_dataset_with_split(dataset_name, **kwargs)
