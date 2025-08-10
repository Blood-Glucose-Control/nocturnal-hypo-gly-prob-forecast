"""
Dataset configurations for automatic data fetching and processing.

This module contains configuration dictionaries for each dataset, specifying
how to fetch raw data from external sources and what files are required
for each dataset type.
"""

from typing import Dict
from src.data.models.data import DatasetConfig, Dataset

# Configuration for the Kaggle Bristol T1D dataset
KAGGLE_BRIST1D_CONFIG = DatasetConfig.model_validate(
    {
        "source": "kaggle",
        "competition_name": "brist1d",
        "required_files": ["train.csv", "test.csv", "sample_submission.csv"],
        "description": "Bristol Type 1 Diabetes dataset from Kaggle",
        "citation": "Bristol Type 1 Diabetes Dataset, Kaggle Competition",
        "url": "https://www.kaggle.com/competitions/brist1d",
    }
)

# Configuration for the Gluroo dataset
GLUROO_CONFIG = DatasetConfig.model_validate(
    {
        "source": "local",
        "source_path": "src/data/gluroo",
        "required_files": ["gluroo_cached.csv"],
        "description": "Gluroo diabetes dataset",
        "citation": "Gluroo Dataset",
    }
)

# Configuration for the SimGlucose dataset
SIMGLUCOSE_CONFIG = DatasetConfig.model_validate(
    {
        "source": "local",
        "source_path": "src/data/datasets/simglucose",
        "description": "SimGlucose synthetic diabetes dataset",
        "citation": "SimGlucose Dataset",
    }
)

# Mapping of dataset names to their configurations
DATASET_CONFIGS = {
    Dataset.KAGGLE_BRIS_T1D: KAGGLE_BRIST1D_CONFIG,
    Dataset.GLUROO: GLUROO_CONFIG,
    Dataset.SIMGLUCOSE: SIMGLUCOSE_CONFIG,
}


def get_dataset_config(dataset_name: str) -> DatasetConfig:
    """
    Get configuration for a specific dataset.

    Args:
        dataset_name (str): Name of the dataset

    Returns:
        Dict[str, Any]: Dataset configuration

    Raises:
        ValueError: If dataset configuration is not found
    """
    dataset_enum = Dataset(dataset_name)
    if dataset_enum not in DATASET_CONFIGS:
        available = list_available_datasets()
        raise ValueError(
            f"Configuration not found for dataset: {dataset_name}. "
            f"Available datasets: {available}"
        )
    return DATASET_CONFIGS[dataset_name]


def list_available_datasets() -> list[str]:
    """
    Get list of available datasets.

    Returns:
        list[str]: List of available dataset names
    """
    return list(DATASET_CONFIGS.keys())


def get_dataset_info(dataset_name: str) -> Dict[str, str]:
    """
    Get basic information about a dataset.

    Args:
        dataset_name (str): Name of the dataset

    Returns:
        Dict[str, str]: Dataset information (description, citation, url)

    Raises:
        ValueError: If dataset configuration is not found
    """
    config = get_dataset_config(dataset_name)
    return {
        "description": config.description,
        "citation": config.citation,
        "url": config.url,
    }
