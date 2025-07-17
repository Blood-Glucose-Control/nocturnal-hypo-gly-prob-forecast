"""
Dataset configurations for automatic data fetching and processing.

This module contains configuration dictionaries for each dataset, specifying
how to fetch raw data from external sources and what files are required
for each dataset type.
"""

from typing import Dict, Any

# Configuration for the Kaggle Bristol T1D dataset
KAGGLE_BRIST1D_CONFIG = {
    "source": "kaggle",
    "competition_name": "brist1d",
    "required_files": ["train.csv", "test.csv", "sample_submission.csv"],
    "description": "Bristol Type 1 Diabetes dataset from Kaggle",
    "citation": "Bristol Type 1 Diabetes Dataset, Kaggle Competition",
    "url": "https://www.kaggle.com/competitions/brist1d",
}

# Configuration for the Gluroo dataset
GLUROO_CONFIG = {
    "source": "local",
    "source_path": "src/data/gluroo",
    "required_files": ["gluroo_cached.csv"],
    "description": "Gluroo diabetes dataset",
    "citation": "Gluroo Dataset",
}

# Configuration for the SimGlucose dataset
SIMGLUCOSE_CONFIG = {
    "source": "local",
    "source_path": "src/data/datasets/simglucose",
    "description": "SimGlucose synthetic diabetes dataset",
    "citation": "SimGlucose Dataset",
}

# Mapping of dataset names to their configurations
DATASET_CONFIGS = {
    "kaggle_brisT1D": KAGGLE_BRIST1D_CONFIG,
    "gluroo": GLUROO_CONFIG,
    "simglucose": SIMGLUCOSE_CONFIG,
}


def get_dataset_config(dataset_name: str) -> Dict[str, Any]:
    """
    Get configuration for a specific dataset.

    Args:
        dataset_name (str): Name of the dataset

    Returns:
        Dict[str, Any]: Dataset configuration

    Raises:
        ValueError: If dataset configuration is not found
    """
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Configuration not found for dataset: {dataset_name}")

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
        "description": config.get("description", "No description available"),
        "citation": config.get("citation", "No citation available"),
        "url": config.get("url", "No URL available"),
    }
