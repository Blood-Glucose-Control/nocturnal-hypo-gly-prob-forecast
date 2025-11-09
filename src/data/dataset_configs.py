# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)
# For commercial licensing, contact: [Add your contact information]

"""
Dataset configurations for automatic data fetching and processing.

This module contains configuration dictionaries for each dataset, specifying
how to fetch raw data from external sources and what files are required
for each dataset type.
"""

from typing import Dict
from pathlib import Path

from src.data.models import DatasetConfig, DatasetSourceType


# Configuration for the Kaggle Bristol T1D dataset
KAGGLE_BRIST1D_CONFIG: DatasetConfig = {
    "source": DatasetSourceType.KAGGLE_BRIS_T1D,
    "competition_name": "brist1d",
    "required_files": ["train.csv", "test.csv", "sample_submission.csv"],
    "description": "Bristol Type 1 Diabetes dataset from Kaggle",
    "cache_path": "kaggle_brisT1D",
    "citation": "Bristol Type 1 Diabetes Dataset, Kaggle Competition",
    "url": "https://www.kaggle.com/competitions/brist1d",
}

# Configuration for the Gluroo dataset
GLUROO_CONFIG: DatasetConfig = {
    "source": DatasetSourceType.LOCAL,
    "required_files": ["gluroo_cached.csv"],
    "description": "Gluroo diabetes dataset",
    "citation": "Gluroo Dataset",
    "cache_path": "gluroo",
}

# Configuration for the SimGlucose dataset
SIMGLUCOSE_CONFIG: DatasetConfig = {
    "source": DatasetSourceType.LOCAL,
    "description": "SimGlucose synthetic diabetes dataset",
    "citation": "SimGlucose Dataset",
    "cache_path": "simglucose",
}


#### Awesome CGM datasets
AWESOME_CGM_CONFIG = {
    "source": DatasetSourceType.AWESOME_CGM,
    "cache_path": "awesome_cgm",
    "description": "Awesome CGM dataset",
}

ALEPPO_CONFIG: DatasetConfig = {
    "source": DatasetSourceType.ALEPPO,
    "cache_path": str(Path(AWESOME_CGM_CONFIG["cache_path"]) / "aleppo"),
    "description": "Aleppo dataset",
    "required_files": ["Data Tables"],
    "url": "https://github.com/IrinaStatsLab/Awesome-CGM/wiki/Aleppo-(2017)",
    "citation": "Aleppo Dataset",
}

# Configuration for the Lynch 2022 dataset
# TODO: This will need to be modified to use the new cache system.
# It should use cached path instead of source path.
# Not sure about the namesapced approach or just a simple path like this.
LYNCH_2022_CONFIG: DatasetConfig = {
    "source": DatasetSourceType.LYNCH_2022,
    "source_path": "src/data/diabetes_datasets/awesome_cgm/lynch_2022",
    "description": "Lynch 2022 IOBP2 RCT dataset",
    "citation": "Lynch et al. 2022",
    "required_files": ["IOBP2 RCT Public Dataset"],
}


# Mapping of dataset names to their configurations
DATASET_CONFIGS = {
    DatasetSourceType.KAGGLE_BRIS_T1D.value: KAGGLE_BRIST1D_CONFIG,
    DatasetSourceType.GLUROO.value: GLUROO_CONFIG,
    DatasetSourceType.SIMGLUCOSE.value: SIMGLUCOSE_CONFIG,
    DatasetSourceType.ALEPPO.value: ALEPPO_CONFIG,
    DatasetSourceType.LYNCH_2022.value: LYNCH_2022_CONFIG,
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
    if dataset_name not in DATASET_CONFIGS:
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
        "description": config.get("description", "No description available"),
        "citation": config.get("citation", "No citation available"),
        "url": config.get("url", "No URL available"),
    }


def register_dataset(dataset_name: str, dataset_config: DatasetConfig):
    """
    This is used to inject test datasset for testing purposes in real time.
    irl, we should use the normal way to register a new dataset.
    """
    DATASET_CONFIGS[dataset_name] = dataset_config
