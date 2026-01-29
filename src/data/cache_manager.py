# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)
# For commercial licensing, contact: christopher/cjrisi AT gluroo/uwaterloo DOT com/ca

"""
Cache Manager for centralized data caching.

This module provides a centralized cache management system for all datasets
in the nocturnal project. It handles:
- Automatic data fetching from external sources (Kaggle, HuggingFace, etc.)
- Cache directory structure management
- Raw and processed data storage
- Cache validation and cleanup

The cache structure follows:
Root_dir/cache/data/{DatasetName}/raw
Root_dir/cache/data/{DatasetName}/processed
"""

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Literal, Optional, Dict
from typing_extensions import deprecated

import pandas as pd

from src.data.dataset_configs import (
    DatasetConfig,
    DatasetSourceType,
    get_dataset_config,
)
from src.utils.os_helper import get_project_root

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Centralized cache manager for dataset storage and retrieval.

    This class manages a unified cache structure for all datasets, handling
    automatic data fetching from external sources and organizing data in a
    consistent directory structure.
    """

    def __init__(
        self,
        cache_root: str = "cache/data",
    ):
        """
        Initialize the cache manager.

        Args:
            cache_root (str): Root directory for cache storage (relative to project root)
        """
        # Make cache root relative to project root, not current working directory
        project_root = get_project_root()
        self.cache_root = project_root / cache_root
        self.cache_root.mkdir(parents=True, exist_ok=True)

    def get_dataset_cache_path(self, dataset_name: str) -> Path:
        """
        Get the cache path for a specific dataset.

        Args:
            dataset_name (str): Name of the dataset (can be namespaced like "group/dataset")

        Returns:
            Path: Path to the dataset's cache directory
        """

        return self.cache_root / dataset_name

    def get_absolute_path_by_type(
        self,
        dataset_name: str,
        data_type: Literal["interim", "raw", "processed", "cleaning_step"],
    ) -> Path:
        """
        Get the absolute cache path for a specific dataset and data type.
        """
        config = get_dataset_config(dataset_name)
        relative_cache_path = config.cache_path
        if relative_cache_path is None:
            raise ValueError(f"Cache path not specified for dataset {dataset_name}")
        return self.get_dataset_cache_path(relative_cache_path) / data_type

    @deprecated("No longer used, use get_absolute_path_by_type directly instead")
    def get_raw_data_path(self, dataset_name: str) -> Path:
        """Get the raw data path for a specific dataset."""
        return self.get_absolute_path_by_type(dataset_name, "raw")

    @deprecated("No longer used, use get_absolute_path_by_type directly instead")
    def get_cleaning_step_data_path(self, dataset_name: str) -> Path:
        """Get the cleaning step data path for a specific dataset."""
        return self.get_absolute_path_by_type(dataset_name, "cleaning_step")

    @deprecated("No longer used, use get_absolute_path_by_type directly instead")
    def get_processed_data_path(self, dataset_name: str) -> Path:
        """Get the processed data path for a specific dataset."""
        return self.get_absolute_path_by_type(dataset_name, "processed")

    def ensure_raw_data(self, dataset_name: str, dataset_config: DatasetConfig) -> Path:
        """
        Ensure raw data iwavailable given a dataset configuration, fetching it if necessary.

        Args:
            dataset_name (str): Name of the dataset
            dataset_config (Dict[str, Any]): Configuration for the dataset

        Returns:
            Path: Path to the raw data directory where we guarantee the raw data exist

        Raises:
            ValueError: If dataset source is not supported
            RuntimeError: If data fetching fails
        """
        # Absolute raw data path thtat should contain the raw data
        raw_path = self.get_absolute_path_by_type(dataset_name, "raw")

        # Check if raw data already exists
        if self._raw_data_exists(raw_path, dataset_config):
            logger.info(f"Raw data for {dataset_name} already exists in cache")
            return raw_path

        # Fetch data from source
        logger.info(
            f"Raw data for {dataset_name} not found in cache: {raw_path} \n fetching from source"
        )
        source = dataset_config.source
        if source == DatasetSourceType.KAGGLE_BRIS_T1D:
            self._fetch_kaggle_data(dataset_name, raw_path, dataset_config)
        elif source == DatasetSourceType.HUGGING_FACE:
            self._fetch_huggingface_data(dataset_name, raw_path, dataset_config)
        elif source in (
            DatasetSourceType.ALEPPO,
            DatasetSourceType.LYNCH_2022,
            DatasetSourceType.BROWN_2019,
        ):
            self._fetch_manual_download_data(dataset_name, raw_path, dataset_config)
        elif source == DatasetSourceType.LOCAL:
            self._copy_local_data(dataset_name, raw_path, dataset_config)
        else:
            raise ValueError(f"Unsupported data source: {source}")

        return raw_path

    def _raw_data_exists(self, raw_path: Path, dataset_config: DatasetConfig) -> bool:
        """
        Check if raw data exists and is valid for a given dataset configuration.

        Args:
            raw_path (Path): Path to raw data directory
            dataset_config (Dict[str, Any]): Dataset configuration

        Returns:
            bool: True if raw data exists and is valid
        """
        if not raw_path.exists():
            return False

        # Check for required files based on dataset configuration
        required_files = dataset_config.required_files
        if required_files:
            return all((raw_path / file).exists() for file in required_files)

        return any(raw_path.iterdir())

    def _fetch_kaggle_data(
        self, dataset_name: str, raw_path: Path, dataset_config: DatasetConfig
    ):
        """
        Fetch data from Kaggle. Note that Kaggle datasets already have their own caching mechanism.

        Args:
            dataset_name (str): Name of the dataset
            raw_path (Path): Path to store the raw data
            dataset_config (Dict[str, Any]): Dataset configuration

        Raises:
            RuntimeError: If Kaggle data fetching fails
        """
        raw_path.mkdir(parents=True, exist_ok=True)

        competition_name = dataset_config.competition_name
        if not competition_name:
            raise ValueError(
                f"Kaggle competition name not specified for dataset {dataset_name}"
            )

        try:
            # Download from Kaggle
            # Reference: https://github.com/Kaggle/kaggle-api/blob/main/docs/README.md#download-competition-files
            # For example: kaggle competitions download -c brist1d -p cache/data/kaggle_brisT1D/Raw
            logger.info(
                f"Downloading {dataset_name} from Kaggle and saving to {raw_path}"
            )
            cmd = [
                "kaggle",
                "competitions",
                "download",
                "-c",
                competition_name,
                "-p",
                str(raw_path),
            ]
            # Show the command to the user and ask for confirmation before running
            print(
                "\nWe automatically run the following command to download data from Kaggle:\n"
            )
            print("  " + " ".join(cmd) + "\n")
            confirm = input("Proceed? Type 'y' or 'yes' to continue: ").strip().lower()
            if confirm not in ("y", "yes"):
                logger.info("Kaggle download aborted by user.")
                return
            print(f"Running command: {cmd}. This may take a while...")
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info(f"Downloaded {dataset_name} from Kaggle")

            # Extract zip files
            zip_files = list(raw_path.glob("*.zip"))
            for zip_file in zip_files:
                shutil.unpack_archive(str(zip_file), str(raw_path))
                zip_file.unlink()  # Remove zip file after extraction

            logger.info(f"Extracted {dataset_name} data")

        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Failed to fetch Kaggle data for {dataset_name}: {e.stderr}"
            )
        except Exception as e:
            raise RuntimeError(
                f"Unexpected error fetching Kaggle data for {dataset_name}: {str(e)}"
            )

    # TODO: Test this function once we do have a HuggingFace dataset to test with
    def _fetch_huggingface_data(
        self, dataset_name: str, raw_path: Path, dataset_config: DatasetConfig
    ):
        """
        Fetch data from HuggingFace.

        Args:
            dataset_name (str): Name of the dataset
            raw_path (Path): Path to store the raw data
            dataset_config (Dict[str, Any]): Dataset configuration

        Raises:
            RuntimeError: If HuggingFace data fetching fails
        """
        import json

        raw_path.mkdir(parents=True, exist_ok=True)

        dataset_id = dataset_config.hf_dataset_id
        if not dataset_id:
            raise ValueError(
                f"HuggingFace dataset ID not specified for dataset {dataset_name}"
            )

        try:
            import datasets as hf_datasets

            # First, check if we imported the wrong module by examining the file path
            module_path = str(getattr(hf_datasets, "__file__", ""))

            # Check if this is our local datasets module
            if any(
                indicator in module_path
                for indicator in ["nocturnal", "src/data/datasets"]
            ):
                raise ImportError(
                    f"Import collision: imported local datasets module instead of HuggingFace datasets.\n"
                    f"Local module path: {module_path}\n"
                    f"This happens when your local 'src/data/datasets' module is found before HuggingFace datasets.\n"
                    f"Solutions:\n"
                    f"  1. Install HuggingFace datasets: pip install datasets\n"
                    f"  2. Check your PYTHONPATH doesn't prioritize local modules\n"
                    f"  3. Run from project root directory"
                )

            # Secondary check: verify it has HuggingFace-specific functionality
            if not hasattr(hf_datasets, "Dataset") or not hasattr(
                hf_datasets, "DatasetDict"
            ):
                raise ImportError(
                    f"Wrong datasets library imported. Module path: {module_path}\n"
                    f"Expected: HuggingFace datasets library with Dataset and DatasetDict classes.\n"
                    f"Install with: pip install datasets"
                )

            # Load dataset from HuggingFace
            dataset = hf_datasets.load_dataset(dataset_id, streaming=False)

            # Try to use save_to_disk if available, otherwise fall back to JSON
            try:
                # Use getattr to avoid type checker issues
                save_method = getattr(dataset, "save_to_disk", None)
                if save_method:
                    save_method(str(raw_path))
                    logger.info(
                        f"Downloaded {dataset_name} from HuggingFace (binary format)"
                    )
                    return
            except (AttributeError, Exception):
                pass  # Fall through to JSON export

            # Fallback: save as JSON
            # Use getattr to safely check for items method (DatasetDict)
            items_method = getattr(dataset, "items", None)
            if items_method:
                # DatasetDict-like object with multiple splits
                for split_name, split_dataset in items_method():
                    split_path = raw_path / f"{split_name}.json"
                    # Convert to list and save as JSON
                    data = list(split_dataset)
                    with open(split_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
            else:
                # Single dataset
                data = list(dataset)
                with open(raw_path / "data.json", "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

            logger.info(f"Downloaded {dataset_name} from HuggingFace (JSON format)")

        except ImportError:
            raise RuntimeError(
                "HuggingFace datasets library not installed. Install with: pip install datasets"
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to fetch HuggingFace data for {dataset_name}: {str(e)}"
            )

    def _fetch_manual_download_data(
        self, dataset_name: str, raw_path: Path, dataset_config: DatasetConfig
    ):
        """
        Fetch data for datasets that require manual download.
        Raises an error with download instructions if data is not present.
        """
        if self._raw_data_exists(raw_path, dataset_config):
            return raw_path

        raise RuntimeError(
            f"Raw data for '{dataset_name}' not found.\n\n"
            f"Download from: {dataset_config.url}\n"
            f"Required files/folders: {dataset_config.required_files}\n"
            f"Place in: {raw_path}\n\n"
            f"See the dataset's README.md for detailed instructions."
        )

    def _copy_local_data(
        self, dataset_name: str, raw_path: Path, dataset_config: DatasetConfig
    ):
        """
        Copy data from local source.

        Args:
            dataset_name (str): Name of the dataset
            raw_path (Path): Path to store the raw data
            dataset_config (Dict[str, Any]): Dataset configuration

        Raises:
            RuntimeError: If local data copying fails
        """
        source_path = dataset_config.cache_path
        if not source_path:
            raise ValueError(
                f"Local source path not specified for dataset {dataset_name}"
            )

        source_path = Path(source_path)
        if not source_path.exists():
            raise RuntimeError(f"Local source path does not exist: {source_path}")

        raw_path.mkdir(parents=True, exist_ok=True)

        try:
            if source_path.is_file():
                shutil.copy2(source_path, raw_path)
            else:
                shutil.copytree(source_path, raw_path, dirs_exist_ok=True)

            logger.info(f"Copied {dataset_name} from local source")

        except Exception as e:
            raise RuntimeError(
                f"Failed to copy local data for {dataset_name}: {str(e)}"
            )

    @deprecated(
        "Deprecated because we no longer want to save by dataset type. It should just be processed data (we split at the code level)"
    )
    def get_processed_data_path_for_type(
        self, dataset_name: str, dataset_type: str
    ) -> Path:
        """
        Get the processed data path for a specific dataset type.

        Args:
            dataset_name (str): Name of the dataset
            dataset_type (str): Type of dataset (train, test, etc.)

        Returns:
            Path: Path to the processed data for the specific type
        """
        processed_path = self.get_processed_data_path(dataset_name)
        return processed_path / dataset_type

    @deprecated(
        "Deprecated because we no longer want to save by dataset type. It should just be processed data (we split at the code level). Use save_full_processed_data instead"
    )
    def save_processed_data(
        self,
        dataset_name: str,
        dataset_type: str,
        patient_id,
        data,
        file_format: str = "csv",
    ):
        """
        Save processed data to cache. The index is datetime.

        Args:
            dataset_name (str): Name of the dataset
            dataset_type (str): Type of dataset (train, test, etc.)
            patient_id: ID of the patient
            data: Data to save (DataFrame, dict, etc.)
            file_format (str): Format to save the data in
        """
        processed_path = self.get_processed_data_path_for_type(
            dataset_name, dataset_type
        )
        processed_path.mkdir(parents=True, exist_ok=True)

        if file_format == "csv":
            if hasattr(data, "to_csv"):
                data.to_csv(
                    processed_path / f"{patient_id}_{dataset_type}.csv", index=True
                )
            else:
                raise ValueError(f"Cannot save data of type {type(data)} as CSV")
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

        logger.info(
            f"\tSaved processed {dataset_type} data for {dataset_name} - patient: {patient_id}"
        )

    def save_full_processed_data(
        self, dataset_name: str, data: dict[str, pd.DataFrame]
    ):
        """
        Save full processed data (before train/validation split) as CSV files.
        We split the data at the code level not the cache level because that gives us more flexibility and control.

        Args:
            dataset_name (str): Name of the dataset
            data (dict[str, pd.DataFrame]): Dictionary mapping patient_id -> DataFrame
        """
        processed_path = self.get_absolute_path_by_type(dataset_name, "processed")
        processed_path.mkdir(parents=True, exist_ok=True)

        for patient_id, patient_df in data.items():
            csv_path = processed_path / f"{patient_id}_full.csv"
            patient_df.to_csv(csv_path, index=True)

        logger.info(
            f"Saved full processed data for {dataset_name} - {len(data)} patients"
        )

    def load_full_processed_data(
        self, dataset_name: str
    ) -> Optional[Dict[str, pd.DataFrame]]:
        """
        Load full processed data (before train/validation split) from CSV files.

        Args:
            dataset_name (str): Name of the dataset

        Returns:
            Dictionary with patient IDs as keys and DataFrames as values, or None if not found
        """
        processed_path = self.get_absolute_path_by_type(dataset_name, "processed")

        if processed_path.exists():
            result = {}
            # Get all CSV files with _full.csv suffix
            csv_files = [
                f for f in processed_path.iterdir() if f.name.endswith("_full.csv")
            ]

            for csv_file in csv_files:
                # Extract patient ID from filename: remove _full.csv suffix
                FULL_SUFFIX = "_full"
                patient_id = csv_file.stem[: -len(FULL_SUFFIX)]  # Remove "_full" suffix
                # Load the CSV with datetime index (first column is the index)
                df = pd.read_csv(
                    csv_file, index_col=0, parse_dates=True, low_memory=False
                )
                result[patient_id] = df

            return result if result else None

        return None

    def save_split_data(
        self,
        dataset_name: str,
        train_data: dict[str, pd.DataFrame],
        validation_data: dict[str, pd.DataFrame],
        split_params: dict,
    ):
        """
        Save train/validation split data as serialized files.

        Args:
            dataset_name (str): Name of the dataset
            train_data (dict[str, pd.DataFrame]): Training data
            validation_data (dict[str, pd.DataFrame]): Validation data
            split_params (dict): Parameters used for the split (for reproducibility)
        """
        import pickle

        processed_path = self.get_processed_data_path(dataset_name)
        splits_path = processed_path / "splits"
        splits_path.mkdir(parents=True, exist_ok=True)

        # Create a unique split identifier based on parameters
        split_id = self._get_split_id(split_params)
        split_file = splits_path / f"split_{split_id}.pkl"

        split_data = {
            "train_data": train_data,
            "validation_data": validation_data,
            "split_params": split_params,
        }

        with open(split_file, "wb") as f:
            pickle.dump(split_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info(
            f"Saved train/validation split for {dataset_name} with split_id: {split_id}"
        )

    def load_split_data(
        self, dataset_name: str, split_params: dict
    ) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]] | None:
        """
        Load train/validation split data from serialized files.

        Args:
            dataset_name (str): Name of the dataset
            split_params (dict): Parameters used for the split

        Returns:
            Tuple of (train_data, validation_data) or None if not found
        """
        import pickle

        processed_path = self.get_processed_data_path(dataset_name)
        splits_path = processed_path / "splits"

        if not splits_path.exists():
            return None

        split_id = self._get_split_id(split_params)
        split_file = splits_path / f"split_{split_id}.pkl"

        if split_file.exists():
            try:
                with open(split_file, "rb") as f:
                    split_data = pickle.load(f)

                # Verify split parameters match
                if split_data["split_params"] == split_params:
                    return split_data["train_data"], split_data["validation_data"]
                else:
                    logger.warning(
                        f"Split parameters mismatch for {dataset_name}, split_id: {split_id}"
                    )
            except Exception as e:
                logger.error(f"Error loading split data: {e}")

        return None

    def _get_split_id(self, split_params: dict) -> str:
        """
        Generate a unique identifier for split parameters.

        Args:
            split_params (dict): Parameters used for the split

        Returns:
            str: Unique identifier for the split
        """
        import hashlib
        import json

        # Sort keys to ensure consistent hashing
        sorted_params = json.dumps(split_params, sort_keys=True)
        return hashlib.md5(sorted_params.encode()).hexdigest()[:8]

    # TODO: Consolidate this with load_full_processed_data
    def load_processed_data(
        self,
        dataset_name: str,
        file_format: str = "csv",
        dataset_type: str | None = None,
    ) -> dict[str, pd.DataFrame] | None:
        """
        Load processed data with datetime index from cache. processed data has a naming convention of {patient_id}_full.csv

        Args:
            dataset_name (str): Name of the dataset
            file_format (str): Format of the saved data
            dataset_type (str): Type of dataset (train, test, etc.) - Only used for kaggle_brisT1D for now because its test data is not in csv format

        Returns:
            Dictionary with patient IDs as keys and DataFrames as values, or None if not found
            Note: For test data with nested structure, returns None to trigger custom loading
        """
        # Special handling for test data with nested structure - return None to trigger custom loading
        if (
            dataset_type == "test"
            and dataset_name == DatasetSourceType.KAGGLE_BRIS_T1D.value
        ):
            return None

        # We do the split at the code level not the cache level so we no longer need dataset_type here.
        processed_path = self.get_absolute_path_by_type(dataset_name, "processed")

        if file_format == "csv":
            if processed_path.exists():
                result = {}
                # Get all CSV files in the directory
                csv_files = [f for f in processed_path.iterdir() if f.suffix == ".csv"]

                for csv_file in csv_files:
                    # Extract patient ID from filename: remove _{dataset_type}.csv suffix
                    filename = csv_file.stem  # filename without extension
                    suffix_to_remove = "_full"

                    if filename.endswith(suffix_to_remove):
                        patient_id = filename[: -len(suffix_to_remove)]
                        # Load the CSV with datetime index
                        df = pd.read_csv(
                            csv_file,
                            index_col="datetime",
                            parse_dates=True,
                            low_memory=False,  # This solved the mixed types warning but not sure it is gonna cause some memory issues.
                        )
                        result[patient_id] = df

                return result if result else None

        return None

    def load_nested_test_data(
        self, dataset_name: str, dataset_type: str
    ) -> dict[str, dict[str, pd.DataFrame]] | None:
        """
        Load nested test data from compressed pickle file.

        Args:
            dataset_name (str): Name of the dataset
            dataset_type (str): Type of dataset (should be "test")

        Returns:
            Nested dictionary {patient_id: {row_id: DataFrame}} or None if not found
        """
        import gzip
        import pickle

        processed_path = self.get_processed_data_path_for_type(
            dataset_name, dataset_type
        )

        nested_data_file = processed_path / "nested_test_data.pkl.gz"

        if nested_data_file.exists():
            try:
                with gzip.open(nested_data_file, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                logger.error(f"Error loading nested test data: {e}")
                return None

        return None

    def nested_test_data_exists(self, dataset_name: str, dataset_type: str) -> bool:
        """
        Check if nested test data exists in cache.

        Args:
            dataset_name (str): Name of the dataset
            dataset_type (str): Type of dataset

        Returns:
            bool: True if nested test data exists, False otherwise
        """
        processed_path = self.get_processed_data_path_for_type(
            dataset_name, dataset_type
        )
        nested_data_file = processed_path / "nested_test_data.pkl.gz"
        return nested_data_file.exists()

    def clear_cache(self, dataset_name: Optional[str] = None):
        """
        Clear cache for a specific dataset or all datasets.

        Args:
            dataset_name (Optional[str]): Name of the dataset to clear, or None for all
        """
        if dataset_name:
            dataset_path = self.get_dataset_cache_path(dataset_name)
            if dataset_path.exists():
                shutil.rmtree(dataset_path)
                logger.info(f"Cleared cache for {dataset_name}")
        else:
            if self.cache_root.exists():
                shutil.rmtree(self.cache_root)
                self.cache_root.mkdir(parents=True, exist_ok=True)
                logger.info("Cleared all cache")


# Global cache manager instance
_cache_manager = None


def get_cache_manager() -> CacheManager:
    """
    Get the global cache manager instance.

    Returns:
        CacheManager: The global cache manager instance
    """
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager
