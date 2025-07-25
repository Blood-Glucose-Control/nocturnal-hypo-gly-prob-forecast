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

import subprocess
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
import logging
import pandas as pd

logger = logging.getLogger(__name__)


def get_project_root() -> Path:
    """
    Get the project root directory.

    This function looks for setup.py or pyproject.toml to identify the project root.

    Returns:
        Path: Path to the project root directory

    Raises:
        FileNotFoundError: If project root cannot be determined
    """
    current_path = Path.cwd()

    # Walk up the directory tree looking for project root indicators
    for path in [current_path] + list(current_path.parents):
        if (path / "setup.py").exists() or (path / "pyproject.toml").exists():
            return path

    # Fallback: if we can't find project root, use current directory
    logger.warning("Could not determine project root, using current directory")
    return current_path


class CacheManager:
    """
    Centralized cache manager for dataset storage and retrieval.

    This class manages a unified cache structure for all datasets, handling
    automatic data fetching from external sources and organizing data in a
    consistent directory structure.
    """

    def __init__(self, cache_root: str = "cache/data"):
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
            dataset_name (str): Name of the dataset

        Returns:
            Path: Path to the dataset's cache directory
        """
        return self.cache_root / dataset_name

    def get_raw_data_path(self, dataset_name: str) -> Path:
        """
        Get the raw data path for a specific dataset.

        Args:
            dataset_name (str): Name of the dataset

        Returns:
            Path: Path to the raw data directory
        """
        return self.get_dataset_cache_path(dataset_name) / "raw"

    def get_processed_data_path(self, dataset_name: str) -> Path:
        """
        Get the processed data path for a specific dataset.

        Args:
            dataset_name (str): Name of the dataset

        Returns:
            Path: Path to the processed data directory
        """
        return self.get_dataset_cache_path(dataset_name) / "processed"

    def ensure_raw_data(
        self, dataset_name: str, dataset_config: Dict[str, Any]
    ) -> Path:
        """
        Ensure raw data is available, fetching it if necessary.

        Args:
            dataset_name (str): Name of the dataset
            dataset_config (Dict[str, Any]): Configuration for the dataset

        Returns:
            Path: Path to the raw data directory

        Raises:
            ValueError: If dataset source is not supported
            RuntimeError: If data fetching fails
        """
        raw_path = self.get_raw_data_path(dataset_name)

        # Check if raw data already exists
        if self._raw_data_exists(raw_path, dataset_config):
            logger.info(f"Raw data for {dataset_name} already exists in cache")
            return raw_path

        # Fetch data from source
        logger.info(
            f"Raw data for {dataset_name} not found in cache, fetching from source"
        )
        source = dataset_config.get("source", "unknown")
        if source == "kaggle":
            self._fetch_kaggle_data(dataset_name, raw_path, dataset_config)
        elif source == "huggingface":
            self._fetch_huggingface_data(dataset_name, raw_path, dataset_config)
        elif source == "local":
            self._copy_local_data(dataset_name, raw_path, dataset_config)
        else:
            raise ValueError(f"Unsupported data source: {source}")

        return raw_path

    def _raw_data_exists(self, raw_path: Path, dataset_config: Dict[str, Any]) -> bool:
        """
        Check if raw data exists and is valid.

        Args:
            raw_path (Path): Path to raw data directory
            dataset_config (Dict[str, Any]): Dataset configuration

        Returns:
            bool: True if raw data exists and is valid
        """
        if not raw_path.exists():
            return False

        # Check for required files based on dataset type
        required_files = dataset_config.get("required_files", [])
        if required_files:
            return all((raw_path / file).exists() for file in required_files)

        # If no specific files required, check if directory has any files
        return any(raw_path.iterdir())

    def _fetch_kaggle_data(
        self, dataset_name: str, raw_path: Path, dataset_config: Dict[str, Any]
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

        competition_name = dataset_config.get("competition_name")
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

    def _fetch_huggingface_data(
        self, dataset_name: str, raw_path: Path, dataset_config: Dict[str, Any]
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

        dataset_id = dataset_config.get("dataset_id")
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

    def _copy_local_data(
        self, dataset_name: str, raw_path: Path, dataset_config: Dict[str, Any]
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
        source_path = dataset_config.get("source_path")
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

    def save_processed_data(
        self, dataset_name: str, dataset_type: str, data, file_format: str = "csv"
    ):
        """
        Save processed data to cache. This function assume nothing about the index so index=False.

        Args:
            dataset_name (str): Name of the dataset
            dataset_type (str): Type of dataset (train, test, etc.)
            data: Data to save (DataFrame, dict, etc.)
            file_format (str): Format to save the data in
        """
        processed_path = self.get_processed_data_path_for_type(
            dataset_name, dataset_type
        )
        processed_path.mkdir(parents=True, exist_ok=True)

        if file_format == "csv":
            if hasattr(data, "to_csv"):
                data.to_csv(processed_path / f"{dataset_type}.csv", index=False)
            else:
                raise ValueError(f"Cannot save data of type {type(data)} as CSV")
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

        logger.info(f"Saved processed {dataset_type} data for {dataset_name}")

    def load_processed_data(
        self, dataset_name: str, dataset_type: str, file_format: str = "csv"
    ):
        """
        Load processed data from cache.

        Args:
            dataset_name (str): Name of the dataset
            dataset_type (str): Type of dataset (train, test, etc.)
            file_format (str): Format of the saved data

        Returns:
            Loaded data or None if not found
        """
        processed_path = self.get_processed_data_path_for_type(
            dataset_name, dataset_type
        )

        if file_format == "csv":
            csv_file = processed_path / f"{dataset_type}.csv"
            if csv_file.exists():
                return pd.read_csv(csv_file)

        return None

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
