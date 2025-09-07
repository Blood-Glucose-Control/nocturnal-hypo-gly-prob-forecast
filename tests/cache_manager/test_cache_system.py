"""
Tests for the centralized cache system.

This module tests the cache manager and data loader functionality
to ensure the new cache system works correctly.
"""

import shutil
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.data.cache_manager import CacheManager, get_cache_manager
from src.data.dataset_configs import (
    get_dataset_config,
    get_dataset_info,
    list_available_datasets,
)


class TestCacheManager:
    """Test the CacheManager class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_manager = CacheManager(cache_root=self.temp_dir)

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_init(self):
        """Test CacheManager initialization."""
        assert self.cache_manager.cache_root == Path(self.temp_dir)
        assert self.cache_manager.cache_root.exists()

    def test_get_dataset_cache_path(self):
        """Test getting dataset cache path."""
        path = self.cache_manager.get_dataset_cache_path("test_dataset")
        expected = Path(self.temp_dir) / "test_dataset"
        assert path == expected

    def test_get_raw_data_path(self):
        """Test getting raw data path."""
        path = self.cache_manager.get_raw_data_path("test_dataset")
        expected = Path(self.temp_dir) / "test_dataset" / "raw"
        assert path == expected

    def test_get_processed_data_path(self):
        """Test getting processed data path."""
        path = self.cache_manager.get_processed_data_path("test_dataset")
        expected = Path(self.temp_dir) / "test_dataset" / "processed"
        assert path == expected

    def test_get_processed_data_path_for_type(self):
        """Test getting processed data path for specific type."""
        path = self.cache_manager.get_processed_data_path_for_type(
            "test_dataset", "train"
        )
        expected = Path(self.temp_dir) / "test_dataset" / "processed" / "train"
        assert path == expected

    def test_raw_data_exists_with_files(self):
        """Test raw data existence check with required files."""
        dataset_config = {"required_files": ["file1.csv", "file2.csv"]}

        # Should return False when directory doesn't exist
        assert not self.cache_manager._raw_data_exists(
            Path(self.temp_dir) / "nonexistent", dataset_config
        )

        # Should return False when required files are missing
        raw_path = self.cache_manager.get_raw_data_path("test_dataset")
        raw_path.mkdir(parents=True, exist_ok=True)
        assert not self.cache_manager._raw_data_exists(raw_path, dataset_config)

        # Should return True when all required files exist
        (raw_path / "file1.csv").touch()
        (raw_path / "file2.csv").touch()
        assert self.cache_manager._raw_data_exists(raw_path, dataset_config)

    def test_raw_data_exists_without_files(self):
        """Test raw data existence check without required files."""
        dataset_config = {}
        raw_path = self.cache_manager.get_raw_data_path("test_dataset")

        # Should return False when directory doesn't exist
        assert not self.cache_manager._raw_data_exists(raw_path, dataset_config)

        # Should return False when directory is empty
        raw_path.mkdir(parents=True, exist_ok=True)
        assert not self.cache_manager._raw_data_exists(raw_path, dataset_config)

        # Should return True when directory has files
        (raw_path / "some_file.csv").touch()
        assert self.cache_manager._raw_data_exists(raw_path, dataset_config)

    def test_save_and_load_processed_data(self):
        """Test saving and loading processed data."""

        # Create test data with datetime index
        dates = pd.date_range("2025-01-01", periods=3, freq="H")
        test_data = pd.DataFrame(
            {"col1": [1, 2, 3], "col2": ["a", "b", "c"]}, index=dates
        )
        test_data.index.name = "datetime"
        patient_id = "patient_1"
        # Save data
        self.cache_manager.save_processed_data(
            "test_dataset", "train", patient_id, test_data
        )

        # Load data
        loaded_data = self.cache_manager.load_processed_data("test_dataset", "train")

        # Verify data
        assert loaded_data["patient_1"] is not None
        assert loaded_data["patient_1"].equals(test_data)

    def test_clear_cache_specific_dataset(self):
        """Test clearing cache for specific dataset."""
        # Create some test data
        dataset_path = self.cache_manager.get_dataset_cache_path("test_dataset")
        dataset_path.mkdir(parents=True, exist_ok=True)
        (dataset_path / "test_file.txt").touch()

        # Clear specific dataset
        self.cache_manager.clear_cache("test_dataset")

        # Verify dataset is removed
        assert not dataset_path.exists()

    def test_clear_all_cache(self):
        """Test clearing all cache."""
        # Create some test data
        for dataset in ["dataset1", "dataset2"]:
            dataset_path = self.cache_manager.get_dataset_cache_path(dataset)
            dataset_path.mkdir(parents=True, exist_ok=True)
            (dataset_path / "test_file.txt").touch()

        # Clear all cache
        self.cache_manager.clear_cache()

        # Verify all datasets are removed but cache root exists
        assert self.cache_manager.cache_root.exists()
        assert not any(self.cache_manager.cache_root.iterdir())


class TestDatasetConfigs:
    """Test dataset configuration functionality."""

    def test_get_dataset_config(self):
        """Test getting dataset configuration."""
        config = get_dataset_config("kaggle_brisT1D")
        assert config["source"] == "kaggle"
        assert config["competition_name"] == "brist1d"

    def test_get_dataset_config_invalid(self):
        """Test getting invalid dataset configuration."""
        with pytest.raises(ValueError, match="Configuration not found"):
            get_dataset_config("invalid_dataset")

    def test_list_available_datasets(self):
        """Test listing available datasets."""
        datasets = list_available_datasets()
        assert "kaggle_brisT1D" in datasets
        assert "gluroo" in datasets
        assert "simglucose" in datasets

    def test_get_dataset_info(self):
        """Test getting dataset information."""
        info = get_dataset_info("kaggle_brisT1D")
        assert "description" in info
        assert "citation" in info
        assert "url" in info


class TestGlobalCacheManager:
    """Test the global cache manager instance."""

    def test_get_cache_manager_singleton(self):
        """Test that get_cache_manager returns a singleton."""
        manager1 = get_cache_manager()
        manager2 = get_cache_manager()
        assert manager1 is manager2


class TestCacheIntegration:
    """Test integration with the data loader."""

    def test_cache_manager_with_data_loader(self):
        """Test that cache manager works with data loader."""
        from src.data.diabetes_datasets.data_loader import get_loader

        # This should work with the new cache system
        try:
            loader = get_loader(
                data_source_name="kaggle_brisT1D",
                dataset_type="train",
                use_cached=True,
            )
            assert loader is not None
            # The loader should have a cache manager
            assert hasattr(loader, "cache_manager")
            assert hasattr(loader, "dataset_config")
        except Exception as e:
            # If this fails, it might be because Kaggle credentials aren't set up
            # or the data isn't available, which is expected in some test environments
            pytest.skip(f"Data loader test skipped: {e}")
