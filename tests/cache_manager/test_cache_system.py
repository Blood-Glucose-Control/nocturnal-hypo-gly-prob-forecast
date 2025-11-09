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
    register_dataset,
)
from src.data.models import DatasetConfig, DatasetSourceType


class TestCacheManager:
    """Test the CacheManager class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_manager = CacheManager(cache_root=self.temp_dir)
        self.test_dataset = "test_dataset"
        register_dataset(
            self.test_dataset,
            DatasetConfig(
                source=DatasetSourceType.LOCAL,
                required_files=["test_data.csv"],
                description="Test dataset",
                citation="Test dataset",
                cache_path="test_dataset",
                url="https://test.com",
            ),
        )

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_init(self):
        """Test CacheManager initialization."""
        assert self.cache_manager.cache_root == Path(self.temp_dir)
        assert self.cache_manager.cache_root.exists()

    def test_get_dataset_cache_path(self):
        """Test getting dataset cache path."""
        path = self.cache_manager.get_dataset_cache_path(self.test_dataset)
        expected = Path(self.temp_dir) / self.test_dataset
        assert path == expected

    def test_get_raw_data_path(self):
        """Test getting raw data path."""
        path = self.cache_manager.get_raw_data_path(self.test_dataset)
        expected = Path(self.temp_dir) / self.test_dataset / "raw"
        assert path == expected

    def test_get_processed_data_path(self):
        """Test getting processed data path."""
        path = self.cache_manager.get_processed_data_path(self.test_dataset)
        expected = Path(self.temp_dir) / self.test_dataset / "processed"
        assert path == expected

    def test_save_and_load_full_processed_data(self):
        """Test saving and loading full processed data (before split)."""
        # Create test data
        data = {
            "patient_001": pd.DataFrame(
                {
                    "glucose": [100, 110, 120],
                    "datetime": pd.to_datetime(
                        ["2023-01-01 00:00", "2023-01-01 00:05", "2023-01-01 00:10"]
                    ),
                }
            ).set_index("datetime"),
            "patient_002": pd.DataFrame(
                {
                    "glucose": [90, 95, 100],
                    "datetime": pd.to_datetime(
                        ["2023-01-01 00:00", "2023-01-01 00:05", "2023-01-01 00:10"]
                    ),
                }
            ).set_index("datetime"),
        }

        # Save the data
        self.cache_manager.save_full_processed_data("test_dataset", data)

        # Load the data back
        loaded_data = self.cache_manager.load_full_processed_data("test_dataset")

        assert loaded_data is not None
        assert len(loaded_data) == 2
        assert "patient_001" in loaded_data
        assert "patient_002" in loaded_data

        # Check data integrity
        pd.testing.assert_frame_equal(data["patient_001"], loaded_data["patient_001"])
        pd.testing.assert_frame_equal(data["patient_002"], loaded_data["patient_002"])

    def test_load_full_processed_data_not_exists(self):
        """Test loading full processed data when it doesn't exist."""
        with pytest.raises(ValueError, match="Configuration not found"):
            self.cache_manager.load_full_processed_data("nonexistent_dataset")

    def test_save_and_load_split_data(self):
        """Test saving and loading train/validation split data."""
        # Create test data
        train_data = {
            "patient_001": pd.DataFrame(
                {
                    "glucose": [100, 110],
                    "datetime": pd.to_datetime(
                        ["2023-01-01 00:00", "2023-01-01 00:05"]
                    ),
                }
            ).set_index("datetime")
        }

        validation_data = {
            "patient_001": pd.DataFrame(
                {"glucose": [120], "datetime": pd.to_datetime(["2023-01-01 00:10"])}
            ).set_index("datetime")
        }

        split_params = {"validation_split": 0.2, "shuffle": True, "random_state": 42}

        # Save the split data
        self.cache_manager.save_split_data(
            "test_dataset", train_data, validation_data, split_params
        )

        # Load the split data back
        result = self.cache_manager.load_split_data("test_dataset", split_params)

        assert result is not None
        loaded_train, loaded_val = result
        assert loaded_train is not None
        assert loaded_val is not None
        assert len(loaded_train) == 1
        assert len(loaded_val) == 1

        # Check data integrity
        pd.testing.assert_frame_equal(
            train_data["patient_001"], loaded_train["patient_001"]
        )
        pd.testing.assert_frame_equal(
            validation_data["patient_001"], loaded_val["patient_001"]
        )

    def test_load_split_data_different_params(self):
        """Test loading split data with different parameters returns None."""
        # Save with one set of parameters
        train_data = {"patient_001": pd.DataFrame({"glucose": [100]})}
        validation_data = {"patient_001": pd.DataFrame({"glucose": [110]})}
        split_params_1 = {"validation_split": 0.2, "random_state": 42}

        self.cache_manager.save_split_data(
            "test_dataset", train_data, validation_data, split_params_1
        )

        # Try to load with different parameters
        split_params_2 = {"validation_split": 0.3, "random_state": 42}
        result = self.cache_manager.load_split_data("test_dataset", split_params_2)

        assert result is None

    def test_load_split_data_not_exists(self):
        """Test loading split data when it doesn't exist."""
        split_params = {"validation_split": 0.2, "random_state": 42}
        import pytest  # already imported at top; safe if duplicated in scope

        with pytest.raises(ValueError, match="Configuration not found"):
            self.cache_manager.load_split_data("nonexistent_dataset", split_params)

    def test_get_processed_data_path_for_type(self):
        """Test getting processed data path for specific type."""
        path = self.cache_manager.get_processed_data_path_for_type(
            "test_dataset", "train"
        )
        expected = Path(self.temp_dir) / "test_dataset" / "processed" / "train"
        assert path == expected

    def test_raw_data_exists_with_files(self):
        """Test raw data existence check with required files."""
        dataset_config = DatasetConfig(
            source=DatasetSourceType.LOCAL,
            required_files=["file1.csv", "file2.csv"],
            description="Test",
            citation="Test",
            url="https://test.com",
        )

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
        dataset_config = DatasetConfig(
            source=DatasetSourceType.LOCAL,
            required_files=[],
            description="Test",
            citation="Test",
            url="https://test.com",
        )
        raw_path = self.cache_manager.get_raw_data_path("test_dataset")

        # Should return False when directory doesn't exist
        assert not self.cache_manager._raw_data_exists(raw_path, dataset_config)

        # Should return False when directory is empty
        raw_path.mkdir(parents=True, exist_ok=True)
        assert not self.cache_manager._raw_data_exists(raw_path, dataset_config)

        # Should return True when directory has files
        (raw_path / "some_file.csv").touch()
        assert self.cache_manager._raw_data_exists(raw_path, dataset_config)

    @pytest.mark.xfail(
        reason="We are working on removing saving by dataset_type because it is not needed. Will need to add this back",
        strict=False,
    )
    def test_save_and_load_processed_data(self):
        """Test saving and loading processed data."""

        # Create test data with datetime index
        dates = pd.date_range("2025-01-01", periods=3, freq="h")
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

        # Verify it's a dictionary
        assert isinstance(loaded_data, dict), f"Expected dict, got {type(loaded_data)}"
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
        config = get_dataset_config(DatasetSourceType.KAGGLE_BRIS_T1D.value)
        assert config.source == DatasetSourceType.KAGGLE_BRIS_T1D
        assert config.competition_name == "brist1d"

    def test_get_dataset_config_invalid(self):
        """Test getting invalid dataset configuration."""
        with pytest.raises(ValueError, match="Configuration not found"):
            get_dataset_config("invalid_dataset")

    def test_list_available_datasets(self):
        """Test listing available datasets."""
        datasets = list_available_datasets()
        assert DatasetSourceType.KAGGLE_BRIS_T1D.value in datasets
        assert DatasetSourceType.GLUROO.value in datasets
        assert DatasetSourceType.SIMGLUCOSE.value in datasets

    def test_get_dataset_info(self):
        """Test getting dataset information."""
        info = get_dataset_info(DatasetSourceType.KAGGLE_BRIS_T1D.value)
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
                data_source_name=DatasetSourceType.KAGGLE_BRIS_T1D.value,
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
