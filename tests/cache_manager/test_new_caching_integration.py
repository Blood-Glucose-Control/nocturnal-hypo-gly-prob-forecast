"""
Integration test for the new caching system.

This test verifies that the updated cache manager and BrisT1D dataset work correctly
with the new approach of storing processed data before splits and serialized train/val data.
"""

import tempfile
import shutil
import pandas as pd
import pytest

from src.data.cache_manager import CacheManager


class TestNewCachingIntegration:
    """Integration tests for the new caching system."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_manager = CacheManager(cache_root=self.temp_dir)

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_full_caching_workflow(self):
        """Test the complete workflow: save full processed data -> split -> cache splits."""
        
        # Mock dataset data
        mock_processed_data = {
            "patient_001": pd.DataFrame(
                {
                    "glucose": [100, 110, 120, 130, 140],
                    "datetime": pd.date_range("2023-01-01", periods=5, freq="5min"),
                }
            ).set_index("datetime"),
            "patient_002": pd.DataFrame(
                {
                    "glucose": [90, 95, 100, 105, 110],
                    "datetime": pd.date_range("2023-01-01", periods=5, freq="5min"),
                }
            ).set_index("datetime"),
        }

        # Step 1: Save full processed data
        self.cache_manager.save_full_processed_data("test_dataset", mock_processed_data)
        
        # Verify full processed data can be loaded
        loaded_full_data = self.cache_manager.load_full_processed_data("test_dataset")
        assert loaded_full_data is not None
        assert len(loaded_full_data) == 2
        assert "patient_001" in loaded_full_data
        assert "patient_002" in loaded_full_data

        # Step 2: Create train/validation splits
        train_data = {}
        validation_data = {}
        
        for patient_id, patient_df in mock_processed_data.items():
            # Simple split: first 3 rows for train, last 2 for validation
            train_data[patient_id] = patient_df.iloc[:3]
            validation_data[patient_id] = patient_df.iloc[3:]

        split_params = {
            "num_validation_days": 2,
            "split_method": "simple_split",
            "dataset_type": "train"
        }

        # Step 3: Save split data
        self.cache_manager.save_split_data("test_dataset", train_data, validation_data, split_params)

        # Step 4: Load split data back
        loaded_split_result = self.cache_manager.load_split_data("test_dataset", split_params)
        assert loaded_split_result is not None
        
        loaded_train, loaded_val = loaded_split_result
        assert len(loaded_train) == 2
        assert len(loaded_val) == 2
        
        # Verify data integrity
        for patient_id in mock_processed_data.keys():
            assert patient_id in loaded_train
            assert patient_id in loaded_val
            pd.testing.assert_frame_equal(train_data[patient_id], loaded_train[patient_id])
            pd.testing.assert_frame_equal(validation_data[patient_id], loaded_val[patient_id])

        # Step 5: Test different split parameters don't load existing cache
        different_params = {
            "num_validation_days": 3,
            "split_method": "simple_split", 
            "dataset_type": "train"
        }
        result_different = self.cache_manager.load_split_data("test_dataset", different_params)
        assert result_different is None

    def test_split_id_generation(self):
        """Test that split ID generation creates unique IDs for different parameters."""
        params_1 = {"num_validation_days": 2, "random_state": 42}
        params_2 = {"num_validation_days": 3, "random_state": 42}
        params_3 = {"num_validation_days": 2, "random_state": 123}
        
        id_1 = self.cache_manager._get_split_id(params_1)
        id_2 = self.cache_manager._get_split_id(params_2)
        id_3 = self.cache_manager._get_split_id(params_3)
        
        # All IDs should be different
        assert id_1 != id_2
        assert id_1 != id_3
        assert id_2 != id_3
        
        # Same parameters should generate same ID
        id_1_repeat = self.cache_manager._get_split_id(params_1)
        assert id_1 == id_1_repeat

    def test_cache_manager_new_methods_work(self):
        """Test that the new cache manager methods work correctly."""
        
        # Test data
        processed_data = {
            "patient_001": pd.DataFrame(
                {
                    "glucose": [100, 110, 120, 130, 140],
                    "datetime": pd.date_range("2023-01-01", periods=5, freq="1h"),
                }
            ).set_index("datetime")
        }
        
        # Test full processed data saving/loading
        self.cache_manager.save_full_processed_data("test_dataset", processed_data)
        loaded_data = self.cache_manager.load_full_processed_data("test_dataset")
        
        assert loaded_data is not None
        assert "patient_001" in loaded_data
        pd.testing.assert_frame_equal(processed_data["patient_001"], loaded_data["patient_001"])
        
        # Test split data
        train_data = {"patient_001": processed_data["patient_001"].iloc[:3]}
        val_data = {"patient_001": processed_data["patient_001"].iloc[3:]}
        split_params = {"validation_days": 2, "method": "test"}
        
        self.cache_manager.save_split_data("test_dataset", train_data, val_data, split_params)
        loaded_split = self.cache_manager.load_split_data("test_dataset", split_params)
        
        assert loaded_split is not None
        loaded_train, loaded_val = loaded_split
        
        pd.testing.assert_frame_equal(train_data["patient_001"], loaded_train["patient_001"])
        pd.testing.assert_frame_equal(val_data["patient_001"], loaded_val["patient_001"])

if __name__ == "__main__":
    pytest.main([__file__])