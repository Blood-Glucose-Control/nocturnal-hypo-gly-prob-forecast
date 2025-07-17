"""
Example script demonstrating the new centralized cache system.

This script shows how to use the refactored data loaders with automatic
data fetching and centralized caching.
"""

import logging
from src.data.datasets.data_loader import get_loader
from src.data.cache_manager import get_cache_manager

# Set up logging to see what's happening
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """
    Demonstrate the new cache system functionality.
    """
    print("=== Centralized Cache System Demo ===\n")

    # Get cache manager to show cache structure
    cache_manager = get_cache_manager()
    print(f"Cache root directory: {cache_manager.cache_root}\n")

    # Example 1: Load Kaggle Bristol T1D dataset (will auto-fetch if not available)
    print("1. Loading Kaggle Bristol T1D dataset...")
    try:
        # This will automatically fetch data from Kaggle if not in cache
        loader = get_loader(
            data_source_name="kaggle_brisT1D",
            dataset_type="train",
            use_cached=True,
            num_validation_days=20,
        )

        print("   ✓ Dataset loaded successfully!")
        print(f"   ✓ Dataset name: {loader.dataset_name}")
        print(f"   ✓ Training data shape: {loader.train_data.shape}")
        print(f"   ✓ Validation data shape: {loader.validation_data.shape}")
        print(f"   ✓ Number of training days: {loader.num_train_days}")

    except Exception as e:
        print(f"   ✗ Error loading dataset: {e}")
        print("   Note: Make sure you have Kaggle API credentials set up")

    print()

    # Example 2: Load test data (will use cached processed data if available)
    print("2. Loading test data...")
    try:
        test_loader = get_loader(
            data_source_name="kaggle_brisT1D", dataset_type="test", use_cached=True
        )

        print("   ✓ Test data loaded successfully!")
        print(f"   ✓ Number of patients: {len(test_loader.processed_data)}")

        # Show first patient's data
        first_patient = list(test_loader.processed_data.keys())[0]
        first_patient_data = test_loader.processed_data[first_patient]
        print(
            f"   ✓ First patient ({first_patient}) has {len(first_patient_data)} test rows"
        )

    except Exception as e:
        print(f"   ✗ Error loading test data: {e}")

    print()

    # Example 3: Show cache structure
    print("3. Cache directory structure:")
    try:
        from pathlib import Path

        cache_root = Path("cache/data")
        if cache_root.exists():
            for dataset_dir in cache_root.iterdir():
                if dataset_dir.is_dir():
                    print(f"   📁 {dataset_dir.name}/")
                    for subdir in dataset_dir.iterdir():
                        if subdir.is_dir():
                            print(f"      📁 {subdir.name}/")
                            if subdir.name == "Raw":
                                for file in subdir.iterdir():
                                    if file.is_file():
                                        print(f"         📄 {file.name}")
                            elif subdir.name == "Processed":
                                for type_dir in subdir.iterdir():
                                    if type_dir.is_dir():
                                        print(f"         📁 {type_dir.name}/")
                                        if type_dir.name == "test":
                                            # Count patient directories
                                            patient_count = len(
                                                [
                                                    d
                                                    for d in type_dir.iterdir()
                                                    if d.is_dir()
                                                ]
                                            )
                                            print(
                                                f"            📁 {patient_count} patient directories"
                                            )
        else:
            print("   No cache directory found yet")

    except Exception as e:
        print(f"   ✗ Error exploring cache: {e}")

    print()

    # Example 4: Cache management
    print("4. Cache management options:")
    print("   - Clear specific dataset: cache_manager.clear_cache('kaggle_brisT1D')")
    print("   - Clear all cache: cache_manager.clear_cache()")
    print(
        "   - Check cache info: cache_manager.get_dataset_cache_path('kaggle_brisT1D')"
    )

    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    main()
