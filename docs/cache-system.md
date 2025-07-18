# Centralized Cache System

The nocturnal project now uses a centralized cache system for managing dataset storage and retrieval. This system provides automatic data fetching from external sources and efficient caching of both raw and processed data.

## Overview

The cache system is designed to:
- Automatically fetch data from external sources (Kaggle, HuggingFace, etc.)
- Store data in a consistent directory structure
- Handle both raw and processed data caching
- Provide a unified interface for all datasets

## Cache Structure

The cache follows this directory structure:

```
cache/
└── data/
    ├── kaggle_brisT1D/
    │   ├── raw/
    │   │   ├── train.csv
    │   │   ├── test.csv
    │   │   └── sample_submission.csv
    │   └── processed/
    │       ├── train/
    │       │   └── train.csv
    │       └── test/
    │           ├── patient_001/
    │           │   ├── row_001.csv
    │           │   └── row_002.csv
    │           └── patient_002/
    │               └── row_003.csv
    ├── gluroo/
    │   ├── raw/
    │   │   └── TBD
    │   └── processed/
    │       └── TBD
    │
    └── simglucose/
        ├── raw/
        └── processed/
```

## Usage

### Basic Usage

```python
from src.data.datasets.data_loader import get_loader

# Load training data (will auto-fetch from Kaggle if not cached)
loader = get_loader(
    data_source_name="kaggle_brisT1D",
    dataset_type="train",
    use_cached=True,
    num_validation_days=20
)

# Access the data
train_data = loader.train_data
validation_data = loader.validation_data
```

### Loading Test Data

```python
# Load test data (will use cached processed data if available)
test_loader = get_loader(
    data_source_name="kaggle_brisT1D",
    dataset_type="test",
    use_cached=True
)

# Test data is organized by patient and row
test_data = test_loader.processed_data
# Structure: {patient_id: {row_id: DataFrame}}
```

### Cache Management

```python
from src.data.cache_manager import get_cache_manager

cache_manager = get_cache_manager()

# Clear cache for a specific dataset
cache_manager.clear_cache("kaggle_brisT1D")

# Clear all cache
cache_manager.clear_cache()

# Get cache path for a dataset
cache_path = cache_manager.get_dataset_cache_path("kaggle_brisT1D")
```

## Automatic Data Fetching

The system automatically fetches data from external sources when needed:

### Kaggle Datasets

For Kaggle datasets, the system:
1. Checks if raw data exists in cache
2. If not, downloads from Kaggle using the Kaggle API
3. Extracts zip files and organizes the data
4. Saves to the cache for future use

**Prerequisites:**
- Kaggle API credentials configured (`~/.kaggle/kaggle.json`)
- Kaggle CLI installed (`pip install kaggle`)

### HuggingFace Datasets

For HuggingFace datasets, the system:
1. Checks if raw data exists in cache
2. If not, downloads using the HuggingFace datasets library
3. Saves to cache in the appropriate format

**Prerequisites:**
- HuggingFace datasets library (`pip install datasets`)

### Local Datasets

For local datasets, the system:
1. Checks if raw data exists in cache
2. If not, copies from the specified local path
3. Organizes the data in the cache structure

## Dataset Configuration

Dataset configurations are defined in `src/data/dataset_configs.py`:

```python
KAGGLE_BRIST1D_CONFIG = {
    "source": "kaggle",
    "competition_name": "brist1d",
    "required_files": ["train.csv", "test.csv", "sample_submission.csv"],
    "description": "Bristol Type 1 Diabetes dataset from Kaggle",
    "citation": "Bristol Type 1 Diabetes Dataset, Kaggle Competition",
    "url": "https://www.kaggle.com/competitions/brist1d",
}
```

### Adding New Datasets

To add a new dataset:

1. **Add configuration** in `src/data/dataset_configs.py`:
```python
NEW_DATASET_CONFIG = {
    "source": "kaggle",  # or "huggingface", "local"
    "competition_name": "your-competition",  # for Kaggle
    "required_files": ["file1.csv", "file2.csv"],
    "description": "Description of your dataset",
    "citation": "Citation information",
    "url": "Dataset URL",
}

DATASET_CONFIGS["new_dataset"] = NEW_DATASET_CONFIG
```

2. **Create data loader** following the pattern in `src/data/datasets/kaggle_bris_t1d/bris_t1d.py`

3. **Update imports** in `src/data/datasets/__init__.py`

4. **Update factory function** in `src/data/datasets/data_loader.py`

## Benefits

1. **User-Friendly**: Users don't need to know about file paths or data sources
2. **Automatic**: Data is fetched automatically when needed
3. **Efficient**: Caching prevents redundant downloads and processing
4. **Consistent**: All datasets follow the same interface and structure
5. **Extensible**: Easy to add new datasets and data sources

## Migration from Old System

The old system used local file paths within each dataset loader. The new system:

- Removes the need for `file_path` parameters
- Centralizes all caching logic
- Provides automatic data fetching
- Maintains the same API for existing code

### Before (Old System)
```python
loader = BrisT1DDataLoader(
    file_path="path/to/data.csv",
    use_cached=True
)
```

### After (New System)
```python
loader = get_loader(
    data_source_name="kaggle_brisT1D",
    use_cached=True
)
```

## Troubleshooting

### Kaggle API Issues
- Ensure Kaggle API credentials are set up correctly
- Check that the Kaggle CLI is installed and working
- Verify the competition name in the dataset configuration

### Cache Issues
- Clear the cache if data becomes corrupted: `cache_manager.clear_cache()`
- Check cache permissions and disk space
- Verify the cache directory structure

### Processing Issues
- Check that all required dependencies are installed
- Verify dataset configuration is correct
- Review logs for specific error messages
