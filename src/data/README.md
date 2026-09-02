# Data Documentation

## Overview
This document describes the data sources and standardized format used in our benchmark pipeline for nocturnal hypoglycemia prediction.

## Supported Datasets

1. Aleppo 2017
2. Brown 2019
3. Lynch 2022
4. Tamborlane 2008

## Data Format Standardization

All datasets are transformed into a standardized format for our benchmark pipeline. The following sections detail the required columns and dataset-specific features.

### Core Columns (Required for All Datasets)

The DataFrame index should be `datetime`.

| Column | Type | Description | Source | Required? |
|--------|------|-------------|---------|-----------|
| `datetime` | `pd.Timestamp` **INDEX** | Primary timestamp for each measurement | Mapped from `time` or original `datetime` | **Required** |
| `patient_id` | `str` | Patient identifier (e.g., "p01", "glu001") | Artificially created by dataset loader | **Required** |
| `bg_mM` | `float` | Blood glucose measurement in mmol/L | Original data source, converted from mg/dL if needed | **Required** |

### Optional Columns (Enhance Features but Don't Block Processing)

| Column | Type | Description | Source | Notes |
|--------|------|-------------|---------|-------|
| `dose_units` | `float` | Insulin dose in units | Original data source | Enables IOB calculation |
| `food_g` | `float` | Carbohydrate intake in grams | Original data source | Enables COB calculation |
| `msg_type` | `str` | Message type indicator ('ANNOUNCE_MEAL' or empty) | Derived from rows where `food_g` is not null | - |
| `rate` | `float` | Basal insulin rate in U/hr | Original data source | Enables basal rollover |
| `basal_duration_mins` | `int` | Duration of basal rate in minutes | Original data source or derived | Required for basal rollover along with `rate` and `dose_units` |

### Derived Physiological Features

These columns are computed by the preprocessing pipeline from the optional columns above. If the source column is missing or all NaN, these will be set to NaN.

| Column | Type | Description | Source |
|--------|------|-------------|---------|
| `cob` | `float` | Carbohydrates on board in grams | Derived from `food_g` using physiological model |
| `carb_availability` | `float` | Estimated total carbohydrates in blood | Derived from `food_g` using physiological model |
| `iob` | `float` | Insulin on board in units | Derived from `dose_units` using physiological model |
| `insulin_availability` | `float` | Insulin in plasma | Derived from `dose_units` using physiological model |

> **Note:** CGM-only datasets (e.g., some Type 2 diabetes patients, pre-training scenarios) are supported. The pipeline will skip COB/IOB calculations and set those features to NaN if the source columns are missing.

## Dataset Standardization Process

To add a new dataset to the pipeline, follow these steps:

A good example: `src\data\diabetes_datasets\brown_2019\brown_2019.py`

### 1. Fetch Raw Data
- If the data source can be fetched, use `cache_manager` to fetch the raw data
- If the data cannot be fetched via API, make sure documentation is included to inform users where to download and place the data in the repository
- Check out `cache-system.md` for more details

All data should be placed in the `cache` directory:
```
cache/data/
├── ${dataset_name}/
│   ├── processed/
│   │   └── patient_1.csv
│   └── raw/
│       └── patient_1.csv
```

### 2. Implementation Steps

**High-level overview:**
1. Create a data loader
2. Load raw data from cache
3. Process raw data
4. Save and return processed data

#### 2.1 Create Data Loader
- Create a new directory: `src/data/datasets/${dataset_name}/`
- Create a new file: `src/data/datasets/${dataset_name}/${dataset_name}_dataset.py`
- Inherit from `DatasetBase` class
- Implement required abstract methods
- Add caching mechanism to avoid reprocessing raw data (check out `cache-system.md`)

```python
from src.data.diabetes_datasets.dataset_base import DatasetBase

class ${DatasetName}Dataset(DatasetBase):
    def __init__(self, cache=True):
        super().__init__()
        self.cache = cache

    @property
    def dataset_name(self):
        return "${dataset_name}"

    def load_raw(self):
        # Implementation for loading raw data
        return raw_df

    def _process_raw_data(self):
        # Process raw data into standardized format
        # Return processed dataframe

    def _translate_raw_data(self):
        # Convert columns name and units

    def load_data(self):
        if self.raw_data is None:
            self.raw_data = self.load_raw()
        return self._process_raw_data()
```

#### General Flow
**Data Processing Pipeline:**
Raw data → Dataset-specific translation (renaming, unit conversion) → Data processing pipeline (`preprocessing_pipeline`) → Store processed data → Load processed data → Training pipeline → Evaluation stage

For local development, we stick with the default parameters for this data cleaning pipeline so everyone can have uniform cached processed data. For training development, we may want to store different sets of processed data to train and compare performance differences.

#### 2.2 Data Cleaning and Transformation
Implement a data cleaner that performs the following steps in order:

1. **Column Standardization**
   - Rename the dataset's blood glucose level column to `bg_mM` and convert to mmol/L
   - Convert the `time` column to `datetime` and set it as the index (if there is no time column, artificially create one starting from a random date)
   - Convert "food" rows to `food_g` and set `msg_type` to `ANNOUNCE_MEAL`
   - Convert "insulin" column to `dose_units` if available. Note that this column can contain small consecutive numbers for pump patients or discrete large numbers for non-pump patients
   - Add `patient_id`. If your dataset has multiple patients, give each a unique ID like "p01", "p02"
   - **Required columns**: `datetime`, `patient_id`, `bg_mM`
   - **Optional columns**: `dose_units`, `food_g`, `msg_type`, `rate` (enhance features but don't block processing)
   - Add `cals`, `steps`, `hr_bpm`, and `activity` if they exist
   - Ensure the `datetime` column exists and is set as the DataFrame index

2. **Cleaning Pipeline**
   - Call `preprocessing_pipeline`
   - This will clean up the data and add extra features like IOB and COB from the columns you just converted

### 3. Register the Dataset
- Update `src/data/datasets/__init__.py` to include your new dataset
- Ensure the dataset can be loaded using the standard interface
- See our documentation for examples of why we register our datasets

### 4. Documentation
Document all the changes

## Future Improvements
- Add `INTERVENTION_SNACK` message type to `preprocessing_pipeline`. We could use a threshold like < 10g = `INTERVENTION_SNACK`, >= 10g = `ANNOUNCE_MEAL`
- Double-check that all `-0:00` suffixes are removed
- Standardize time interval handling across all datasets
- Implement automated data validation checks (should be completed)
- Add data quality metrics reporting
- Create dataset-specific documentation templates
