# Data Documentation

## Overview
This document describes the data sources and standardized format used in our benchmark pipeline for nocturnal hypoglycemia prediction.

## Historical Context
Our benchmark pipeline was originally built on the Kaggle Bristol Type 1 Diabetes Dataset, which has influenced some of our naming conventions and data structures. This historical context helps explain certain design decisions:

- Column naming convention: The `-0:00` suffix in column names (e.g., `bg-0:00`) indicates current time measurements. While we could use `+1:00` for future measurements, this is not currently utilized in our pipeline.
- Time intervals: The Kaggle dataset combines data from different types of continuous glucose monitors (CGMs):
  - Dexcom users: 5-minute intervals
  - Libre users: 15-minute intervals
- The `time` column was originally used to distinguish between these different interval types in the Kaggle dataset

## Important Implementation Notes
- The `datetime` column serves as the index for data manipulation
- While models focus on row relationships rather than timestamps, maintaining consistent intervals is crucial
- Missing data must be either filled or imputed to ensure data quality
- These requirements apply across all supported datasets

## Supported Datasets

1. [Bristol Type 1 Diabetes Dataset](https://www.kaggle.com/competitions/brist1d)
   - A comprehensive dataset from Kaggle
   - Contains both Dexcom (5-min interval) and Libre (15-min interval) users
   - Multiple patients in a single CSV file (not that common)
   - prerequisite: Check out [Authentication](https://www.kaggle.com/docs/api) of the Kaggle API. We used CLI tool to fetch data from Kaggle.

2. Gluroo Dataset
   - Internal dataset from Christopher and Walker
   - Contains additional features like protein content and glucose trends
   - Includes meal announcements and intervention data

3. simglucose (Coming Soon)
   - Planned integration from benchmark repo as a package
   - Will provide simulated data for testing and validation

## Data Format Standardization

All datasets are transformed into a standardized format for our benchmark pipeline. The following sections detail the required columns and dataset-specific features.

### Core Columns (Required for All Datasets)

Index of the dataframe should just be an integer.

| Column | Type | Description | Source |
|--------|------|-------------|---------|
| `datetime` | `pd.Timestamp` | Primary timestamp for data manipulation | Created during processing |
| `p_num` | `str` | Patient identifier | Original dataset |
| `bg-0:00` | `float` | Blood glucose measurement in mg/dL (70-120) | Original dataset |
| `insulin-0:00` | `float` | Insulin dose in units | Original dataset |
| `carbs-0:00` | `float` | Carbohydrate intake in grams | Original dataset |
| `cob` | `float` | Carbohydrates on board in grams | Derived from `carbs-0:00` |
| `carb_availability` | `float` | Estimated total carbohydrates in blood | Derived from `carbs-0:00` |
| `iob` | `float` | Insulin on board in units | Derived from `insulin-0:00` |
| `insulin_availability` | `float` | Insulin in plasma | Derived from `insulin-0:00` |

### Optional Activity Metrics
These columns are available but not typically used in statistical models:

| Column | Type | Description |
|--------|------|-------------|
| `cals-0:00` | `float` | Total calories burnt in last 5 minutes |
| `steps-0:00` | `float` | Number of steps taken |
| `hr-0:00` | `float` | Heart rate |

### Dataset-Specific Features

#### Kaggle Dataset
| Column | Type | Description |
|--------|------|-------------|
| `time` | `pd.Timestamp` | Time of day (HH:MM:SS) - Used to determine patient interval type |

#### Gluroo Dataset
| Column | Type | Description |
|--------|------|-------------|
| `food_protein` | `float` | Protein content in grams |
| `trend` | `str` | Glucose trend (e.g., "rising", "falling") from Dexcom/Libre |
| `food_g` | `float` | Food grams (converts to `carbs-0:00`) |
| `food_g_keep` | `float` | Original meal carbohydrate values (for tracking only) |
| `affects_fob` | `bool` | Food on board flag |
| `affects_iob` | `bool` | Insulin on board flag |
| `day_start_shift` | `int` | Day start definition for data processing |

## Message Types (Gluroo Dataset)
The Gluroo dataset includes three types of messages that are processed during data transformation:

1. `DOSE_INSULIN` - Records insulin administration events
2. `ANNOUNCE_MEAL` - Captures meal announcements and carbohydrate intake
3. `INTERVENTION_SNACK` - Tracks snack interventions

## Dataset Standardization Process

To add a new dataset to the pipeline, follow these steps:

### 1. Directory Structure
```
src/data/
├── datasets/
│ ├── init.py
│ ├── data_loader.py
│ ├── dataset_base.py
│ └── ${dataset_name}/
│ │ ├── init.py
│ │ ├── ${dataset_name}.py # Dataset implementation
│ │ ├── ${dataset_name}_cleaner.py #
│ │ ├── raw/ # Original data files
│ │ └── processed/ # Processed and cached data
```

### 2. Implementation Steps

#### 2.1 Create Data Loader
- Create a new directory: `src/data/datasets/${dataset_name}/`
- Create a new file: `src/data/datasets/${dataset_name}/${dataset_name}_dataset.py`
- Inherit from `DatasetBase` class
- Implement required abstract methods
- Add caching mechanism to avoid reprocessing raw data
- Raw data should be fetched via API of the host if available (We are thinking of just hosting a private HF datasets now)

```python
from src.data.datasets.dataset_base import DatasetBase

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

    def load_data(self):
        if self.raw_data is None:
            self.raw_data = self.load_raw()
        return self._process_raw_data()
```

#### 2.2 Data Cleaning and Transformation
Implement a data cleaner that performs the following steps in order:

1. **Column Standardization**
   - Map original column names to standardized format
   - Ensure all required columns are present
   - Ensure `datetime` column exists and is set as the DataFrame index
   - Convert data types as needed

2. **Data Cleaning**
   - Remove duplicate entries
   - Handle missing values
   - Validate data ranges using the _validate_data method
   - Example: See `clean_gluroo_data` for reference

3. **Time Series Processing**
   - Ensure regular time intervals in the data
      - Call `data_transforms.ensure_regular_time_intervals(cleaned_df)`
      - Handle timezone information consistently
      - Create missing rows to maintain consistent sampling frequency
      - Missing data will be imputed in the benchmark pipeline

4. **Derived Features**
   - Generate carbohydrate-related features:
     ```python
     data.physiological.carb_model.create_cob_and_carb_availability_cols(df)
     ```
   - Generate insulin-related features:
     ```python
     data.physiological.insulin_model.create_iob_and_ins_availability_cols(df)
     ```
   - Add any dataset-specific features needed.

### 3. Register the Dataset
- Update `src/data/datasets/__init__.py` to include your new dataset.
- Ensure the dataset can be loaded using the standard interface.
- See our documentation contains on article with examples of why we register our datasets.

### 4. Documentation
Document all the changes

### 5. Reference Implementation
See existing implementations in ``src/data/datasets/gluroo` and `src/data/datasets/kaggle_bris_t1d` for working examples.

## Future Improvements
- Rename `bg-0:00` to `bg_mgdl-0:00` to explicitly include units
- Evaluate the need for the `-0:00` suffix in column names
- Standardize time interval handling across all datasets
- Implement automated data validation checks
- Add data quality metrics reporting
- Create dataset-specific documentation templates
