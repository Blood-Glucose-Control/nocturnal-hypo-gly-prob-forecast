!!! warning "Under Construction"
    This documentation is currently under active development and subject to change.
    Some sections may be incomplete or missing.

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

2. Gluroo Dataset
      - Internal dataset from Christopher and Walker
      - Contains additional features like protein content and glucose trends
      - Includes meal announcements and intervention data

3. simglucose (Coming Soon)
      - Planned integration from benchmark repo as a package
      - Will provide simulated data for testing and validation
4. Lynch 2022 Dataset
5. Aleppo Dataset

## Data Format Standardization

All datasets are transformed into a standardized format for our benchmark pipeline. The following sections detail the required columns and dataset-specific features.

### Core Columns (Required for All Datasets)

| Column                 | Type                   | Description                                 | Source                      | Required?              |
| ---------------------- | ---------------------- | ------------------------------------------- | --------------------------- | ---------------------- |
| `datetime`             | `pd.Timestamp` (INDEX) | Primary timestamp for data manipulation     | Created during processing   | Required               |
| `p_num`                | `str`                  | Patient identifier                          | Original dataset            | Required               |
| `bg-0:00`              | `float`                | Blood glucose measurement in mg/dL (70-120) | Original dataset            | Required               |
| `insulin-0:00`         | `float`                | Insulin dose in units                       | Original dataset            | Preferred              |
| `carbs-0:00`           | `float`                | Carbohydrate intake in grams                | Original dataset            | Preferred              |
| `cob`                  | `float`                | Carbohydrates on board in grams             | Derived from `carbs-0:00`   | Preferred              |
| `carb_availability`    | `float`                | Estimated total carbohydrates in blood      | Derived from `carbs-0:00`   | Preferred              |
| `iob`                  | `float`                | Insulin on board in units                   | Derived from `insulin-0:00` | Preferred              |
| `insulin_availability` | `float`                | Insulin in plasma                           | Derived from `insulin-0:00` | Preferred              |

### Optional Activity Metrics
These columns are available but not typically used in statistical models:

| Column       | Type    | Description                            |
| ------------ | ------- | -------------------------------------- |
| `cals-0:00`  | `float` | Total calories burnt in last 5 minutes |
| `steps-0:00` | `float` | Number of steps taken                  |
| `hr-0:00`    | `float` | Heart rate                             |

### Dataset-Specific Features

#### Kaggle Dataset
| Column | Type           | Description                                                      |
| ------ | -------------- | ---------------------------------------------------------------- |
| `time` | `pd.Timestamp` | Time of day (HH:MM:SS) - Used to determine patient interval type |

#### Gluroo Dataset
| Column            | Type    | Description                                                 |
| ----------------- | ------- | ----------------------------------------------------------- |
| `food_protein`    | `float` | Protein content in grams                                    |
| `trend`           | `str`   | Glucose trend (e.g., "rising", "falling") from Dexcom/Libre |
| `food_g`          | `float` | Food grams (converts to `carbs-0:00`)                       |
| `food_g_keep`     | `float` | Original meal carbohydrate values (for tracking only)       |
| `affects_fob`     | `bool`  | Food on board flag                                          |
| `affects_iob`     | `bool`  | Insulin on board flag                                       |
| `day_start_shift` | `int`   | Day start definition for data processing                    |

#### Lynch Dataset

#### Aleppo Dataset

## Message Types (Gluroo Dataset)
The Gluroo dataset includes three types of messages that are processed during data transformation:

1. `DOSE_INSULIN` - Records insulin administration events
2. `ANNOUNCE_MEAL` - Captures meal announcements and carbohydrate intake
3. `INTERVENTION_SNACK` - Tracks snack interventions

## Dataset Standardization Process

To add a new dataset to the pipeline, follow these steps:

### 1. Directory Structure
```
cache/data/{dataset_name}/
│   ├── raw/                    # Original data files
│   |   └── .gitkeep
│   └── processed/              # Processed and cached data
|   |   └── .gitkeep
src/data/diabetes_datasets/
├── ${dataset_name}/
│   ├── __init__.py             # Data initialization
│   ├── ${dataset_name}.py      # Data loader class
│   ├── data_cleaner.py         # Data cleaning functions specific ONLY to this dataset
│   └── README.md               # Instructions on how to access raw data, and where it should go.
```

### 2. Implementation Steps

#### 2.1 Create Data Loader
- Create a new file: `src/data/diabetes_datasets/${dataset_name}/${dataset_name}.py`
- Inherit from `DatasetBase` class
- Implement caching mechanism to avoid reprocessing
- Raw data should be fetchd via API of the host if available (We are thinking of just hosting a private HF datasets now)
```python
class ${DatasetName}Loader(DatasetBase):
    def __init__(self, cache=True):
        super().__init__()
        self.cache = cache
        # Implementation details...
```

#### 2.2 Data Cleaning and Transformation
Implement a data cleaner that performs the following steps in order:

1. **Column Standardization**
      - Map original column names to standardized format
      - Ensure all required columns are present
      - Ensure `datatime` column exists
      - Convert data types as needed

2. **Data Cleaning**
      - Remove duplicate entries
      - Handle missing values
      - Validate data ranges
      - Example: See `clean_gluroo_data` for reference

3. **Time Series Processing**
      - Call `data_transforms.ensure_regular_time_intervals(cleaned_df)`
      - This creates missing rows to maintain consistent intervals
      - Missing data will be imputed in the benchmark pipeline

4. **Derived Features**
      - Generate carbohydrate-related features:
         ```python
         data_transforms.create_cob_and_carb_availability_cols(df)
         ```
      - Generate insulin-related features:
         ```python
         data_transforms.create_iob_and_ins_availability_cols(df)
         ```
#### 2.3 Batch Processing Script
Write a shell script for WATGPU:
```
scripts/watgpu/data_processing_scripts/{datasets}_data_processing.sh
```
Make sure the partition you're asking for isn't going to request too many resources.
Check [SLURM documentation ](https://slurm.schedmd.com/pdfs/summary.pdf) for more details.
e.g., `sinfo`, `squeue`, `scontrol show node watgpu608` etc.

Script template:
```
#!/bin/bash

#SBATCH --job-name="{dataset}_data_processing"
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=30
#SBATCH --mem-per-cpu=1GB
#SBATCH --partition=HI
#SBATCH -o results/runs/{dataset}_data_processing/slurm-%j.out
#SBATCH -e results/runs/{dataset}_data_processing/slurm-%j.err
#SBATCH --mail-user={your_email@domain.com}
#SBATCH --mail-type=ALL

# Activate the virtual environment
source $HOME/nocturnal/.noctprob-venv/bin/activate


# Inline Python code to process the aleppo data (not the best practice but the task is simple enough)
echo "Starting {dataset} data processing"
python -c "
from src.data.diabetes_datasets.data_loader import get_loader
loader = get_loader(
    data_source_name='{dataset}',
    use_cached=False,
    parallel=True,
    max_workers=30,
)
"
echo "{dataset} data processing completed"

## Run sbatch {dataset}_data_processing.sh
```

### 3. Documentation
Document all the changes

### 4. Reference Implementation
See `src/data/gluroo/` for a working example of dataset implementation.

## Future Improvements
   - Rename `bg-0:00` to `bg_mgdl-0:00` to explicitly include units
   - Evaluate the need for the `-0:00` suffix in column names
   - Standardize time interval handling across all datasets
   - Implement automated data validation checks
   - Add data quality metrics reporting
   - Create dataset-specific documentation templates
