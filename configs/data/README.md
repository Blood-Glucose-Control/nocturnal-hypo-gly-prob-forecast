# Data Holdout System

## Overview

This system ensures consistent and reproducible train/holdout data splits across all experiments. It supports both **temporal splits** (holding out the end of time series) and **patient-based splits** (holding out specific patients), as well as **hybrid** approaches combining both.

## Key Features

- **Reproducible Splits**: Fixed patient selections ensure every experiment uses the same train/holdout separation
- **Multiple Strategies**: Support for temporal, patient-based, and hybrid holdout approaches
- **Validation**: Built-in checks to prevent data leakage between train and holdout sets
- **Easy Integration**: Simple API for loading training-only or holdout-only data

## Directory Structure

```
configs/data/holdout/          # Holdout configuration files
├── kaggle_brisT1D.yaml       # Config for Kaggle dataset
├── gluroo.yaml               # Config for Gluroo dataset
├── aleppo.yaml               # Config for Aleppo dataset
└── ...                       # Other dataset configs

src/data/preprocessing/
├── holdout_config.py         # Configuration classes
├── holdout_manager.py        # Split implementation
└── ...

src/data/versioning/
├── dataset_registry.py       # Dataset loading with splits
└── ...

scripts/data_processing_scripts/
├── generate_holdout_configs.py   # Generate configs for all datasets
└── validate_holdout_configs.py   # Validate configs and splits
```

## Holdout Strategies

### 1. Temporal Holdout

Holds out a percentage of data at the **end** of each patient's time series.

**Use case**: Evaluating model performance on future predictions

**Example**: 20% of most recent data held out for testing

### 2. Patient-Based Holdout

Holds out **specific patients** entirely from training.

**Use case**: Evaluating model generalization to new patients

**Example**: 20% of patients (randomly selected but fixed) never used in training

### 3. Hybrid Holdout (Recommended)

Combines both strategies:
1. Hold out specific patients completely
2. For training patients, also hold out temporal portion

**Use case**: Comprehensive evaluation of both temporal prediction and patient generalization

## Quick Start

### 1. Generate Holdout Configurations

Run this **once** to generate holdout configurations for all datasets:

```bash
python scripts/data_processing_scripts/generate_holdout_configs.py
```

This creates configuration files in `configs/data/holdout/` with:
- Random patient selection (seed=42 for reproducibility)
- 20% temporal holdout on training patients
- 20% of patients as holdout patients

### 2. Load Training Data in Your Code

**Always use this method when training models:**

```python
from src.data.versioning.dataset_registry import load_training_data

# Load ONLY training data (holdout is excluded)
train_data = load_training_data("kaggle_brisT1D")

# Train your model on train_data
model.fit(train_data)
```

### 3. Load Holdout Data for Evaluation

**Use this ONLY for final model evaluation:**

```python
from src.data.versioning.dataset_registry import load_holdout_data

# Load ONLY holdout data for evaluation
holdout_data = load_holdout_data("kaggle_brisT1D")

# Evaluate your trained model
results = model.evaluate(holdout_data)
```

### 4. Load Both Splits

If you need both splits explicitly:

```python
from src.data.versioning.dataset_registry import load_split_data

train_data, holdout_data = load_split_data("kaggle_brisT1D")
```

## Validation

### Validate All Datasets

```bash
python scripts/data_processing_scripts/validate_holdout_configs.py
```

This checks:
- ✓ Configuration files exist
- ✓ Data loads successfully
- ✓ No patient overlap between train and holdout
- ✓ Temporal ordering is correct
- ✓ Minimum sample requirements met

### Validate Specific Dataset

```bash
python scripts/data_processing_scripts/validate_holdout_configs.py kaggle_brisT1D
```

## Configuration Format

Example holdout configuration (`configs/data/holdout/kaggle_brisT1D.yaml`):

```yaml
dataset_name: kaggle_brisT1D
holdout_type: hybrid
description: 'Hybrid holdout strategy: 20% temporal split on training patients + holdout patients (20% of total)'
created_date: '2026-01-09'
version: '1.0'

temporal_config:
  holdout_percentage: 0.2        # 20% of data at end of time series
  min_train_samples: 50          # Minimum samples per patient for training
  min_holdout_samples: 10        # Minimum samples per patient for holdout

patient_config:
  holdout_patients:              # These patients NEVER used for training
    - p04
    - p12
    - p18
  min_train_patients: 2          # Minimum patients needed for training
  min_holdout_patients: 1        # Minimum patients needed for holdout
  random_seed: 42                # For reproducibility
```

## Advanced Usage

### Custom Holdout Manager

For custom split logic:

```python
from src.data.versioning.holdout_config import HoldoutConfig
from src.data.versioning.holdout_manager import HoldoutManager

# Load configuration
config = HoldoutConfig.load("configs/data/holdout/kaggle_brisT1D.yaml")

# Create manager
manager = HoldoutManager(config)

# Apply split to your data
train_data, holdout_data = manager.split_data(
    data=your_dataframe,
    patient_col="p_num",
    time_col="time"
)

# Validate split
validation = manager.validate_split(train_data, holdout_data)
print(validation)  # {'no_overlap': True, 'train_not_empty': True, ...}
```

### Get Split Information

```python
from src.data.versioning.dataset_registry import get_dataset_registry

registry = get_dataset_registry()
info = registry.get_split_info("kaggle_brisT1D")

print(f"Holdout type: {info['holdout_type']}")
print(f"Holdout patients: {info['patient_split']['holdout_patients']}")
```

### List Available Datasets

```python
from src.data.versioning.dataset_registry import get_dataset_registry

registry = get_dataset_registry()
datasets = registry.list_available_datasets()
print(f"Datasets with holdout configs: {datasets}")
```

## Best Practices

### ✅ DO

- **Always use `load_training_data()`** when training models
- **Only use `load_holdout_data()`** for final evaluation
- **Validate configs** before running experiments
- **Document** which holdout config version was used in experiments
- **Keep holdout configs in version control** for reproducibility

### ❌ DON'T

- Don't train on holdout data
- Don't modify holdout configs after experiments start
- Don't use different patient splits across experiments
- Don't skip validation before running experiments

## Integration with Training Pipeline

Example training script:

```python
from src.data.versioning.dataset_registry import load_training_data, load_holdout_data
from src.models.ttm.trainer import TTMTrainer

# 1. Load ONLY training data
train_data = load_training_data("kaggle_brisT1D")

# 2. Train model
trainer = TTMTrainer(config)
model = trainer.train(train_data)

# 3. Save model
model.save("trained_models/artifacts/ttm/my_model/")

# 4. Final evaluation on holdout (DO THIS ONCE)
holdout_data = load_holdout_data("kaggle_brisT1D")
results = trainer.evaluate(model, holdout_data)
print(f"Holdout RMSE: {results['rmse']}")
```

## Modifying Holdout Configurations

If you need to change holdout strategies:

1. **Modify generation script**: Edit parameters in `generate_holdout_configs.py`
2. **Regenerate configs**: Run the generation script
3. **Validate changes**: Run validation script
4. **Update version**: Increment version in configs
5. **Document changes**: Note changes in experiment logs

## Troubleshooting

### "No holdout configuration found"

**Solution**: Run `python scripts/data_processing_scripts/generate_holdout_configs.py`

### "Patient overlap detected"

**Issue**: Training and holdout sets have overlapping patients

**Solution**: Check configuration - patient_config.holdout_patients may be incorrect

### "Training set very small"

**Issue**: Not enough data for training after split

**Solution**: Adjust `holdout_percentage` or `min_train_samples` in config

### "Temporal ordering issue"

**Issue**: Training data extends into holdout time period

**Solution**: Check time column format and ensure data is properly sorted

## File Locations

- **Configurations**: `configs/data/holdout/*.yaml`
- **Source code**: `src/data/preprocessing/holdout_*.py` and `src/data/versioning/dataset_registry.py`
- **Scripts**: `scripts/data_processing_scripts/{generate,validate}_holdout_configs.py`
- **Documentation**: This file

## Support

For issues or questions:
1. Check validation output for specific errors
2. Review configuration file for dataset
3. Consult example scripts in `scripts/examples/`
4. Check logs for detailed error messages
