# Data Holdout System - Quick Reference

## ⚡ Quick Start (3 Steps)

### 1. Generate Configs (Run Once)

```bash
python scripts/data_processing_scripts/generate_holdout_configs.py
```

### 2. Validate Configs

```bash
python scripts/data_processing_scripts/validate_holdout_configs.py
```

### 3. Use in Your Code

```python
from src.data.versioning.dataset_registry import load_training_data

# Load ONLY training data (holdout excluded)
train_data = load_training_data("kaggle_brisT1D")

# Train your model
model.fit(train_data)
```

---

## 📋 Common Commands

### Load Training Data

```python
from src.data.versioning.dataset_registry import load_training_data

train_data = load_training_data("kaggle_brisT1D")
```

### Load Holdout Data (Final Evaluation Only)

```python
from src.data.versioning.dataset_registry import load_holdout_data

holdout_data = load_holdout_data("kaggle_brisT1D")
results = model.evaluate(holdout_data)
```

### Load Both Splits

```python
from src.data.versioning.dataset_registry import load_split_data

train_data, holdout_data = load_split_data("kaggle_brisT1D")
```

### Get Split Info

```python
from src.data.versioning.dataset_registry import get_dataset_registry

registry = get_dataset_registry()
info = registry.get_split_info("kaggle_brisT1D")
print(f"Holdout patients: {info['patient_split']['holdout_patients']}")
```

### List Available Datasets

```python
from src.data.versioning.dataset_registry import get_dataset_registry

registry = get_dataset_registry()
datasets = registry.list_available_datasets()
print(datasets)  # ['kaggle_brisT1D', 'gluroo', "aleppo_2017", ...]
```

---

## 📁 File Locations

| Item | Location |
|------|----------|
| **Holdout Configs** | `configs/data/holdout/*.yaml` |
| **Config Classes** | `src/data/preprocessing/holdout_config.py` |
| **Split Manager** | `src/data/preprocessing/holdout_manager.py` |
| **Dataset Registry** | `src/data/versioning/dataset_registry.py` |
| **Generation Script** | `scripts/data_processing_scripts/generate_holdout_configs.py` |
| **Validation Script** | `scripts/data_processing_scripts/validate_holdout_configs.py` |
| **Examples** | `scripts/examples/example_data_holdout_system.py` |
| **Full Documentation** | `configs/data/holdout/README.md` |

---

## 🎯 Holdout Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| **Temporal** | Hold out end of time series (e.g., last 20%) | Evaluate temporal prediction |
| **Patient-Based** | Hold out specific patients entirely | Evaluate patient generalization |
| **Hybrid** | Both temporal + patient holdout | Comprehensive evaluation (recommended) |

---

## ✅ Best Practices

### DO ✅

- Use `load_training_data()` for training
- Use `load_holdout_data()` only for final evaluation
- Validate configs before experiments
- Keep configs in version control
- Document which config version you used

### DON'T ❌

- Train on holdout data
- Modify configs after experiments start
- Use different splits across experiments
- Skip validation

---

## 🔍 Validation Checks

The validation script checks:

- ✓ Configuration files exist
- ✓ Data loads successfully
- ✓ No patient overlap between train/holdout
- ✓ Temporal ordering is correct
- ✓ Minimum sample requirements met

---

## 🐛 Troubleshooting

| Error | Solution |
|-------|----------|
| "No holdout configuration found" | Run `generate_holdout_configs.py` |
| "Patient overlap detected" | Check `holdout_patients` in config |
| "Training set very small" | Adjust `holdout_percentage` in config |
| "Temporal ordering issue" | Check time column format |

---

## 📊 Example Config

```yaml
dataset_name: kaggle_brisT1D
holdout_type: hybrid

temporal_config:
  holdout_percentage: 0.2      # 20% of data at end
  min_train_samples: 50
  min_holdout_samples: 10

patient_config:
  holdout_patients: [p04, p12, p18]  # Fixed holdout patients
  min_train_patients: 2
  random_seed: 42
```

---

## 🚀 Examples

Run complete examples:

```bash
python scripts/examples/example_data_holdout_system.py
```

Or see specific examples in the script:
- Example 1: Load training data only
- Example 2: Load holdout data only
- Example 3: Load both splits
- Example 4: Get split information
- Example 5: Validate a split
- Example 6: Typical training workflow
- Example 7: Multi-dataset training

---

## 📖 Full Documentation

For comprehensive documentation, see: [configs/data/holdout/README.md](README.md)

---

## 💡 Integration with Training Scripts

### Minimal Example

```python
from src.data.versioning.dataset_registry import load_training_data, load_holdout_data

# 1. Load training data
train_data = load_training_data("kaggle_brisT1D")

# 2. Train model
model = MyModel()
model.fit(train_data)

# 3. Save model
model.save("trained_models/artifacts/my_model/")

# 4. Final evaluation
holdout_data = load_holdout_data("kaggle_brisT1D")
results = model.evaluate(holdout_data)
```

### With Experiment Tracking

```python
from src.data.versioning.dataset_registry import load_training_data, load_holdout_data, get_dataset_registry

# Get split info for metadata
registry = get_dataset_registry()
split_info = registry.get_split_info("kaggle_brisT1D")

# Load data
train_data = load_training_data("kaggle_brisT1D")

# Train with experiment tracking
experiment = Experiment(
    name="my_experiment",
    holdout_config_version=split_info["version"],
    holdout_patients=split_info["patient_split"]["holdout_patients"],
)

model = train_model(train_data)
experiment.log_model(model)

# Final evaluation
holdout_data = load_holdout_data("kaggle_brisT1D")
results = evaluate_model(model, holdout_data)
experiment.log_results(results)
```

---

**Last Updated**: 2026-01-09
**Version**: 1.0
