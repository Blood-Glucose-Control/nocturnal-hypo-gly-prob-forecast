# TTM (TinyTimeMixer) Model Implementation

This module provides a production-ready implementation of the TTM (TinyTimeMixer) foundation model for time series forecasting, with a focus on blood glucose prediction.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         TTM Module Architecture                         │
└─────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│                         Configuration Layer                              │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────┐         ┌──────────────────────────────────────┐    │
│  │  ModelConfig    │         │        TTMConfig                     │    │
│  │  (Base Class)   │◄────────│  - Extends ModelConfig               │    │
│  │                 │         │  - Adds TTM-specific params          │    │
│  │  • batch_size   │         │  • input_features                    │    │
│  │  • learning_rate│         │  • target_features                   │    │
│  │  • num_epochs   │         │  • scaler_type                       │    │
│  │  • context_len  │         │  • imputation_strategy               │    │
│  │  • forecast_len │         │  • logging_dir                       │    │
│  └─────────────────┘         │  • num_input_channels                │    │
|                              │  • resolution_min                    │    │
│                              └──────────────────────────────────────┘    │
│                                          │                               │
│                                          │ to_training_config()          │
│                                          │ to_data_config()              │
│                     ┌────────────────────┴─────────────────┐             │
│                     │                                      │             │
│          ┌──────────▼───────────┐               ┌──────────▼─────────┐   │
│          │ TTMTrainingConfig    │               │  TTMDataConfig     │   │
│          │ • freeze_backbone    │               │  • input_features  │   │
│          │ • loss_function      │               │  • target_features │   │
│          │ • scaler_type        │               │  • split_config    │   │
│          └──────────────────────┘               │  • num_channels    │   │
│                                                 └────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│                          Model Layer                                     │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────┐         ┌──────────────────────────────────────┐    │
│  │BaseTSFM         │         │ TTMForecaster                        │    │
│  │ (Abstract Base) │◄────────│  - Concrete TTM implementation       │    │
│  │                 │         │                                      │    │
│  │ Abstract:       │         │ Implements:                          │    │
│  │ • fit()         │         │ • _initialize_model()                │    │
│  │ • predict()     │         │ • _prepare_training_data()                    │    │
│  │ • save()        │         │ • _train_model()                     │    │
│  │ • load()        │         │ • _save_model_weights()              |    │
│  │ • evaluate()    │         │ • _load_model_weights()              │    │
│  └─────────────────┘         │                                      │    │
│                              │ TTM-Specific:                        │    │
│                              │ • predict_zero_shot()                │    │
│                              │ • get_ttm_specific_info()            │    │
│                              └──────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────┘
```

## Data Flow Pipeline

```
┌───────────────────────────────────────────────────────────────────────────┐
│                      Training Data Flow                                   │
└───────────────────────────────────────────────────────────────────────────┘

Input Data Format
      │
      ├──► String (dataset name)
      │         │
      │         └──► get_loader() ──► Dict[patient_id → DataFrame]
      │
      ├──► Dict (multi-patient)
      │         │
      │         └──► reduce_features_multi_patient()
      │                   │
      │                   ├─ Select correct resolution
      │                   ├─ Filter to required columns
      │                   ├─ Impute missing values (per patient)
      │                   └─ Concatenate → Single DataFrame
      │
      └──► DataFrame (already combined)
                │
                └──► Used directly
                            │
                            ▼
            ┌─────────────────────────────────┐
            │   Data Quality Checks           │
            │   (in workflow script)          │
            │   • Check for NaN values        │
            │   • Check for zero variance     │
            │   • Check for infinite values   │
            │   • Impute missing values       │
            │   • Drop zero-variance columns  │
            └───────────────┬─────────────────┘
                            │
                            ▼
            ┌─────────────────────────────────┐
            │  _create_column_specifiers()    │
            │  • Identify id_columns          │
            │  • Identify target_columns      │
            │  • Identify observable_columns  │
            └───────────────┬─────────────────┘
                            │
                            ▼
            ┌─────────────────────────────────┐
            │   TimeSeriesPreprocessor        │
            │   (from tsfm_public)            │
            │   • StandardScaler normalization│
            │   • Create sliding windows      │
            │   • Context length = 512        │
            │   • Forecast length = 96        │
            └───────────────┬─────────────────┘
                            │
                            ▼
            ┌─────────────────────────────────┐
            │      get_datasets()             │
            │   • Split: train/val/test       │
            │   • Create PyTorch DataLoaders  │
            │   • Apply batch_size            │
            └───────────────┬─────────────────┘
                            │
                            ▼
            ┌─────────────────────────────────┐
            │   HuggingFace Trainer           │
            │   • Training loop               │
            │   • Gradient computation        │
            │   • Optimizer step              │
            │   • Logging to TensorBoard      │
            └─────────────────────────────────┘
```

## Training Workflow

```
┌───────────────────────────────────────────────────────────────────────────┐
│                     Training Workflow Sequence                            │
└───────────────────────────────────────────────────────────────────────────┘

1. Configuration Setup
   ┌────────────────────────────────────────┐
   │  config = TTMConfig(                   │
   │      model_path="...",                 │
   │      context_length=512,               │
   │      forecast_length=96,               │
   │      batch_size=2048,                  │
   │      logging_dir="/path/to/logs"       │
   │  )                                     │
   └───────────────┬────────────────────────┘
                   │
                   ▼
2. Model Initialization
   ┌────────────────────────────────────────┐
   │  model = TTMForecaster(                │
   │      config=config,                    │
   │      distributed_config=dist_config    │
   │  )                                     │
   │                                        │
   │  Internal: _initialize_model()         │
   │    • Load pre-trained weights          │
   │    • Set gradient requirements         │
   │    • Configure for fine-tuning         │
   └───────────────┬────────────────────────┘
                   │
                   ▼
3. Data Preparation & Training
   ┌────────────────────────────────────────┐
   │  model.fit(                            │
   │      train_data=combined_data,         │
   │      output_dir="./trained_models"     │
   │  )                                     │
   │                                        │
   │  Internal Flow:                        │
   │    ├─► _prepare_training_data()                 │
   │    │     • Create column specifiers    │
   │    │     • Initialize preprocessor     │
   │    │     • Create DataLoaders          │
   │    │                                   │
   │    ├─► _create_training_arguments()    │
   │    │     • Set learning_rate           │
   │    │     • Set num_epochs              │
   │    │     • Set logging_dir             │
   │    │     • Configure callbacks         │
   │    │                                   │
   │    └─► _train_model()                  │
   │          • Create HF Trainer           │
   │          • Execute training loop       │
   │          • Save checkpoints            │
   │          • Log metrics                 │
   └───────────────┬────────────────────────┘
                   │
                   ▼
4. Model Persistence
   ┌────────────────────────────────────────┐
   │  model.save("model.pt")          │
   │                                        │
   │  Saves:                                │
   │    • Model weights (HF format)         │
   │    • Configuration (JSON)              │
   │    • Metadata (timestamps, etc.)       │
   └───────────────┬────────────────────────┘
                   │
                   ▼
5. Inference
   ┌────────────────────────────────────────┐
   │  loaded_model = TTMForecaster(config)  │
   │  loaded_model.load("model.pt")   │
   │                                        │
   │  predictions = loaded_model.predict(   │
   │      data=test_data,                   │
   │      batch_size=32                     │
   │  )                                     │
   └────────────────────────────────────────┘
```

## Key Components

### 1. Configuration Classes

#### TTMConfig
Main configuration class that extends `ModelConfig` with TTM-specific parameters.

**Key Attributes:**
- `model_path`: HuggingFace model ID or local path
- `context_length`: Number of input time steps (default: 512)
- `forecast_length`: Number of prediction time steps (default: 96)
- `input_features`: List of input column names
- `target_features`: List of target column names
- `logging_dir`: Custom directory for TensorBoard logs
- `scaler_type`: Normalization method ("standard", "minmax", "robust")
- `imputation_strategy`: How to handle NaN values ("mean", "median", "forward_fill")

**Factory Functions:**
- `create_default_ttm_config()`: Standard training configuration
- `create_ttm_fine_tuning_config()`: Lower LR, fewer epochs
- `create_ttm_zero_shot_config()`: No training, inference only

#### TTMTrainingConfig
Dataclass for training-specific parameters.

#### TTMDataConfig
Dataclass for data preprocessing parameters.

### 2. Model Class

#### TTMForecaster
Main model class that implements the TTM forecasting pipeline.

**Public Methods:**
- `fit(train_data, output_dir)`: Train the model
- `predict(data)`: Generate predictions
- `save(path)`: Save model and config
- `load(path)`: Load model from disk
- `predict_zero_shot(data)`: Inference without training

**Private Methods:**
- `_initialize_model()`: Load pre-trained TTM
- `_prepare_training_data()`: Create DataLoaders
- `_train_model()`: Execute training loop
- `_create_training_arguments()`: Configure HF Trainer
- `_compute_trainer_metrics()`: Custom metrics

### 3. Data Processing

The module handles three input formats:

1. **String (dataset name)**: Loads via `get_loader()`
2. **Dict (multi-patient)**: Uses `reduce_features_multi_patient()`
3. **DataFrame**: Direct use after quality checks

All paths converge to a single DataFrame that gets processed through:
- Column specification
- TimeSeriesPreprocessor (scaling + windowing)
- PyTorch DataLoader creation

## Usage Examples

### Basic Training

```python
from src.models.ttm import TTMConfig, TTMForecaster

# Configure model
config = TTMConfig(
    model_path="ibm-granite/granite-timeseries-ttm-r2",
    context_length=512,
    forecast_length=96,
    batch_size=2048,
    learning_rate=1e-4,
    num_epochs=1,
    logging_dir="./logs/ttm_run_001"
)

# Create and train model
model = TTMForecaster(config)
model.fit(
    train_data=combined_dataframe,
    output_dir="./trained_models"
)

# Save model
model.save("./trained_models/model.pt")
```

### Custom Logging Directory

```python
from pathlib import Path

# Create custom logging path
job_id = "12345"
datasets = ["lynch_2022", "brown_2019"]
logging_dir = Path("./outputs") / f"logs_job{job_id}_{'_'.join(datasets)}"

config = TTMConfig(
    model_path="ibm-granite/granite-timeseries-ttm-r2",
    logging_dir=str(logging_dir),  # Custom TensorBoard logs location
    # ... other params
)
```

### Zero-Shot Inference

```python
# Load pre-trained model without training
config = TTMConfig(
    model_path="ibm-granite/granite-timeseries-ttm-r2",
    training_mode="zero_shot",
    num_epochs=0
)

model = TTMForecaster(config)
predictions = model.predict_zero_shot(test_data)
```

### Fine-Tuning

```python
from src.models.ttm import create_ttm_fine_tuning_config

# Use pre-configured fine-tuning defaults
config = create_ttm_fine_tuning_config(
    model_path="ibm-granite/granite-timeseries-ttm-r2",
    batch_size=1024,
    # Fine-tuning defaults: lr=1e-5, epochs=5, warmup=500
)

model = TTMForecaster(config)
model.fit(train_data=data, output_dir="./fine_tuned_model")
```

## Data Requirements

### Input DataFrame Structure

The model expects a pandas DataFrame with:

**Required Columns:**
- `datetime`: Timestamp column
- Target feature(s): e.g., `bg_mM` (blood glucose)
- Input feature(s): e.g., `cob`, `iob`, `dose_units`, etc.
- Patient ID: `p_num`, `id`, or similar

**Example:**
```
| datetime            | bg_mM | cob  | iob  | dose_units | p_num      |
|---------------------|-------|------|------|------------|------------|
| 2024-01-01 00:00:00 | 5.2   | 12.3 | 0.5  | 0.0        | patient_01 |
| 2024-01-01 00:05:00 | 5.3   | 11.8 | 0.48 | 0.0        | patient_01 |
| ...                 | ...   | ...  | ...  | ...        | ...        |
```

### Data Quality Considerations

**Common Issues:**
1. **NaN Values**: Imputed automatically if using dict input, manual for DataFrame
2. **Zero Variance**: Columns with constant values should be dropped
3. **Infinite Values**: Should be handled before passing to model
4. **Resolution**: Data should be at 5-minute intervals (configurable)

**Best Practice**: Use data quality checks before training:
```python
# Check for issues
for col in df.select_dtypes(include=['number']).columns:
    nan_pct = df[col].isna().sum() / len(df) * 100
    if nan_pct > 0:
        print(f"{col}: {nan_pct:.2f}% NaN")

    if df[col].std() == 0:
        print(f"{col}: Zero variance - consider dropping")
```

## Troubleshooting

### RuntimeWarning: invalid value encountered in divide

**Cause**: StandardScaler encountering NaN or zero-variance columns

**Solution**:
```python
from src.data.preprocessing.imputation import impute_missing_values

# Impute NaN values
df = impute_missing_values(df, columns=numeric_columns)

# Drop zero-variance columns
zero_var_cols = [col for col in df.columns if df[col].std() == 0]
df = df.drop(columns=zero_var_cols)
```

### Model not loading correctly

**Cause**: Version mismatch or corrupted checkpoint

**Solution**:
```python
# Ensure config matches saved model
config = TTMConfig(model_path="ibm-granite/granite-timeseries-ttm-r2")
model = TTMForecaster(config)
model.load("./path/to/model.pt")  # Load after initialization
```

### Out of Memory (OOM) errors

**Solutions**:
1. Reduce `batch_size` in config
2. Reduce `context_length` (e.g., 256 instead of 512)
3. Enable gradient checkpointing (if available)
4. Use mixed precision training (`fp16=True`)

## Directory Structure

```
src/models/ttm/
├── README.md              # This file
├── __init__.py            # Package exports
├── config.py              # Configuration classes
└── model.py               # TTMForecaster implementation
```

## Related Documentation

- [Base Model Framework](../base/base_model.py)
- [Data Loading](../../data/diabetes_datasets/data_loader.py)
- [Preprocessing](../../data/preprocessing/)
- [Holdout System](../../data/versioning/)

## References

- IBM Granite TTM Model: https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2
- TinyTimeMixer Paper: [Link to paper]
- TSFM Public Library: https://github.com/ibm-granite/granite-tsfm
