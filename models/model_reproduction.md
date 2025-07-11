# Model Reproduction Guide

This document explains how to reproduce the trained models in this repository.

## Quick Start

```bash
# Submit an ARIMA training job
./scripts/slurm/submit_arima_job.sh

# Or run locally (not recommended for full dataset)
python scripts/training/train_statistical.py --config scripts/training/configs/arima_config.yaml
```

## Available Models

### StatsForecastAutoARIMA
- **Config**: `scripts/training/configs/arima_config.yaml`
- **Description**: Automatic ARIMA model selection using statsforecast library
- **Compatible with**: numpy 2.x (avoids compatibility issues)
- **Training time**: ~1-2 minutes with cached data

### AutoARIMA (Alternative)
- **Config**: `scripts/training/configs/autoarima_config.yaml`
- **Description**: Traditional AutoARIMA from sktime
- **Note**: May require numpy<2 (compatibility issues with newer numpy)

## Training Pipeline

### 1. Configuration
Models are configured via YAML files in `scripts/training/configs/`:

```yaml
# Key configuration options
model_type: "statistical"
model_name: "StatsForecastAutoARIMA"
dataset_name: "kaggle_brisT1D"

# Important: Use cached data to avoid heavy processing
data_config:
  use_cached: true
  dataset_type: "train"
```

### 2. SLURM Execution (Recommended)
```bash
# Basic submission
./scripts/slurm/submit_arima_job.sh

# With custom config
./scripts/slurm/submit_arima_job.sh -c scripts/training/configs/autoarima_config.yaml

# With specific patients
./scripts/slurm/submit_arima_job.sh -p "patient_001 patient_002"

# Test without submitting
./scripts/slurm/submit_arima_job.sh --test
```

### 3. Local Execution (Testing Only)
```bash
# Not recommended for full dataset - use for testing only
python scripts/training/train_statistical.py --config scripts/training/configs/arima_config.yaml
```

## Output Structure

When training completes, models are saved to `models/` with this structure:
```
models/
└── {timestamp}_{model_name}_{dataset}_v1/
    ├── config_{timestamp}.yaml    # Original training config
    ├── metadata.json              # Training metrics and info
    └── model.pkl                  # Trained model (excluded from git)
```

### Example metadata.json:
```json
{
  "model_type": "statistical",
  "model_name": "StatsForecastAutoARIMA",
  "dataset_name": "kaggle_brisT1D",
  "training_date": "2025-07-10T22:42:43",
  "training_samples": 1000,
  "patient_ids": ["patient_001", "patient_002"],
  "in_sample_mse": 0.124,
  "in_sample_mae": 0.089
}
```

## Monitoring Jobs

```bash
# Check job status
squeue -u $USER

# View logs in real-time
tail -f logs/{JOB_ID}_train_statistical.out

# Check successful runs
ls logs/*_out | head -5
```

## Troubleshooting

### Common Issues:

1. **"Training failed with exit code: 1"**
   - Check error log: `cat logs/{JOB_ID}_train_statistical.err`
   - Ensure `use_cached: true` in config

2. **"numpy version conflict"**
   - Use `StatsForecastAutoARIMA` instead of `ARIMA`
   - Avoid downgrading numpy

3. **"Data loading hangs"**
   - Verify cached data exists: `ls data/processed/`
   - Set `use_cached: true` in config

4. **"No suitable target column found"**
   - Check data format: verify cached data has expected columns
   - Review data preprocessing pipeline

## Data Requirements

- **Cached data**: Must exist in `data/processed/` directory
- **Format**: Time series with numeric glucose/sensor values
- **Patients**: 15 patients available in current dataset

## Performance Notes

- **With cached data**: 1-2 minutes training time
- **Without cached data**: Can take hours (data processing bottleneck)
- **Resources needed**: 4 CPUs, 16GB RAM (statistical models)

## Next Steps

After reproducing a model:
1. **Evaluate**: Use evaluation scripts (when available)
2. **Experiment**: Try different model parameters
3. **Scale**: Train on more patients or different datasets
4. **Deploy**: Use trained models for inference
