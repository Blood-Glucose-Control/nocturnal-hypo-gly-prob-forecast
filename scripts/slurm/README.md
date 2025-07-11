# SLURM Training Scripts

This directory contains SLURM batch scripts for training models on the cluster.

## Quick Start

1. **Validate setup**: `./validate_setup.sh`
2. **Submit ARIMA job**: `./submit_arima_job.sh`
3. **Monitor jobs**: `squeue -u $USER`

## Scripts

### Training Scripts
- `train_cpu.sh` - For statistical models (ARIMA, AutoARIMA) that don't need GPU
- `train_gpu.sh` - For deep learning models that require GPU
- `submit_arima_job.sh` - Convenient wrapper for submitting ARIMA jobs

### Utility Scripts
- `validate_setup.sh` - Check if everything is configured correctly
- `evaluate_models.sh` - Evaluate trained models (if available)

## Usage Examples

### Submit ARIMA training job:
```bash
# Train with default config (ARIMA)
./submit_arima_job.sh

# Train with AutoARIMA
./submit_arima_job.sh -c scripts/training/configs/autoarima_config.yaml

# Train on specific patients
./submit_arima_job.sh -p "patient_001 patient_002"

# Custom job name
./submit_arima_job.sh -n my_arima_experiment
```

### Monitor jobs:
```bash
# Check job queue
squeue -u $USER

# Watch specific job
squeue -j <JOB_ID>

# View logs in real-time
tail -f logs/<JOB_ID>_train_statistical.out
tail -f logs/<JOB_ID>_train_statistical.err
```

### Cancel jobs:
```bash
scancel <JOB_ID>          # Cancel specific job
scancel -u $USER          # Cancel all your jobs
```

## Configuration

### Available partitions:
- `HI` - High priority GPU partition
- `GUEST` - General access partition
- `SCHOOL` - School partition
- See `sinfo` for full list

### Resource requirements:
- **Statistical models**: 4 CPUs, 16GB RAM, 4 hours
- **Deep learning models**: 1 GPU, 8 CPUs, 64GB RAM, 24 hours

### File locations:
- **Configs**: `scripts/training/configs/`
- **Logs**: `logs/`
- **Models**: `models/`

## Troubleshooting

1. **Job pending**: Check `squeue` and partition availability
2. **Permission denied**: Run `chmod +x scripts/slurm/*.sh`
3. **Config not found**: Check paths in config files
4. **Environment issues**: Verify virtual environment exists

For detailed setup validation, run `./validate_setup.sh`
