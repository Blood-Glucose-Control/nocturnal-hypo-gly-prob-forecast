# Holdout System Workflow Example

Complete end-to-end workflow demonstrating the holdout system with TTM training and evaluation.

## Quick Start

**Recommended: Submit as batch job (don't run on login node)**

```bash
# Run with default dataset (lynch_2022)
sbatch scripts/examples/run_holdout_ttm_workflow.sh

# Run with different dataset
sbatch --export=DATASETS="aleppo" scripts/examples/run_holdout_ttm_workflow.sh

# Use different config directory
sbatch --export=DATASETS="brown_2019",CONFIG_DIR=configs/data/holdout scripts/examples/run_holdout_ttm_workflow.sh

# More epochs for better training
sbatch --export=EPOCHS=5 scripts/examples/run_holdout_ttm_workflow.sh

# Skip training (evaluate existing model)
sbatch --export=DATASETS="lynch_2022",SKIP_TRAINING=true scripts/examples/run_holdout_ttm_workflow.sh
```

## Workflow Steps

The workflow performs these steps automatically:

1. **Generate holdout configs** - Checks if configurations exist
2. **Validate configs** - Validates the holdout configuration and data split
3. **Load training data** - Loads training data only (holdout excluded)
4. **Train TTM model** - Trains on single GPU with 2 epochs (configurable)
5. **Save model** - Saves trained model to artifacts directory
6. **Load model** - Reloads the saved model to verify persistence
7. **Evaluate on holdout** - Evaluates on holdout set and reports metrics

## Configuration Options

All options can be set via `--export` when submitting:

- `DATASETS` - Space-separated list of datasets (default: `"lynch_2022 aleppo brown_2019 tamborlane_2008"`)
  - Available: `lynch_2022`, `aleppo`, `brown_2019`, `tamborlane_2008`
- `CONFIG_DIR` - Holdout config directory (default: `configs/data/holdout_5pct`)
- `OUTPUT_BASE_DIR` - Training output base directory (default: auto-generated with job ID)
- `SKIP_TRAINING` - Skip training step (default: `false`)
- `EPOCHS` - Number of training epochs (default: `1`)

## SLURM Settings

Default resource allocation (modify in the script as needed):

```bash
#SBATCH --time=12:00:00        # 12 hours (adjust based on dataset size)
#SBATCH --cpus-per-task=8      # 8 CPU cores
#SBATCH --mem=32GB             # 32GB RAM
#SBATCH --gres=gpu:1           # 1 GPU
#SBATCH --partition=HI        # GPU partition
```

## Output Files

After completion, you'll find:

```
trained_models/artifacts/<YYYY-MM-DD_HH:MM_JID<jobid>_holdout_workflow/
├── model.pt                    # Saved model
├── training_logs/              # Training logs
├── holdout_ttm_<jobid>.out    # SLURM stdout
├── holdout_ttm_<jobid>.err    # SLURM stderr
└── holdout_workflow_<jobid>.log  # Detailed workflow log
└── checkpoints/                # Model checkpoints (if enabled)
```

## Monitoring

Check job status:
```bash
squeue -u $USER
```

View logs in real-time:
```bash
tail -f trained_models/logs/holdout_ttm_<jobid>.out
```
Note: These get moved at the end of the training run to the respective artifacts directory.

## Examples

### Example 1: Full workflow on Lynch dataset
```bash
sbatch scripts/examples/run_holdout_ttm_workflow.sh
```

### Example 2: Multiple datasets with more epochs
```bash
sbatch --export=DATASETS="lynch_2022 tamborlane_2008",EPOCHS=3 scripts/examples/run_holdout_ttm_workflow.sh

```

### Example 3: Quick test with single epoch (default)
```bash
sbatch --export=DATASETS="lynch_2022" scripts/examples/run_holdout_ttm_workflow.sh
```

## Direct Execution (Not Recommended for Training do not run on Log-in Node, use salloc)

For quick testing or debugging only (with salloc):

```bash
python scripts/examples/example_holdout_ttm_workflow.py \
    --dataset lynch_2022 \
    --config-dir configs/data/holdout_5pct
```

⚠️ **Warning**: Direct execution on login nodes may impact other users and violate cluster policies. Always use batch submission for actual training.

## Troubleshooting

**Job fails immediately:**
- Check if configs exist: `ls configs/data/holdout_5pct/`
- Verify dataset name is correct
- Check SLURM queue limits: `sinfo`

**Out of memory:**
- Reduce batch size in the script (edit TTM config)
- Request more memory: `sbatch --export=... --mem=64GB ...`

**GPU issues:**
- Check GPU availability: `sinfo -p gpu`
- Verify CUDA version: `module list`

## Related Scripts

- `load_holdout_data_example.py` - Simple data loading examples
- `example_single_gpu_ttm.py` - Basic TTM training without holdout system
- `validate_holdout_configs.py` - Standalone config validation
