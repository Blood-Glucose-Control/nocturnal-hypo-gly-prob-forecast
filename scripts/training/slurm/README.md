# SLURM Training Scripts

This directory contains SLURM launchers for running the maintained forecasting
workflow stack on cluster infrastructure.

## Generic launchers (Phase C rewired)

These launchers now route to:

- `scripts/experiments/run_forecasting_workflow.sh`

without depending on deprecated legacy example entrypoints.

- `single_gpu.sh`: fixed single-GPU workflow launcher.
- `multi_gpu.sh`: multi-GPU launcher (explicit list or count-based selection).
- `adaptive_resources.sh`: auto-selects CPU/single-GPU/multi-GPU strategy.

## Usage

```bash
sbatch scripts/training/slurm/single_gpu.sh
sbatch scripts/training/slurm/multi_gpu.sh
sbatch scripts/training/slurm/adaptive_resources.sh
```

### Common overrides (`sbatch --export=...`)

- `MODEL_TYPE` (default: `ttm`)
- `MODEL_CONFIG` (or legacy alias `CONFIG_PATH`)
- `DATASETS` (space-separated)
- `CONFIG_DIR`
- `OUTPUT_BASE_DIR` (or legacy `OUTPUT_DIR` + `EXPERIMENT_NAME`)
- `SKIP_TRAINING`
- `SKIP_STEPS`
- `EPOCHS`
- `BATCH_SIZE`
- `VENV_NAME`

### GPU-specific overrides

- `single_gpu.sh`:
  - `CUDA_VISIBLE_DEVICES` (default `0`)
- `multi_gpu.sh`:
  - `NUM_GPUS` (default `4`)
  - `GPUS` (space-separated explicit list, e.g. `"0 1 2"`)
  - `MASTER_PORT` (default `29500`)
- `adaptive_resources.sh`:
  - `FORCE_NUM_GPUS` (e.g. `0`, `1`, `2`)

### Compatibility aliases

For migration safety, generic launchers still accept:

- `CONFIG_PATH` -> `MODEL_CONFIG`
- `DATA_CONFIG` -> derives `DATASETS` + `CONFIG_DIR` when not explicitly set
- `OUTPUT_DIR` + `EXPERIMENT_NAME` -> `OUTPUT_BASE_DIR`

## Dry-run validation mode

All three generic launchers support:

```bash
DRY_RUN=1 bash scripts/training/slurm/single_gpu.sh
```

to validate command wiring and configuration resolution without executing the
workflow.

## Model-specific scripts

Model-specific launchers remain available for specialized workflows:

- `chronos2_eval.sh`
- `chronos2_finetune.sh`
- `chronos2_forecasting_workflow.sh`
- `chronos2_time_covariate.sh`
- `per_patient_finetune.sh`
- `per_patient_finetune_all.sh`
- `toto_eval.sh`
