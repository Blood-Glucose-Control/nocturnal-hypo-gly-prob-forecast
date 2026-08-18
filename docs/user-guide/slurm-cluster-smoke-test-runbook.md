# SLURM Cluster Smoke Test Runbook (P1-37 Phase C)

Use this runbook on your private SLURM cluster to validate the Phase C launcher
rewire with real `sbatch` execution.

For a compact command-only version, see:

- `docs/user-guide/slurm-cluster-smoke-test-checklist.md`

## Goal

Validate that these rewired launchers run end-to-end on cluster:

- `scripts/training/slurm/single_gpu.sh`
- `scripts/training/slurm/multi_gpu.sh`
- `scripts/training/slurm/adaptive_resources.sh`

All three should route to:

- `scripts/workflows/forecasting/run_forecasting_workflow.sh`

## 1) Clone repo on cluster

Choose one remote URL you have access to:

```bash
# Public
git clone https://github.com/Blood-Glucose-Control/nocturnal-hypo-gly-prob-forecast.git

# Private mirror
git clone https://github.com/Blood-Glucose-Control/nocturnal-hypo-gly-prob-forecast-private.git
```

Then:

```bash
cd nocturnal-hypo-gly-prob-forecast
```

Checkout target branch:

```bash
# If Phase C PR is not merged yet
git checkout p1-scripts-phase-c-launcher-rewire

# Or, if merged already
# git checkout main && git pull --ff-only
```

## 2) Prepare environment

Create/activate the shared AutoGluon environment (used for Chronos2 smoke):

```bash
source scripts/setup_model_env.sh autogluon
```

If your cluster has multiple Python versions, ensure Python 3.12 is available
for this environment.

## 3) Optional preflight checks (fast)

```bash
bash -n scripts/training/slurm/single_gpu.sh
bash -n scripts/training/slurm/multi_gpu.sh
bash -n scripts/training/slurm/adaptive_resources.sh

DRY_RUN=1 bash scripts/training/slurm/single_gpu.sh
DRY_RUN=1 NUM_GPUS=2 bash scripts/training/slurm/multi_gpu.sh
DRY_RUN=1 FORCE_NUM_GPUS=1 bash scripts/training/slurm/adaptive_resources.sh
```

## 4) Submit real smoke runs

Recommended smoke profile (fast-ish, low risk):

- `MODEL_TYPE=chronos2`
- `VENV_NAME=autogluon`
- `MODEL_CONFIG=configs/models/chronos2/bg_only_test.yaml`
- `DATASETS="brown_2019"`
- `EPOCHS=1`
- `SKIP_STEPS="7"` (run through checkpoint load, skip resumed training)

### A. Single GPU launcher

```bash
sbatch --export=ALL,MODEL_TYPE=chronos2,VENV_NAME=autogluon,MODEL_CONFIG=configs/models/chronos2/bg_only_test.yaml,DATASETS="brown_2019",CONFIG_DIR=configs/data/holdout_10pct,SKIP_TRAINING=false,EPOCHS=1,SKIP_STEPS="7" \
  scripts/training/slurm/single_gpu.sh
```

### B. Multi GPU launcher

Adjust `NUM_GPUS`/`GPUS` for your node:

```bash
sbatch --export=ALL,MODEL_TYPE=chronos2,VENV_NAME=autogluon,MODEL_CONFIG=configs/models/chronos2/bg_only_test.yaml,DATASETS="brown_2019",CONFIG_DIR=configs/data/holdout_10pct,SKIP_TRAINING=false,EPOCHS=1,SKIP_STEPS="7",NUM_GPUS=2,GPUS="0 1" \
  scripts/training/slurm/multi_gpu.sh
```

### C. Adaptive launcher

```bash
sbatch --export=ALL,MODEL_TYPE=chronos2,VENV_NAME=autogluon,MODEL_CONFIG=configs/models/chronos2/bg_only_test.yaml,DATASETS="brown_2019",CONFIG_DIR=configs/data/holdout_10pct,SKIP_TRAINING=false,EPOCHS=1,SKIP_STEPS="7",FORCE_NUM_GPUS=1 \
  scripts/training/slurm/adaptive_resources.sh
```

## 5) Monitor and collect evidence

Track jobs:

```bash
squeue -u "$USER"
```

After completion:

```bash
sacct -j <job_id> --format=JobID,JobName,State,ExitCode,Elapsed,AllocTRES%60
```

Launcher logs are configured in each script via `#SBATCH --output/--error`
(default under `logs/`).

Workflow artifacts and logs are written under:

- `trained_models/artifacts/<model_type>/...`
- `trained_models/logs/...`

## 6) Pass criteria

For each launcher (`single_gpu`, `multi_gpu`, `adaptive_resources`):

1. Job exits `COMPLETED` in SLURM (`ExitCode=0:0`).
2. No launcher-level shell errors in SLURM stdout/err logs.
3. Workflow emits final completion block with `Exit code: 0`.
4. Output artifact directory is created under `trained_models/artifacts/chronos2/`.

## 7) If something fails

Capture and share:

1. Exact `sbatch` command used.
2. Job ID and `sacct` output.
3. Tail of launcher output/error logs.
4. Any traceback from `trained_models/logs/forecasting_workflow_*`.

This is sufficient to triage whether failure is cluster-environment-related
(partition, CUDA, env setup) or workflow/launcher logic.
