# SLURM Smoke Test Copy/Paste Checklist

Use this on your private SLURM cluster after logging in.

## 0) Clone + checkout

```bash
git clone https://github.com/Blood-Glucose-Control/nocturnal-hypo-gly-prob-forecast.git
cd nocturnal-hypo-gly-prob-forecast
git checkout p1-scripts-phase-c-launcher-rewire
```

If the branch is already merged:

```bash
git checkout main
git pull --ff-only
```

## 1) Environment

```bash
source scripts/setup_model_env.sh autogluon
```

## 2) Quick preflight (optional)

```bash
bash -n scripts/training/slurm/single_gpu.sh
bash -n scripts/training/slurm/multi_gpu.sh
bash -n scripts/training/slurm/adaptive_resources.sh
DRY_RUN=1 bash scripts/training/slurm/single_gpu.sh
DRY_RUN=1 NUM_GPUS=2 bash scripts/training/slurm/multi_gpu.sh
DRY_RUN=1 FORCE_NUM_GPUS=1 bash scripts/training/slurm/adaptive_resources.sh
```

## 3) Submit smoke jobs

```bash
J1=$(sbatch --parsable --export=ALL,MODEL_TYPE=chronos2,VENV_NAME=autogluon,MODEL_CONFIG=configs/models/chronos2/bg_only_test.yaml,DATASETS="brown_2019",CONFIG_DIR=configs/data/holdout_10pct,SKIP_TRAINING=false,EPOCHS=1,SKIP_STEPS="7" scripts/training/slurm/single_gpu.sh)
echo "single_gpu job: $J1"

J2=$(sbatch --parsable --export=ALL,MODEL_TYPE=chronos2,VENV_NAME=autogluon,MODEL_CONFIG=configs/models/chronos2/bg_only_test.yaml,DATASETS="brown_2019",CONFIG_DIR=configs/data/holdout_10pct,SKIP_TRAINING=false,EPOCHS=1,SKIP_STEPS="7",NUM_GPUS=2,GPUS="0 1" scripts/training/slurm/multi_gpu.sh)
echo "multi_gpu job: $J2"

J3=$(sbatch --parsable --export=ALL,MODEL_TYPE=chronos2,VENV_NAME=autogluon,MODEL_CONFIG=configs/models/chronos2/bg_only_test.yaml,DATASETS="brown_2019",CONFIG_DIR=configs/data/holdout_10pct,SKIP_TRAINING=false,EPOCHS=1,SKIP_STEPS="7",FORCE_NUM_GPUS=1 scripts/training/slurm/adaptive_resources.sh)
echo "adaptive_resources job: $J3"
```

## 4) Monitor

```bash
squeue -u "$USER"
```

```bash
sacct -j "$J1,$J2,$J3" --format=JobID,JobName,State,ExitCode,Elapsed,AllocTRES%60
```

## 5) Success criteria

- All three jobs end with `COMPLETED` and `ExitCode=0:0`.
- No launcher shell errors in SLURM logs.
- Workflow reports final `Exit code: 0`.
