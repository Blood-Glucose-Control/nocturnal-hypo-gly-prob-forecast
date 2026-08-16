# Example Scripts Guide

This guide lists the maintained scripts in `scripts/examples/` and
`scripts/experiments/` and when to use each one.

## Active examples

| Script | Purpose | Typical use |
|---|---|---|
| `example_forecasting_workflow.py` | Thin onboarding entrypoint for generic forecasting workflow | Quick walkthrough and CLI discovery |
| `forecasting_workflow_orchestrator.py` (`scripts/experiments/`) | Production CLI entrypoint for generic forecasting workflow | Stable runtime surface for wrappers and automation |
| `run_forecasting_workflow.sh` (`scripts/experiments/`) | Local shell wrapper for the production orchestrator | Repeatable local runs with environment variables |
| `forecasting_workflow_regression_smoke.sh` (`scripts/experiments/`) | Deterministic bounded regression profile | Major-change workflow smoke/regression runs |
| `example_data_holdout_system.py` | Demonstrates holdout config generation/validation/loading APIs | Data split and holdout debugging |
| `example_load_holdout_data.py` | Minimal holdout loading example | Quick data-access sanity checks |

## Model-specific experiment scripts

These are intentionally in `scripts/experiments/` (not `scripts/examples/`):

- `chronos2_finetune.py` (Chronos-2 fine-tuning workflow)
- `ttm_forecasting_workflow.py` (TTM-specific forecasting workflow variant)

## Python-first sweep orchestration

Sweep orchestration now uses model-agnostic Python entrypoints with canonical
taxonomy-aligned shell launchers:

- Generic training CLI: `scripts/experiments/sweep_train.py`
- Generic evaluation CLI: `scripts/experiments/sweep_eval.py`
- Canonical training launcher: `scripts/training/sweeps/run_sweep_train.sh`
- Canonical evaluation launcher: `scripts/evaluation/sweeps/run_sweep_eval.sh`
- Chronos-2 compatibility launchers remain in `scripts/experiments/` as thin
  wrappers only (`chronos2_sweep_train.sh`, `chronos2_sweep_eval.sh`)

The Python sweep CLIs are now dispatcher entrypoints keyed by:

- `--task-family` / `TASK_FAMILY`
- `--experiment-type` / `EXPERIMENT_TYPE`

Currently implemented adapter:

- `task-family=forecasting`, `experiment-type=nocturnal_forecast`

Chronos-2 profile spec:

- `configs/experiments/nocturnal_forecast/chronos2_forecasting_train_sweep.yaml`
- `configs/experiments/nocturnal_forecast/chronos2_forecasting_eval_sweep.yaml`

Local usage:

```bash
MODEL_TYPE=chronos2 \
TASK_FAMILY=forecasting \
EXPERIMENT_TYPE=nocturnal_forecast \
SWEEP_SPEC=configs/experiments/nocturnal_forecast/chronos2_forecasting_train_sweep.yaml \
GPUS="0 1" JOBS_PER_GPU=2 \
bash scripts/training/sweeps/run_sweep_train.sh
```

Path-check (no training execution):

```bash
DRY_RUN=1 GPUS="0" JOBS_PER_GPU=1 bash scripts/training/sweeps/run_sweep_train.sh
```

Chronos-2 eval path-check (no evaluation execution):

```bash
DRY_RUN=1 GPUS="0" JOBS_PER_GPU=1 bash scripts/evaluation/sweeps/run_sweep_eval.sh
```

Generic config-directory mode (same datasets applied to every model config):

```bash
python scripts/experiments/sweep_train.py \
  --model-type chronos2 \
  --model-config-dir configs/models/chronos2 \
  --datasets aleppo_2017 brown_2019 lynch_2022 tamborlane_2008 \
  --gpus "0 1" \
  --jobs-per-gpu 2 \
  --dry-run
```

SLURM usage pattern (launch wrapper inside one allocation):

```bash
sbatch --gres=gpu:2 --wrap 'cd /path/to/repo && TASK_FAMILY=forecasting EXPERIMENT_TYPE=nocturnal_forecast MODEL_TYPE=chronos2 SWEEP_SPEC=configs/experiments/nocturnal_forecast/chronos2_forecasting_train_sweep.yaml GPUS="0 1" JOBS_PER_GPU=2 bash scripts/training/sweeps/run_sweep_train.sh'
```

## Canonical workflow example

Use the production orchestrator as the default entrypoint for training runs:

```bash
python scripts/experiments/forecasting_workflow_orchestrator.py \
  --model-type chronos2 \
  --datasets brown_2019 lynch_2022 \
  --config-dir configs/data/holdout_10pct
```

Use the onboarding example wrapper for quick discovery:

```bash
python scripts/examples/example_forecasting_workflow.py --help
```

Or use the local shell wrapper:

```bash
MODEL_TYPE=chronos2 \
DATASETS="brown_2019 lynch_2022" \
CONFIG_DIR="configs/data/holdout_10pct" \
bash scripts/experiments/run_forecasting_workflow.sh
```

Regression profile (bounded end-to-end guardrail):

```bash
bash scripts/experiments/forecasting_workflow_regression_smoke.sh
```

## Canonical run manifest (pre-MLflow v1)

The maintained forecasting training workflow and nocturnal evaluation workflow now
emit a canonical run manifest at:

- `<workflow_output_dir>/run_manifest.json`

This includes the ratified v1 fields from the design handbook:

- identity (`run_id`, `workflow_name`, `workflow_version`)
- timing (`created_at_utc`, `started_at_utc`, `ended_at_utc`, `duration_seconds`)
- code provenance (`git_commit`, `git_branch`, `git_dirty`, `repository`)
- execution context (`launcher_type`, `host`, `user`, `python_version`, CUDA/SLURM context)
- inputs (data/model config paths + resolved runtime config)
- outputs (artifact root + checkpoint/prediction/plot paths)
- result summary (`key_metrics`, `status`, `failure_message`)

## SLURM launcher entrypoints (cluster runs)

For cluster execution, use the rewired launchers in `scripts/training/slurm/`
that now route to the same maintained workflow path:

- `single_gpu.sh`
- `multi_gpu.sh`
- `adaptive_resources.sh`

Example:

```bash
sbatch scripts/training/slurm/single_gpu.sh
```

The main override surface is the same as local wrapper runs (`MODEL_TYPE`,
`MODEL_CONFIG`, `DATASETS`, `CONFIG_DIR`, `SKIP_TRAINING`, `SKIP_STEPS`).

For a step-by-step private-cluster validation workflow, see:

- `docs/user-guide/slurm-cluster-smoke-test-runbook.md`
- `docs/user-guide/slurm-cluster-smoke-test-checklist.md`

## Notes on legacy examples

Several older example scripts were removed during P1 scripts cleanup because they
depended on model-base surfaces that were already pruned from runtime. If you need
the old behavior, use git history for reference and prefer rebuilding on top of
the maintained scripts listed above.
