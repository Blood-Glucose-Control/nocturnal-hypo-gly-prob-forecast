# Scripts Directory Cleanup Plan

Audit conducted 2026-04-27 on branch `feat/calibration-visualizations`.

---

## Already Done (this branch)

Three files were manually moved from `scripts/analysis/` → `scripts/visualization/` without `git mv`, leaving stale path
references in their docstrings. These have been patched:

| File | Fix |
|---|---|
| `scripts/visualization/compare_forecasts.py` | 4 docstring usage examples updated |
| `scripts/visualization/plot_rmse_vs_horizon.py` | 1 Usage line updated |
| `scripts/visualization/plot_step_sweep.py` | 2 Usage lines updated |

---

## Files to MOVE

### Root-level strays → `scripts/experiments/`

| Current | Destination |
|---|---|
| `scripts/chronos2_time_covariate_experiment.py` | `scripts/experiments/` |
| `scripts/verify_time_features.py` | `scripts/experiments/` |

### `scripts/training/` mis-placement

| Current | Destination | Reason |
|---|---|---|
| `scripts/training/setup_moirai_env.py` | `scripts/setup_moirai_env.py` | Its own docstring says `python scripts/setup_moirai_env.py`; logically lives next to `setup_model_env.sh` |

### Analysis scripts that belong with training infra

| Current | Destination | Reason |
|---|---|---|
| `scripts/analysis/analyze_logs.py` | `scripts/training/` | Analyzes SLURM training logs and GPU utilization |
| `scripts/analysis/analyze_resources.py` | `scripts/training/` | Analyzes SLURM resource usage across training runs |

### `scripts/examples/` — operational scripts masquerading as examples

| Current | Destination | Reason |
|---|---|---|
| `scripts/examples/run_holdout_ttm_workflow.sh` | `scripts/training/slurm/` | Has `#SBATCH` directives; identical in kind to `scripts/training/slurm/chronos2_holdout_workflow.sh` |
| `scripts/examples/run_single_gpu_ttm.sh` | `scripts/training/slurm/` | Has `#SBATCH` directives; its own comment says "for production training, use `scripts/training/slurm/single_gpu.sh`" |
| `scripts/examples/run_distributed_ttm.sh` | `scripts/training/` | Distributed training launcher (no SBATCH); fits alongside `chronos2_finetune.sh` |
| `scripts/examples/run_holdout_generic_workflow.sh` | `scripts/training/` | Explicitly described as "non-SLURM version" of the holdout workflow; it's an operational local-run script |
| `scripts/examples/ttm_holdout_workflow.py` | `scripts/training/` | Described as "a template for model-specific holdout scripts"; training workflow, not a demo |
| `scripts/examples/check_distributed_training_setup.py` | `scripts/training/` | Environment validator for distributed training; utility tool, not demo |
| `scripts/examples/show_hardware_info.py` | `scripts/training/` | Hardware diagnostic tool; same rationale |

---

## Files to RENAME

| Current | Proposed | Reason |
|---|---|---|
| `scripts/examples/load_holdout_data_example.py` | `scripts/examples/example_load_holdout_data.py` | All siblings are `example_*.py`; this is the only one with reversed naming |
| `scripts/examples/ttm_holdout_workflow.py` | `scripts/training/ttm_holdout_workflow.py` | (rename happens automatically via the move above) |

---

## Directory to RENAME

| Current | Proposed | Reason |
|---|---|---|
| `scripts/data_processing_scripts/` | `scripts/data_processing/` | `_scripts` suffix is redundant inside a `scripts/` directory; inconsistent with all other subdirectories |

---

## Files to DELETE

| File | Reason |
|---|---|
| `scripts/scratch/build_midnight_episode_experiment.py` | Docstring says "TODO: move to documentation later"; content belongs in docs, not a script |
| `scripts/code_benchmarking/benchmark_datetime_index.py` | One-off micro-benchmark for a specific internal function; no ongoing value |
| `scripts/competition_submission/NaiveForecaster_submission.csv` | Binary data artifact; belongs in a data store, not source control |
| `scripts/competition_submission/TinyTimeMixerForecaster_submission.csv` | Same as above |
| `scripts/examples/example_holdout_ttm_workflow.py` | Superseded by `example_holdout_generic_workflow.py`, which explicitly describes itself as "a refactored version" of this file |
| `scripts/examples/example_distributed_ttm.py` | `example_distributed_strategies.py` covers the same ground more comprehensively |
| `scripts/examples/example_single_gpu_ttm.py` | Thin wrapper subsumed by `example_base_framework.py` |

### Directories that become empty after deletions (also delete)

- `scripts/scratch/`
- `scripts/code_benchmarking/`
- `scripts/competition_submission/` — or keep the `README` + `kaggle_submission.sh` and move them to `scripts/data_downloads/` if the Kaggle workflow is still relevant

---

## Files that stay in `scripts/examples/` (legitimate demos)

- `example_base_framework.py` — comprehensive TSFM framework walkthrough
- `example_chronos2_finetune.py` — Chronos2 end-to-end fine-tune demo
- `example_data_holdout_system.py` — data holdout API demo
- `example_distributed_strategies.py` — distributed config patterns
- `example_holdout_generic_workflow.py` — generic multi-model holdout workflow
- `example_load_holdout_data.py` (renamed from `load_holdout_data_example.py`)

---

## `scripts/analysis/` after moves

After moving `analyze_logs.py` and `analyze_resources.py` to `scripts/training/`, the only remaining file is
`summarize_experiments.py`, which is a legitimate analysis/reporting script. The directory can stay with just that one
file, or `summarize_experiments.py` can be promoted to `scripts/` root if the directory feels too sparse.
