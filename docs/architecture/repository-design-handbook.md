# Repository Design Handbook (Ratified v1)

Status: Ratified v1 (2026-08-14, P1-37 Phase F1; PR-F2 scaffold complete)

## 1) Purpose

This handbook defines how we organize code, scripts, configs, experiments, and
documentation for an initial public release of this repository.

Primary goals:

1. Make the repository understandable for new researchers.
2. Keep reusable logic in stable Python modules.
3. Keep orchestration explicit and reproducible.
4. Avoid script sprawl and duplicate launcher logic.

## 2) External guidance reviewed (and adopted)

There is no single universal standard for ML repository design. We adopt a
composed approach from reputable sources:

1. **Diátaxis** (documentation architecture):
   - https://diataxis.fr/start-here/
   - Adopted for docs taxonomy (tutorial/how-to/reference/explanation).
2. **PyPA `src` layout guidance** (Python package/runtime boundaries):
   - https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/
   - Adopted for keeping importable runtime code under `src/`.
3. **Cookiecutter Data Science opinions** (data pipeline and reproducibility):
   - https://cookiecutter-data-science.drivendata.org/
   - Adopted for immutable raw data and DAG-like pipeline thinking.
4. **Twelve-Factor config principle** (deploy-specific configuration):
   - https://12factor.net/config
   - Adopted for separating deploy/runtime environment configuration from code.

## 3) Repository mental model

### 3.1 Data processing

- Data ingestion/cleaning/holdout preparation lives in `src/data/`.
- Prepared data snapshots are treated as versioned artifacts.
- Validation reports are required for prepared snapshots.
- E2E scripts should verify that required validation artifacts exist rather than
  re-validating unchanged snapshots by default.

### 3.2 Model training core

- Reusable model logic lives in `src/models/`.
- Training/inference code should accept:
  - one or more data config files (or a config directory),
  - one model config file.

### 3.3 Sweep orchestration

- Sweep launchers can remain in `scripts/` as thin orchestration entrypoints.
- Their role is job orchestration (local shell/tmux/SLURM), not business logic.
- Reusable scheduling/manifest logic should be centralized in `src/` modules
  and called by thin wrappers.

### 3.4 Evaluation core

- Metrics, statistical analyses, and evaluation helpers live in `src/evaluation/`
  and related `src/experiments/*` modules.

### 3.5 Experiments

- Experiments combine data snapshot(s), model configuration(s), and evaluation
  definition(s) to answer a specific research question.
- MLflow is the target experiment-tracking model; manual workflows should match
  the same run/metadata shape where practical.

### 3.6 Visualization

- Plot/report generation utilities live in `scripts/visualization/` and reusable
  plotting utilities in `src/` when shared.

## 4) Layer boundaries and ownership

### `src/` (importable runtime code)

Use `src/` for reusable, testable modules:

- pipelines/workflows
- model factory/model adapters
- dataset preparation/validation
- evaluation and metrics
- shared utility modules

Avoid in `src/`:

- cluster-specific shell assumptions
- hardcoded user paths
- one-off experiment shell orchestration

### `scripts/` (entrypoints and wrappers)

Use `scripts/` for:

- user-facing examples/tutorial scripts
- launchers (local, tmux, SLURM)
- thin wrappers over `src` modules

Avoid in `scripts/`:

- duplicated business logic across many scripts
- long-lived model factories/config parsing duplicated from `src`

### `configs/` (declarative inputs)

- Model configs, data configs, and experiment/sweep profiles are declarative.
- Config schema evolution should be explicit and documented.

### `docs/` (Diátaxis)

- Tutorials: teaching-oriented walkthroughs
- How-to guides: task-oriented operational instructions
- Reference: exact CLI/config/API semantics
- Explanation: design rationale and tradeoffs (this handbook lives here)

## 5) Naming conventions

Use names that communicate role:

1. `*_pipeline.py`: executes one bounded pipeline run.
2. `*_workflow.py`: similar to pipeline, usually user-facing sequence.
3. `*_orchestrator.py` / `*_sweep*.sh`: coordinates multiple runs/jobs.
4. `example_*.py`: pedagogical example with clear narrative and minimal hidden
   behavior.

### Terminology quick reference

| Term | Primary purpose | Where it should live |
|---|---|---|
| `pipeline` | Execute one bounded runtime flow | `src/workflows/` |
| `workflow` | User-facing sequence over one pipeline run | `src/workflows/` (core) + thin `scripts/` entrypoint |
| `orchestrator` | Coordinate multiple runs/jobs (sweeps, batches) | Python-first in `src/workflows/` with thin shell launch wrappers |
| `example` | Teach users how to use core functionality | `scripts/examples/` |

## 6) Decomposition rules for Python modules

1. Keep model factory/registry logic in model-layer modules (`src/models/*`),
   not inside workflow entry modules.
2. Keep config loading/parsing in shared config modules (`src/utils/*` or a
   dedicated config package), not duplicated per workflow.
3. Keep hardware/runtime detection helpers in shared runtime utilities if reused
   across multiple pipelines.
4. Keep plotting and evaluation transformations in dedicated modules, imported by
   workflows.

## 7) Data and validation policy

1. Raw data is immutable.
2. Processed snapshot outputs are deterministic for fixed inputs/config.
3. Validation reports are persisted with versioned snapshots.
4. E2E pipelines default to validating report presence/status before training,
   with explicit opt-in for full re-validation.

## 8) Configuration policy

1. Code-level defaults: safe and explicit.
2. Research configuration: YAML in `configs/`.
3. Deploy/runtime-specific values (paths, credentials, cluster env) are passed
   via environment variables and launcher arguments.
4. No credentials in tracked config files.

## 9) Quality gates

Minimum bar for script and workflow changes:

1. `ruff check` for touched Python modules.
2. `bash -n` for touched shell scripts.
3. Targeted functional validation for changed runtime paths.
4. Docs updated when entrypoints, paths, or behavioral contracts change.

## 10) Adoption plan (Phase F rollout)

1. **Ratify boundaries and naming**
   - Confirm taxonomy (`pipeline`, `orchestrator`, `example`) and directory roles.
2. **Normalize forecasting workflow surfaces**
   - Rename/refactor forecasting runtime modules to match approved taxonomy.
   - Extract model factory/config/runtime helpers into proper shared modules.
3. **Apply to nocturnal eval unification**
   - Reuse the same boundary rules while merging shared eval cores.
4. **Document contributor rules**
   - Add concise boundary checklist to `docs/contributing.md`.
5. **Enforce with CI-adjacent checks**
   - Keep lint/syntax checks green and ensure no stale docs references.

## 11) Decisions and remaining open items

### 11.1 Confirmed decisions

1. Create `src/workflows/` as a top-level package for multi-step runtime
   pipelines.
2. Use **Python-first orchestration** for experiment/sweep control flow, with
   optional thin shell wrappers for environment activation and SLURM submission.
3. Adopt a lightweight **pre-MLflow canonical run manifest** now, then map it
   1:1 into MLflow tags/params/metrics/artifacts later.

### 11.2 Remaining open items

1. Extend the task/experiment sweep dispatcher pattern (`src/workflows/sweeps/`)
   with additional adapters beyond the initial
   `forecasting/nocturnal_forecast` implementation.
2. Continue extending the initial `src/workflows/forecasting/` placement pattern
   to additional shared pipeline components during transition.

## 12) Canonical experiment metadata schema (pre-MLflow decision)

Every experiment run should persist a machine-readable run manifest (JSON)
containing:

1. **Identity**
   - `run_id`, `parent_run_id` (optional), `workflow_name`, `workflow_version`
2. **Timing**
   - `created_at_utc`, `started_at_utc`, `ended_at_utc`, `duration_seconds`
3. **Code provenance**
   - `git_commit`, `git_branch`, `git_dirty`, `repository`
4. **Execution context**
   - `launcher_type` (`local|tmux|slurm`), `host`, `user`, `python_version`,
     `cuda_visible_devices`, `slurm_job_id` (optional)
5. **Inputs**
   - `data_config_paths`, `data_snapshot_ids` (or validation report IDs),
     `model_config_path`, `experiment_config_path` (optional), `seed`
6. **Resolved runtime config**
   - fully resolved key runtime parameters after default resolution
7. **Outputs**
   - `artifact_root`, `checkpoint_paths`, `prediction_paths`, `plot_paths`
8. **Result summary**
   - key metric dictionary, status (`success|failed|interrupted`),
     failure message (if failed)
