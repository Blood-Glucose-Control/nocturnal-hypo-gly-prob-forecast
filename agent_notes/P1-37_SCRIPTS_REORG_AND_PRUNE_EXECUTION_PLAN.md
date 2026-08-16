# P1-37 Scripts Reorg/Prune Execution Plan

Refreshed on 2026-08-14 after PR #433 merge and a full re-audit of `scripts/`.

---

## 0) Execution log

### 2026-08-13 (completed)

- Migrated plan from `scripts/scripts_dir_cleanup.md` to `agent_notes/`.
- Deleted low-risk dead/placeholder surfaces and stale example launchers.
- Updated docs to remove stale script references.
- Set `scripts/training/slurm/{single_gpu,multi_gpu,adaptive_resources}.sh`
  to explicit fail-fast deprecation mode pending rewiring.

### 2026-08-14 (morning start-up complete)

- Confirmed PR #433 merged to `main`.
- Fast-forwarded local `main` to `713f577`.
- Cleaned stale local branches:
  - deleted `p1-scripts-phase-b`
  - deleted `p1-scripts-phase-b2`
- Created Phase C branch:
  - `p1-scripts-phase-c-launcher-rewire`
- Re-audited current tracked scripts (HEAD on new branch):
  - total tracked files in `scripts/`: **108**
  - Python: **46**
  - Shell: **59**
  - Markdown: **3**
- Confirmed baseline lint status:
  - `ruff check scripts --statistics` -> **0 issues**

### 2026-08-14 (Phase C implementation pass)

- Rewired generic SLURM launchers to the maintained workflow entrypoint:
  - `scripts/training/slurm/single_gpu.sh`
  - `scripts/training/slurm/multi_gpu.sh`
  - `scripts/training/slurm/adaptive_resources.sh`
  - all now invoke `scripts/experiments/run_holdout_generic_workflow.sh`
- Preserved backward-compatible variable aliases:
  - `CONFIG_PATH`, `DATA_CONFIG`, `OUTPUT_DIR`, `EXPERIMENT_NAME`
- Added non-destructive validation mode for launchers:
  - `DRY_RUN=1`
- Updated docs:
  - `scripts/training/slurm/README.md`
  - `docs/user-guide/example-scripts-guide.md`
- Validation completed:
  - `bash -n` on all three rewired launchers
  - dry-run invocation on all three launchers
- Deferred validation task (manual, private cluster):
  - execute real SLURM smoke runs after cloning on cluster
  - runbook: `docs/user-guide/slurm-cluster-smoke-test-runbook.md`
  - checklist: `docs/user-guide/slurm-cluster-smoke-test-checklist.md`

### 2026-08-14 (Phase D1 progress pass)

- Curated `scripts/examples` to keep onboarding-focused surfaces only.
- Moved model-specific scripts into `scripts/experiments/`:
  - `scripts/examples/example_chronos2_finetune.py` -> `scripts/experiments/chronos2_finetune.py`
  - `scripts/examples/ttm_holdout_workflow.py` -> `scripts/experiments/ttm_holdout_workflow.py`
- Updated references and docs:
  - `scripts/training/slurm/chronos2_finetune.sh`
  - `docs/user-guide/example-scripts-guide.md`
- Added private-cluster operator materials:
  - `docs/user-guide/slurm-cluster-smoke-test-runbook.md`
  - `docs/user-guide/slurm-cluster-smoke-test-checklist.md`
- Hardened wrapper command execution:
  - `scripts/experiments/run_holdout_generic_workflow.sh` now uses argument arrays
    and `PIPESTATUS` exit capture (removed temp-file + string `eval` path)

### 2026-08-14 (D2 planning pass: generic workflow replacement)

- Mapped active generic workflow runtime callers:
  - `scripts/experiments/run_holdout_generic_workflow.sh` -> Python entrypoint
    `scripts/examples/example_holdout_generic_workflow.py`
  - 15 experiment training sweep scripts invoke the wrapper.
  - rewired SLURM generic launchers (`single_gpu`, `multi_gpu`, `adaptive_resources`)
    invoke the wrapper.
  - `scripts/training/slurm/chronos2_holdout_workflow.sh` still invokes
    `scripts/examples/example_holdout_generic_workflow.py` directly.
- Identified critical contract fields used by callers:
  - `MODEL_TYPE`, `MODEL_CONFIG`, `DATASETS`, `CONFIG_DIR`, `SKIP_TRAINING`,
    `SKIP_STEPS`, `OUTPUT_BASE_DIR`, `RUN_ID`, `VENV_NAME`, `EPOCHS`, `BATCH_SIZE`.
- Added replacement plan to split "production orchestrator" vs "light example"
  while preserving current caller contracts during migration.

### 2026-08-14 (D2 execution pass: steps 1-5 completed)

- Extracted reusable workflow core module:
  - `src/experiments/holdout_workflow_orchestrator.py`
  - Added typed request dataclass: `HoldoutWorkflowRequest`.
- Introduced production orchestrator CLI:
  - `scripts/experiments/holdout_workflow_orchestrator.py`
- Reduced onboarding example to a thin shim:
  - `scripts/examples/example_holdout_generic_workflow.py` now delegates to core.
- Migrated runtime callers to production orchestrator:
  - `scripts/experiments/run_holdout_generic_workflow.sh`
  - `scripts/training/slurm/chronos2_holdout_workflow.sh`
- Added deterministic generic E2E regression profile:
  - `scripts/experiments/holdout_workflow_regression_smoke.sh`
- Updated related docs to distinguish production vs example surfaces.

### 2026-08-14 (Phase F1 kickoff: repository design handbook draft)

- Added first draft handbook:
  - `docs/architecture/repository-design-handbook.md`
- Added handbook to docs navigation:
  - `mkdocs.yml`
  - `docs/index.md`
- Anchored draft to external guidance (Diátaxis, PyPA `src` layout,
  Cookiecutter Data Science, Twelve-Factor config) and mapped these
  to repository-specific boundaries.
- Added explicit follow-on adoption plan to formalize naming/layering
  rules before additional refactors.

### 2026-08-14 (Phase F1 review update: decision capture)

- Confirmed architecture decision:
  - adopt `src/workflows/` as top-level package for multi-step runtime pipelines.
- Confirmed orchestration direction decision:
  - Python-first orchestrators with optional thin shell wrappers for SLURM/env setup.
- Confirmed pre-MLflow metadata decision:
  - adopt a canonical run-manifest schema now and map it 1:1 into MLflow fields later.
- Added Phase F ratification checklist PR plan artifact:
  - `agent_notes/P1-37_PHASE_F_RATIFICATION_CHECKLIST_PR_PLAN.md`

### 2026-08-14 (PR-F1 completed: handbook ratification pass)

- Promoted repository design handbook to ratified v1:
  - `docs/architecture/repository-design-handbook.md`
- Added contributor-facing boundary checklist:
  - `docs/contributing.md`
- Confirmed docs navigation and index links:
  - `mkdocs.yml`
  - `docs/index.md`
- Marked PR-F1 checklist items complete in:
  - `agent_notes/P1-37_PHASE_F_RATIFICATION_CHECKLIST_PR_PLAN.md`

### 2026-08-14 (PR-F2 completed: workflows package scaffold + holdout migration)

- Created top-level workflows package:
  - `src/workflows/`
  - `src/workflows/holdout/`
- Migrated holdout multi-step pipeline module to taxonomy-aligned path:
  - `src/workflows/holdout/pipeline.py`
- Kept compatibility shim for old import path:
  - `src/experiments/holdout_workflow_orchestrator.py`
- Updated production/example CLIs to import canonical workflow path:
  - `scripts/experiments/holdout_workflow_orchestrator.py`
  - `scripts/examples/example_holdout_generic_workflow.py`
- Validation completed:
  - `ruff check` on touched Python files
  - `py_compile` import-path checks
  - `--help` checks for production and example entrypoints

### 2026-08-14 (PR-F2b planning update: boundary cleanup scope added)

- Added explicit post-migration cleanup checklist to ratification PR plan:
  - `agent_notes/P1-37_PHASE_F_RATIFICATION_CHECKLIST_PR_PLAN.md`
- Scope now explicitly includes:
  - compatibility shim removal
  - taxonomy-aligned module title finalization
  - public-release docstring tone updates
  - model-factory/config-loading extraction to canonical shared modules
  - helper-function decomposition
  - mixed-responsibility boundary resolution

### 2026-08-14 (PR-F3 completed: Python-first orchestrator baseline)

- Implemented representative Python-first sweep orchestrator path:
  - `src/workflows/holdout/orchestrators/chronos2_sweep_train.py`
  - CLI entrypoint: `scripts/experiments/chronos2_sweep_train.py`
- Reduced shell wrapper to thin launcher:
  - `scripts/experiments/chronos2_sweep_train.sh`
- Preserved environment-variable contract (`GPUS`, `JOBS_PER_GPU`,
  `CONFIG_DIR`, `SKIP_STEPS`) and added optional dry-run guardrail:
  - `DRY_RUN=1`
- Added operational usage docs for local and SLURM patterns:
  - `docs/user-guide/example-scripts-guide.md`
- Validation completed:
  - `bash -n` on touched shell wrappers
  - `ruff check` + `py_compile` on orchestrator Python files
  - `--help` and dry-run path checks for orchestrator CLI

### 2026-08-14 (PR-F2b + PR-F3b joint implementation pass)

- Removed temporary compatibility shim path:
  - deleted `src/experiments/holdout_workflow_orchestrator.py`
- Updated holdout pipeline public-facing module docstring tone and moved shared
  runtime GPU cache clearing to a dedicated runtime utility:
  - `src/workflows/runtime/hardware.py`
- Shifted sweep orchestration to a model-agnostic core:
  - `src/workflows/sweeps/train.py`
  - `scripts/experiments/sweep_train.py`
- Converted Chronos-2 training sweep path to a thin profile wrapper over the
  generic orchestrator:
  - `src/workflows/holdout/orchestrators/chronos2_sweep_train.py`
- Added declarative Chronos-2 sweep profile config:
  - `configs/experiments/nocturnal_forecast/chronos2_holdout_train_sweep.yaml`
- Updated usage docs to show generic and profile-oriented sweep entrypoints:
  - `docs/user-guide/example-scripts-guide.md`

### 2026-08-14 (PR-F2b naming reorganization pass: holdout -> forecasting)

- Moved canonical workflow module namespace to:
  - `src/workflows/forecasting/pipeline.py`
- Added compatibility wrappers for legacy import paths:
  - `src/workflows/holdout/pipeline.py`
  - `src/workflows/holdout/orchestrators/chronos2_sweep_train.py`
- Moved Chronos-2 sweep profile wrapper to canonical forecasting namespace:
  - `src/workflows/forecasting/orchestrators/chronos2_train_sweep_profile.py`
- Rewired production/example/sweep entrypoint imports to canonical path:
  - `scripts/experiments/holdout_workflow_orchestrator.py`
  - `scripts/examples/example_holdout_generic_workflow.py`
  - `scripts/experiments/chronos2_sweep_train.py`
- Renamed Chronos-2 sweep profile config to taxonomy-aligned title:
  - `configs/experiments/nocturnal_forecast/chronos2_forecasting_train_sweep.yaml`

### 2026-08-16 (scripts-surface cleanup continuation: generic launchers)

- Added generic sweep evaluation CLI entrypoint:
  - `scripts/experiments/sweep_eval.py`
- Added canonical taxonomy-aligned shell launchers:
  - `scripts/training/sweeps/run_sweep_train.sh`
  - `scripts/evaluation/sweeps/run_sweep_eval.sh`
- Removed model-specific Chronos-2 Python profile wrappers:
  - `scripts/experiments/chronos2_sweep_train.py` (deleted)
  - `scripts/experiments/chronos2_sweep_eval.py` (deleted)
  - `src/workflows/forecasting/orchestrators/*` Chronos-2 profile wrappers (deleted)
- Kept compatibility shell wrappers in `scripts/experiments/` as thin delegators
  to canonical launchers while references migrate:
  - `scripts/experiments/chronos2_sweep_train.sh`
  - `scripts/experiments/chronos2_sweep_eval.sh`

---

## 1) Current-state snapshot (facts)

### 1.1 Largest scripts (highest decomposition value)

- `src/workflows/forecasting/pipeline.py` (workflow core; migrated from `src/experiments` path)
- `scripts/experiments/per_patient_finetune.py` (887 lines)
- `scripts/visualization/compare_forecasts.py` (761 lines)
- `scripts/experiments/sliding_window_eval.py` (716 lines)
- `scripts/experiments/ttm_holdout_workflow.py` (649 lines)
- `scripts/visualization/plot_pit_horizon_heatmap.py` (659 lines)

### 1.2 Structural findings

1. **Examples directory has been partially curated.**
   - model-specific workflows moved to `scripts/experiments/`
     (`chronos2_finetune.py`, `ttm_holdout_workflow.py`).
2. **Canonical local wrapper is operational and safer, but still large.**
   - `scripts/experiments/run_holdout_generic_workflow.sh` now uses safer command
     argument assembly and pipeline exit-code handling.
     - It is currently the dependency hub for sweep training wrappers and rewired
       SLURM launchers.
3. **SLURM generic launchers are rewired locally; cluster smoke pending.**
   - `single_gpu.sh`, `multi_gpu.sh`, `adaptive_resources.sh` now route to the
     maintained generic workflow wrapper.
   - real `sbatch` smoke validation is deferred to private-cluster execution.
4. **Sweep train/eval shell duplication is high.**
   - Similarity scan shows many pairs at >=0.92 (e.g. DeepAR/PatchTST/TFT variants).
5. **Orchestration is fragmented.**
   - We have script-per-model plus ad-hoc chain scripts (`run_ctx_ablation_sweeps.sh`,
     `run_overnight_deep_sweeps.sh`, `chronos2_sweep.sh`) with overlapping worker logic.
6. **Data processing loaders are repetitive and hardcoded.**
   - dataset-specific shell wrappers are near-identical and include user-path assumptions.

---

## 2) Public-branch script principles (decision baseline)

These are the principles for what we keep publicly visible and maintained:

1. **Onboarding-first examples only in `scripts/examples/`.**
   - Keep minimal, reproducible, model-agnostic examples.
   - Move model-specific deep dives to experiment/recipe paths.
2. **One responsibility per script.**
   - Shell: orchestration/wrapping only.
   - Python: actual pipeline logic, reusable internals, typed interfaces.
3. **No redundant near-clone launchers.**
   - Prefer parameterized orchestrators over copied script families.
4. **No user-specific paths or machine assumptions.**
   - All env/path resolution must be repo-relative or configurable inputs.
5. **Naming consistency and discoverability.**
   - `example_*.py` strictly for onboarding examples.
   - Sweep/orchestration scripts follow explicit `<experiment>_<mode>.{py,sh}`.
6. **Quality bar for retained scripts.**
   - `ruff check scripts` stays clean.
   - shell scripts pass `bash -n`.

---

## 3) Target shape for `scripts/` (incremental)

### 3.1 Examples (public onboarding surface)

Keep as the stable baseline:

- `scripts/examples/example_holdout_generic_workflow.py` (short-term; later slimmed)
- `scripts/examples/example_data_holdout_system.py`
- `scripts/examples/example_load_holdout_data.py`

Relocate out of `examples/`:

- `scripts/examples/example_chronos2_finetune.py` -> `scripts/experiments/chronos2_finetune.py` (done)
- `scripts/examples/ttm_holdout_workflow.py` -> `scripts/experiments/ttm_holdout_workflow.py` (done)

### 3.2 Orchestration

Move toward:

- one reusable orchestrator implementation (Python-first),
- thin shell entrypoints (optional),
- experiment profile/config directory as the primary input.

---

## 4) Phased execution plan

### Phase C (current branch) — launcher rewire + contract hardening

Branch: `p1-scripts-phase-c-launcher-rewire`

Scope:

1. Choose one maintained training/eval entrypoint for SLURM generic launchers.
2. Rewire:
   - `scripts/training/slurm/single_gpu.sh`
   - `scripts/training/slurm/multi_gpu.sh`
   - `scripts/training/slurm/adaptive_resources.sh`
3. Keep env vars and caller contract stable where practical.
4. Update docs:
   - `scripts/training/slurm/README.md`
   - `docs/user-guide/example-scripts-guide.md`

Validation:

- `bash -n` on touched shell scripts.
- one no-op/help path invocation of rewired entrypoint.
- targeted regression checks for workflow summaries as needed.
- manual private-cluster smoke run (deferred):
  - see `docs/user-guide/slurm-cluster-smoke-test-runbook.md`

### Phase D — examples curation + decomposition

D1. **Examples curation**

1. Keep only onboarding-focused examples in `scripts/examples/`.
2. Rename/move model-specific example scripts to experiment/recipe locations.
3. Simplify `example-scripts-guide.md` to clearly separate:
   - onboarding examples
   - model-specific experiments
   - production/orchestration launchers

D2. **Decompose large Python scripts into reusable modules**

Priority order:

1. `example_holdout_generic_workflow.py` -> split into reusable workflow module + thin CLI. ✅ completed
2. `nocturnal_hypo_eval.py` + `nocturnal_hypo_eval_ctx_ablation.py` -> unify shared core.
3. `per_patient_finetune.py` -> isolate pipeline, IO, and orchestration logic.
4. `compare_forecasts.py` and large plotting scripts -> separate data load/transform/plot.

#### D2-A: `example_holdout_generic_workflow.py` replacement plan (contract-safe)

Goal: separate concerns without breaking current training/orchestration callers.

1. **Extract reusable core workflow module (no behavior change first)**
   - New module under `src/` (workflow core + typed request/result dataclasses).
   - Move step orchestration logic out of `scripts/examples/example_holdout_generic_workflow.py`.
   - Keep current step semantics (`skip_steps`) and artifact layout intact.

2. **Introduce a production orchestrator CLI**
   - New script in `scripts/experiments/` as the stable runtime surface for wrappers.
   - Accept current contract args used by sweep/SLURM wrappers:
     `model_type`, `model_config`, `datasets`, `config_dir`, `skip_training`,
     `skip_steps`, `output_dir`, `epochs`, `batch_size`.
   - Preserve output directory and file naming expected by sweep manifests.

3. **Keep a lightweight onboarding example CLI**
   - Reduce `scripts/examples/example_holdout_generic_workflow.py` to a thin
     user-facing example entrypoint (quickstart defaults, clear docs).
   - Internally call shared core module (not own the heavy orchestration logic).

4. **Compatibility migration layer**
   - Update `scripts/experiments/run_holdout_generic_workflow.sh` to call the new
     production orchestrator CLI first.
   - Keep `scripts/examples/example_holdout_generic_workflow.py` temporarily as a
     compatibility shim to avoid immediate breakage.
   - Repoint `scripts/training/slurm/chronos2_holdout_workflow.sh` to wrapper or
     production orchestrator (remove direct old-entrypoint dependency).

5. **Regression harness role**
   - Define one deterministic "generic e2e regression" invocation profile
     (train-focused + bounded config), executable locally and on SLURM.
   - Treat this as a major-change guardrail for model/runtime refactors.
   - Keep smoke profile docs in:
     - `docs/user-guide/slurm-cluster-smoke-test-runbook.md`
     - `docs/user-guide/slurm-cluster-smoke-test-checklist.md`

6. **Sequencing**
   - PR 1: extract core + new production orchestrator + zero caller changes.
   - PR 2: switch wrapper + direct SLURM caller(s) to production orchestrator.
   - PR 3: slim onboarding example + refresh docs + remove compatibility shim
     once no runtime callers depend on old path.

### Phase E — unified sweep orchestration foundation (new)

Goal: remove sweep-script redundancy and support future tracking backends.

Deliverables:

1. Introduce a single sweep orchestrator interface (train/eval/both) that accepts:
   - experiment type/profile
   - config directory (or explicit config list)
   - datasets/group key
   - backend options (local, SLURM)
2. Convert duplicated model sweep shells into thin wrappers or profile files.
3. Establish extension points for:
   - MLflow
   - Optuna
   - Weights & Biases

Note: this is intentionally a separate phase from Phase C to avoid overloading the
launcher rewire PR and to preserve behavior safety.

### Phase F — repository design handbook + boundary standardization (new)

Goal: codify a stable repository architecture and naming policy for public
release, then apply it to remaining refactors.

F1. **Handbook draft (this pass)**

1. Draft repository design handbook in docs with:
   - `src` vs `scripts` boundary rules
   - naming taxonomy (`pipeline`, `workflow`, `orchestrator`, `example`)
   - data/validation/config policies
   - Diátaxis-aligned docs architecture mapping
2. Reference external standards and state explicit adoption decisions.

F2. **Review and ratification**

1. Team review of open decisions in handbook.
2. Freeze v1 terminology and directory rules.
3. Update contributor docs with short boundary checklist.
4. Execute Phase F ratification checklist PR sequence:
   - see `agent_notes/P1-37_PHASE_F_RATIFICATION_CHECKLIST_PR_PLAN.md`.

F3. **Implementation alignment**

1. Apply ratified rules to holdout workflow decomposition follow-ups.
2. Apply same rules to nocturnal eval unification and remaining large scripts.
3. Enforce consistency with targeted lint/syntax and no-stale-doc checks.
4. Execute PR-F2b cleanup sequence (shim removal + boundary decomposition)
   before closing holdout workflow restructuring.

---

## 5) Acceptance criteria / risk controls

1. No behavior regressions to critical experiment outputs unless explicitly scoped.
2. No stale doc references to moved/deleted scripts.
3. Scripts retained in public-facing paths have clear purpose and non-overlapping role.
4. `ruff check scripts` remains clean (currently green baseline).
5. Critical experiment entrypoints remain runnable:
   - `scripts/experiments/nocturnal_hypo_eval.py`
   - `scripts/experiments/sliding_window_eval.py`
   - `scripts/experiments/per_patient_finetune.py`

---

## 6) Decisions currently resolved

1. Legacy stale/broken examples should be hard-deleted (already done).
2. Generated competition submission artifacts are not script-source and should not live
   in maintained runtime script surfaces.
3. Sweep orchestration unification is approved as a dedicated follow-on phase (Phase E),
   not bundled into Phase C launcher rewiring.
