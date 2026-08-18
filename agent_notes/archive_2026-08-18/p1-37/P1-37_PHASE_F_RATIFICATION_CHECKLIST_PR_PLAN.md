# P1-37 Phase F Ratification Checklist PR Plan

Date: 2026-08-14
Scope: Formalize and implement repository architecture decisions that emerged
during Phase C/D execution.

## 1) Ratified decisions (must remain stable)

1. Create `src/workflows/` as the top-level package for multi-step runtime pipelines.
2. Use Python-first orchestrators, with optional thin shell wrappers for env/SLURM bootstrap.
3. Adopt a canonical pre-MLflow run manifest schema now, mapped later 1:1 to MLflow.

## 2) Why this is now in scope

This was not explicit in the original P1-37 plan, but became necessary while
decomposing and rewiring high-impact workflow scripts. Without ratified
boundaries and naming rules, follow-on refactors risk drift and repeated churn.

## 3) PR sequence (checklist)

## PR-F1: Handbook ratification + contributor boundary checklist

Goal: freeze architecture vocabulary and boundary rules.

Checklist:

- [x] Promote handbook status from draft to ratified v1 after final review edits.
- [x] Add boundary checklist summary to `docs/contributing.md`.
- [x] Ensure docs navigation includes handbook and no stale references.
- [x] Confirm terminology table (`pipeline`, `workflow`, `orchestrator`, `example`).

Validation:

- [x] docs links resolve (handbook/contributing/index).
- [x] no stale path references in updated docs.

Status note: Completed on 2026-08-14.

## PR-F2: `src/workflows/` scaffold + holdout migration skeleton

Goal: establish canonical package location and migration-safe structure.

Checklist:

- [x] Create `src/workflows/` package and subpackages for holdout runtime flow.
- [x] Move/refactor holdout multi-step pipeline entry module under `src/workflows/`.
- [x] Keep compatibility shims for current imports/callers during transition.
- [x] Rename modules to match ratified taxonomy where safe.

Validation:

- [x] `ruff check` on touched Python files.
- [x] targeted `py_compile` / import path checks for old and new entrypoints.

Status note: Completed on 2026-08-14.

## PR-F2b: Forecasting workflow boundary cleanup + shim removal

Goal: complete workflow-module quality cleanup after migration to canonical forecasting namespace.

Checklist:

- [x] Remove compatibility shim once all call sites use canonical workflow path.
- [x] Rename holdout modules to final taxonomy-aligned titles.
- [x] Move canonical workflow package namespace to `src/workflows/forecasting/`.
- [x] Rewrite module docstrings in public-release tone (no legacy framing).
- [x] Move model factory/config loading concerns out of holdout pipeline module
      into canonical shared locations.
- [x] Refactor helper functions into clear modules with single responsibilities.
- [x] Resolve mixed module boundaries so orchestration, model-construction,
      config parsing, and evaluation helpers are cleanly separated.

Status note: Completed on 2026-08-14 — canonical workflow namespace is
`src/workflows/forecasting/`, model/evaluation concerns execute from dedicated
modules (`src/workflows/forecasting/modeling.py`,
`src/workflows/forecasting/evaluation.py`), legacy duplicate in-module
implementations were removed from the pipeline module, and holdout compatibility
wrapper modules/import surfaces were retired.

Operational naming cleanup: entrypoints were renamed to forecasting-first names
(`forecasting_workflow_orchestrator.py`, `example_forecasting_workflow.py`,
`run_forecasting_workflow.sh`, `forecasting_workflow_regression_smoke.sh`,
`chronos2_forecasting_workflow.sh`) and call sites/docs were rewired.

Validation:

- [x] `ruff check` and `py_compile` on touched workflow/model/config modules.
- [x] import/call-path verification for production and example entrypoints.
- [x] no stale references to removed shim path.

## PR-F3b: Model-agnostic sweep orchestrator transition

Goal: pivot sweep orchestration from model-specific implementations to a generic
config-driven Python-first runtime.

Checklist:

- [x] Add a model-agnostic sweep training orchestrator under `src/workflows/sweeps/`.
- [x] Keep model-specific sweep entrypoints as thin profile wrappers.
- [x] Add a declarative Chronos-2 sweep profile YAML for dataset/config mapping.
- [x] Add a generic `scripts/experiments/sweep_train.py` CLI entrypoint.
- [x] Update user docs to show both profile-based and generic directory-based usage.

Validation:

- [x] `ruff check` + `py_compile` on new/updated sweep modules.
- [x] `--help` and dry-run path checks for generic and profile wrappers.

Status note: Completed on 2026-08-14.

## PR-F3: Python-first orchestrator baseline

Goal: establish reusable Python orchestration CLI pattern.

Checklist:

- [x] Introduce Python orchestrator CLI for one representative sweep path.
- [x] Keep shell wrappers thin (env activation + process launch only).
- [x] Preserve current environment variable contract expected by launchers.
- [x] Add/how update operational docs for local + SLURM usage.

Validation:

- [x] `bash -n` on touched shell wrappers.
- [x] orchestrator `--help` and no-op/smoke path checks.

Status note: Completed on 2026-08-14.

## PR-F4: Run-manifest v1 implementation slice

Goal: start canonical metadata logging before MLflow full integration.

Checklist:

- [x] Define a run-manifest writer utility (JSON schema fields per handbook).
- [x] Wire manifest output into one training workflow and one evaluation workflow.
- [x] Persist manifest with run artifacts/logs.
- [x] Document manifest location/fields in reference docs.

Validation:

- [x] verify manifest file existence and required fields on sample runs.
- [x] verify failure status captures error summary fields.

Status note: Completed on 2026-08-14 — canonical manifest utility added at
`src/workflows/runtime/manifest.py`; manifest emission wired into
`src/workflows/forecasting/pipeline.py` and
`scripts/experiments/nocturnal_hypo_eval.py`; reference documentation updated in
`docs/user-guide/example-scripts-guide.md`; success and failure-path manifest
output validated with sample runs.

## 4) Exit criteria for Phase F ratification workstream

1. Handbook is ratified and linked from contributor-facing docs.
2. `src/workflows/` exists and is in active use for at least one multi-step runtime pipeline.
3. At least one orchestrator path is Python-first with shell wrappers reduced to thin launchers.
4. Canonical run manifest v1 is emitted by representative training/eval entrypoints.
