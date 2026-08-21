# P1-15 Pydantic Config Schemas — Execution Plan

**Date:** 2026-08-20
**Task ID:** `pydantic-config-schemas`
**Status:** Phase 1 complete; Phase 2 pilot adapter implementation in progress

## What this task means (plain English)

Today, many config paths accept loosely typed dict/dataclass inputs.
This task introduces **Pydantic v2 schemas** as the validation layer so config files are:

- validated at load time with clear errors,
- strongly typed in code,
- versionable/documentable (JSON schema output),
- safer to evolve without silent misconfiguration.

This is **not** a behavior rewrite of model training/eval logic; it is a config contract hardening effort.
We do **not** need to preserve legacy functionality, we don't want to introduce other compatibility maintenance burden.

## Goals

1. Validate model/data/evaluation config inputs before runtime execution.
2. Keep migration incremental (no flag-day break).
3. Preserve existing entrypoint behavior while adding strict validation boundaries.
4. Generate machine-readable schema docs from source-of-truth models.
5. Have consistent configuration settings across all models and datasets, with extra field handling when necessary.
6. Reassess what the proper design pattern should be for configurations, the current organization doesn't make perfect sense, this is a good opportunity to reorganize the configuration storage structure.
7. Clean-up and clear out old legacy configuration files and documentation that are from very old iterations of the code base. I.e. don't treat documents in /configs/ as ground truth. Update it to our current understanding of design. Treat this as a way to understand how it used to work, but all of this needs to be updated to present functionality.

## Non-goals

- Rewriting all workflow logic in one pass.
- Introducing parallel config systems long-term.

## Proposed rollout

### Phase 1 — Schema foundation

- Create schema package: `src/config/schemas/`.
- Add shared base model policy (strict mode, extra field handling, aliases).
- Add loader helpers:
  - parse YAML → schema validate → normalized typed object
  - emit actionable validation errors with file path and field path.

### Phase 2 — Pilot family migration

- Pick one active model family as pilot (recommend Darts/TSMixer path due recent runtime hardening).
- Add schema + adapter layer from schema object to runtime config class.
- Standardize a shared model-config routing registry (schema + runtime adapter pairs) so each future model migration follows the same wiring pattern.
- Cover with focused tests for valid + invalid configs.

### Phase 3 — Broaden by domain

- Expand migration slices:
  1. model configs
  2. data configs
  3. evaluation/workflow configs
- Add compatibility shims where needed during transition (explicitly time-bounded).

### Phase 4 — Consolidation

- Remove obsolete duplicate validation paths.
- Generate JSON schema artifacts for docs.
- Finalize contributor documentation for adding/modifying config schema fields.

## Validation plan

- Unit tests for schema parsing and error reporting.
- Regression tests for existing canonical entrypoints using migrated configs.
- `ruff` + targeted pytest on touched modules.
- Smoke-check that previously valid configs still run unchanged unless intentionally tightened.

## Risks and controls

- **Risk:** partial migration causes dual-path confusion.
  **Control:** single loader helper and explicit migration table per domain.

- **Risk:** over-strict schema breaks common configs.
  **Control:** pilot first, capture real-world config examples, tighten iteratively.

- **Risk:** undocumented behavior changes.
  **Control:** changelog notes + updated contributor docs in same PR slices.

## Exit criteria

- Canonical model/data/eval config paths validate through Pydantic schemas.
- Existing runtime entrypoints pass targeted regression checks.
    - A clear entrypoint document is created with the main useablility maintained entrypoints clearly defined.
- Schema docs are generated from code and linked in docs.
- Deprecated pre-schema validation paths are removed or explicitly sunset-tracked.

## Immediate next slice (what to implement next)

1. ✅ Expand pilot coverage beyond model config load (workflow request + data holdout + evaluation feature-override schema lanes wired).
2. ✅ Add fixture-backed regression tests against active `configs/models/tsmixer/*.yaml` profiles.
3. Draft the Phase 3 migration table (model/data/eval ownership and target schema classes).

## Phase 3 migration table (draft)

| Domain | Current runtime surface | Current contract path | Target schema module | Migration owner |
|---|---|---|---|---|
| Model config (pilot complete) | `src/workflows/forecasting/modeling.py` (`load_model_config_from_yaml`, `ModelFactory.create_model`) | `src/config/schemas/model_configs.py` route registry + adapter | `src/config/schemas/model_configs.py` | Forecasting workflow |
| Data holdout config | `src/data/versioning/dataset_registry.py` + `src/data/versioning/holdout_manager.py` | Dataclass validation in `src/data/versioning/holdout_config.py` | `src/config/schemas/data_configs.py` (active) | Data/versioning |
| Workflow/evaluation request config | `src/workflows/forecasting/pipeline.py` (`ForecastingWorkflowRequest`, `argparse` in `run_with_args`) + `src/workflows/forecasting/evaluation.py` | Ad-hoc dict/arg parsing (`model_config_overrides`, CLI args) | `src/config/schemas/workflow_configs.py` (active for request lane) | Forecasting workflow |

### Phase 3 rollout order

1. ✅ Data holdout schema adapter (`configs/data/holdout*` files) with parity checks against `HoldoutConfig`.
2. ✅ Workflow request schema for `ForecastingWorkflowRequest`/CLI normalization.
3. ✅ Evaluation override schema for the subset read in `evaluation.py` and `pipeline.py`.
