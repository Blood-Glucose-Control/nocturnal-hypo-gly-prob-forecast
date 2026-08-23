# P1-15 Pydantic Config Schemas — Execution Plan

**Date:** 2026-08-20
**Task ID:** `pydantic-config-schemas`
**Status:** Phase 1-3 complete; Phase 4.1-4.2 complete; Phase 4.3 + model-family rollout in progress

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

1. ✅ Phase 4.1 — consolidate duplicate holdout/model-config validation entrypoints:
   - migrated `src/experiments/nocturnal/holdout_split_analysis.py` from `HoldoutConfig.load(...)` to schema loader lane;
   - migrated workflow evaluation scripts that were still loading loose model YAML dicts (`src/workflows/evaluation/nocturnal_hypo_eval.py`, `src/workflows/evaluation/sliding_window_eval.py`, `src/workflows/evaluation/validate_predict_batch.py`) to shared model-config schema loader path.
2. ✅ Phase 4.2 — generate JSON schema artifacts from active schema modules (`model_configs.py`, `data_configs.py`, `workflow_configs.py`) into docs-visible location (`docs/architecture/generated-config-schemas/`) via `python -m src.config.schemas.json_schema_artifacts`.
3. Phase 4.3 — contributor doc pass: document canonical “add a schema + adapter” workflow and deprecate any stale validation guidance.
4. Phase 4.x follow-on — execute model-family schema rollout backlog (table below) so all active model IDs use registry-backed schema adapters.

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

## Model-family schema rollout matrix (explicit backlog)

| Model ID | Runtime factory support | Schema adapter status | Next action |
|---|---|---|---|
| `tsmixer` | Active | ✅ Completed (pilot) | Keep as reference implementation for adapter pattern |
| `sundial` | Active | ✅ Completed | Schema + adapter wired; includes zero-shot-only validation plus fixture + invalid-config tests |
| `ttm` | Active | ✅ Completed | Schema + adapter wired; includes strict split/target validation and workflow factory routing |
| `chronos` | Active | ⏳ Pending | Add schema + adapter; validate zero-shot/fine-tune fields |
| `chronos2` | Active | ✅ Completed | Schema + adapter wired; active configs validate through model schema registry |
| `tide` | Active | ✅ Completed | Schema + adapter wired; enforces TiDE architectural/training constraints and `learning_rate` alias normalization to `lr` |
| `moirai` | Active | ⏳ Pending | Add schema + adapter for MOIRAI-specific args |
| `timegrad` | Active | ⏳ Pending | Add schema + adapter for TimeGrad config lane |
| `moment` | Active | ⏳ Pending | Add schema + adapter for MOMENT lane |
| `toto` | Active | ⏳ Pending | Add schema + adapter for Toto lane |
| `timesfm` | Active | ⏳ Pending | Add schema + adapter for TimesFM lane |
| `deepar` | Active | ✅ Completed | Schema + adapter wired; supports `learning_rate` alias normalization to `lr` |
| `patchtst` | Active | ✅ Completed | Schema + adapter wired; includes PatchTST head/dimension validation |
| `tft` | Active | ✅ Completed | Schema + adapter wired; supports `learning_rate` alias normalization to `lr` |
| `naive_baseline` | Active | ✅ Completed | Schema + adapter wired with explicit model-name enum (`Naive`/`Average`) |
| `statistical` | Active | ✅ Completed | Schema + adapter wired with explicit statistical-model enum + ARIMA bounds |

### Rollout execution timing

- Start immediately after Phase 4.2 schema artifact generation.
- Execute in small PR waves (2-3 model families per PR) so each wave has focused
  fixtures + invalid-config tests and fast review cycles.
- Target order: high-use families first (`chronos2`, `ttm`, `timesfm`, `tide`),
  then remaining active families.

## Phase 4 kickoff checklist (prepared)

- [x] Merge Phase 3 PR and fast-forward local `main`.
- [x] Branch housekeeping (remove stale local Phase 3 branch).
- [x] Create dedicated Phase 4 branch.
- [x] Implement validation-path consolidation changes.
- [x] Add JSON schema artifact generation.
- [ ] Final contributor documentation pass for schema evolution workflow.
