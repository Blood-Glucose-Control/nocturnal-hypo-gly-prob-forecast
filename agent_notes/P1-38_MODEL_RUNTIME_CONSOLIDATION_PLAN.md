# P1-38 Model Runtime Consolidation Plan

**Date:** 2026-08-23
**Status:** Proposed (pending execution)
**Tracking row:** `model-runtime-consolidation-wave` in [project_tracking.csv](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/project_tracking.csv)

---

## 1) Why this task exists

Several active model-family runtime modules still carry large amounts of duplicated
or model-local orchestration logic that should be shared:

- [ttm/model.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/ttm/model.py)
- [timesfm/model.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/timesfm/model.py)
- [moirai/model.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/moirai/model.py)
- [moment/model.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/moment/model.py)

Secondary candidates with medium complexity:

- [chronos2/model.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/chronos2/model.py)
- [toto/model.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/toto/model.py)

This is a **P1 stabilization/maintainability** task that should land before
Optuna/MLflow/stats/experiments-collapse so those later workstreams build on a
cleaner and more uniform model runtime surface.

---

## 2) Placement in current P1 sequence

This task should start **after**:

1. Remaining `pydantic-config-schemas` closeout for model-family rollout and
   schema-evolution contributor documentation in
   [P1-15_PYDANTIC_CONFIG_SCHEMAS_EXECUTION_PLAN.md](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/agent_notes/P1-15_PYDANTIC_CONFIG_SCHEMAS_EXECUTION_PLAN.md).
2. Canonical model extension contract publication:
   `model-extension-contract-doc`.

This task should complete **before**:

- `optuna-integration`
- `mlflow-integration`
- `stats-rigor-module`
- `experiments-collapse`

Rationale: those tasks require reliable, predictable, low-duplication model entrypoints.

---

## 3) Goals

1. Reduce duplicate runtime logic across heavy model families.
2. Move generic training/prediction/checkpoint helpers into shared bases/utilities.
3. Keep public workflow/model interfaces behavior-compatible.
4. Increase confidence via contract tests and fixture-backed regression checks.
5. Improve contributor velocity for future model-family additions/refactors.

---

## 4) Non-goals

- No model architecture redesign.
- No dataset semantics changes.
- No behavioral changes to evaluation metrics definition unless explicitly documented.
- No broad script reorg (already covered by prior script taxonomy waves).

---

## 5) Scope and candidate ranking

### Tier A (execute in this task)

1. [ttm/model.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/ttm/model.py)
2. [timesfm/model.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/timesfm/model.py)
3. [moirai/model.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/moirai/model.py)
4. [moment/model.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/moment/model.py)

### Tier B (best-effort in same wave, otherwise immediate follow-on)

5. [chronos2/model.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/chronos2/model.py)
6. [toto/model.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/toto/model.py)

### Tier C (already compact enough; monitor only)

- [tide/model.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/tide/model.py)
- [timegrad/model.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/timegrad/model.py)
- [tsmixer/model.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/tsmixer/model.py)
- thin AutoGluon wrappers already centralized by
  [autogluon_base.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/autogluon_base.py)

---

## 6) Consolidation strategy (workstreams)

### WS0 — Baseline and hotspots (no behavior change)

- Capture per-family baseline:
  - LOC and long-method inventory
  - existing schema/factory/contract test coverage
  - known edge cases (zero-shot, checkpoint reload, probabilistic outputs)
- Produce a short baseline appendix in this note before first refactor PR.

### WS1 — Shared helper extraction

Introduce/reuse shared utilities for repeated patterns:

- checkpoint save/load path probing and metadata handling
- common training-argument assembly patterns (where backend-equivalent)
- prediction tensor shape normalization / inverse scaling hooks
- panel/segment preprocessing handoff helpers

Likely homes:

- [src/models/base/base_model.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/base/base_model.py)
- [src/models/](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/) shared helper module(s), if needed

### WS2 — Tier A wave 1 (TTM pilot)

- Refactor TTM first as the proving slice.
- Keep behavior stable; reduce method size by delegating helper logic.
- Remove pass-through `_impl` indirection wrappers where they only preserve
  ordering (for example `_predict`/`_save_checkpoint`/`_load_checkpoint` in
  [ttm/model.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/ttm/model.py)),
  by relocating canonical abstract-method bodies into order and defining each
  method once.
- Add explicit regression coverage for:
  - checkpoint + preprocessor loading assumptions
  - inverse scaling behavior
  - batch predict path semantics

### WS3 — Tier A wave 2 (TimesFM + Moirai)

- Apply proven helper patterns.
- Remove duplicate local training/inference/checkpoint blocks.
- Ensure schema/factory route behavior remains unchanged.

### WS4 — Tier A wave 3 (Moment)

- Break up long training path and context/target preparation surfaces into tested helpers.
- Preserve existing forecasting API semantics.

### WS5 — Tier B completion (Chronos2 + Toto)

- Consolidate medium-complexity duplication.
- Keep Chronos2-specific special handling where required (intentional deviations documented).

### WS6 — Documentation + contributor contract updates

- Update the canonical extension/runbook docs with the final shared patterns.
- Ensure references align with:
  - [model-implementation-quality-checklist.md](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/docs/architecture/model-implementation-quality-checklist.md)
  - `model-extension-contract-doc` output

---

## 7) PR slicing plan

Use small, reviewable waves:

1. **PR-C0:** Baseline + shared helper scaffolding (no behavior change)
2. **PR-C1:** TTM consolidation
3. **PR-C2:** TimesFM + Moirai consolidation
4. **PR-C3:** Moment consolidation
5. **PR-C4:** Chronos2 + Toto consolidation + docs finalization

Each PR must include targeted tests for changed families and avoid cross-family
incidental edits.

---

## 8) Validation gates per PR wave

Minimum:

```bash
ruff check <touched model/base/workflow/schema files>
pytest -q tests/models/test_model_family_contract_suite.py
pytest -q tests/workflows/forecasting/test_model_config_schema_loader.py
```

Plus family-specific tests for each touched model module (and add missing ones when
gaps are found).

For any changed schema route behavior, also run:

```bash
pytest -q tests/workflows/forecasting/test_config_schema_artifact_generation.py
```

---

## 9) Risks and controls

1. **Risk:** Hidden behavior drift while extracting helpers.
   **Control:** no mixed behavior + refactor changes in same PR; preserve fixture-backed outputs.

2. **Risk:** Over-generalizing backend-specific code.
   **Control:** extract only patterns proven common in at least 2 families; keep explicit per-family overrides.

3. **Risk:** Contract-test blind spots.
   **Control:** expand model-family contract coverage before large wave merges.

4. **Risk:** Sequencing conflict with P1-15 closeout.
   **Control:** do not start PR-C1 until P1-15 contributor-doc pass and model-family schema rollout gate are marked done.

---

## 10) Exit criteria

Task is done when:

1. Tier A families are consolidated and validated.
2. Tier B candidates are either landed or split into a dated immediate follow-on task.
3. All affected families pass contract/schema/factory gates.
4. Shared helper surfaces are documented and referenced by contributor docs.
5. `project_tracking.csv` reflects completed state and links final artifacts.

---

## 11) Definition of readiness for downstream P1 work

`optuna-integration`, `mlflow-integration`, `stats-rigor-module`, and
`experiments-collapse` should only proceed once this task reaches done (or an
explicitly accepted partial with residuals tracked as separate blocking rows).
