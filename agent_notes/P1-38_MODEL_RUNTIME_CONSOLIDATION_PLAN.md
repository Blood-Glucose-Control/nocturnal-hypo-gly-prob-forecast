# P1-38 Model Runtime Consolidation Plan

**Date:** 2026-08-23
**Update Date:** 2026-08-26
**Status:** In progress (all-model pre-change baseline is green, including TimeGrad on a v2-compatible lane; PR-C1 kickoff queued)
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
- [tide/model.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/tide/model.py)

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
3. `ruff-baseline-cleanup-pass` completion so this wave starts from a restored
   repo-wide lint baseline and avoids mixing baseline churn with runtime
   consolidation changes.

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

### Explicit architecture commitments (locked for P1-38)

1. **Constructor unification target is explicit:** maintained runtime entrypoints
   should converge on schema-routed `ModelFactory` constructor semantics.
2. **Current-state acknowledgment:** today, primary evaluation flows still rely on
   [`create_model_and_config`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/factory.py:35);
   this task owns migration away from that direct dependency.
3. **Shim policy:** `create_model_and_config` may exist only as a temporary
   migration shim during intermediate PR slices; no new direct call sites should
   be introduced.
4. **End-state requirement:** by task closeout, maintained runtime entrypoints are
   no longer dependent on `create_model_and_config`; final wave removes the shim
   once parity is validated.
5. **ModelRegistry boundary:** keep
   [`ModelRegistry`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/base/registry.py)
   scoped to registration/discovery/test support, not as an independent third
   runtime constructor lane.
6. **Runtime priority scope:** preserve and prioritize behavior for
   [`nocturnal_hypo_eval.py`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/workflows/evaluation/nocturnal_hypo_eval.py)
   (primary) and
   [`sliding_window_eval.py`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/workflows/evaluation/sliding_window_eval.py)
   (secondary); experimental/scratch surfaces remain non-blocking unless they
   share the maintained constructor lane.

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
7. [tide/model.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/tide/model.py)

### Tier C (already compact enough; monitor only)

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
- PR-C0 status (2026-08-25): complete. Baseline appendix is captured in
  [section 8A](#8a-pr-c0-baseline-appendix-2026-08-25).

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

### WS5 — Tier B completion (Chronos2 + Toto + Tide)

- Consolidate medium-complexity duplication.
- Keep Chronos2-specific special handling where required (intentional deviations documented).
- Clean up family-local utility module boundaries for Chronos2/Tide (for example
  [chronos2/utils.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/chronos2/utils.py) and
  shared helpers in
  [autogluon_data_utils.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/autogluon_data_utils.py))
  so model modules are not coupled to misplaced helper surfaces.

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
5. **PR-C4:** Chronos2 + Toto + Tide consolidation + docs finalization

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

Pre/post regression gate for consolidation PR waves:

```bash
# Pre-change baseline suite (all maintained models, Aleppo ultra-smoke holdout)
make smoke-suite-aleppo SUITE_LABEL=pre_<wave_label>

# Post-change suite
make smoke-suite-aleppo SUITE_LABEL=post_<wave_label>

# Compare artifact parity + key metric drift tolerance
make smoke-suite-compare \
  BASELINE=trained_models/artifacts/regression_smoke/all_models_aleppo/pre_<wave_label>/suite_manifest.json \
  CANDIDATE=trained_models/artifacts/regression_smoke/all_models_aleppo/post_<wave_label>/suite_manifest.json
```

Fast-profile notes (2026-08-26):

- Suite default holdout is [`configs/data/holdout_smoke_aleppo_ultra/aleppo_2017.yaml`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/configs/data/holdout_smoke_aleppo_ultra/aleppo_2017.yaml) to reduce train-window volume.
- Heavy families now use `97_regression_smoke_balanced.yaml` smoke configs where needed to bound runtime (notably Chronos2 step budget and AutoGluon-family `max_epochs` / `num_batches_per_epoch`).
- Chronos2 smoke keeps progress visibility with `fine_tune_logging_steps: 500` while disabling in-step validation for speed.

---

## 8A) PR-C0 baseline appendix (2026-08-25)

### LOC + long-method inventory (>=40 lines per function/method)

| Family | File | LOC | Long defs |
| --- | --- | ---: | ---: |
| TTM | [src/models/ttm/model.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/ttm/model.py) | 1107 | 10 |
| TimesFM | [src/models/timesfm/model.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/timesfm/model.py) | 1197 | 6 |
| Moirai | [src/models/moirai/model.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/moirai/model.py) | 1140 | 9 |
| Moment | [src/models/moment/model.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/moment/model.py) | 1133 | 7 |
| Chronos2 | [src/models/chronos2/model.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/chronos2/model.py) | 945 | 7 |
| Toto | [src/models/toto/model.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/toto/model.py) | 565 | 5 |
| Tide | [src/models/tide/model.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/tide/model.py) | 379 | 3 |

Top hotspots by long-method span (selected):

- TTM:
  - `_train_model` (~124 lines)
  - `_compute_trainer_metrics` (~118 lines)
  - `_prepare_training_data` (~116 lines)
- TimesFM:
  - `_prepare_training_data` (~225 lines)
  - `forward` (~156 lines)
  - `_train_model` (~106 lines)
- Moirai:
  - `_train_model` (~233 lines)
  - `_prepare_training_tensors` (~146 lines)
  - `evaluate_probabilistic` (~111 lines)
- Moment:
  - `_train_model` (~286 lines)
  - `_get_context_target_pairs` (~162 lines)
  - `_forecast_batch` (~109 lines)
- Chronos2:
  - `_materialize_intermediate_checkpoints` (~169 lines)
  - `_predict_batch` (~142 lines)
- Toto:
  - `_predict_batch` (~119 lines)
  - `_train_model` (~103 lines)
- Tide:
  - `_train_model` (~64 lines)
  - `_predict_batch` (~49 lines)

### Existing schema/factory/contract coverage baseline

Cross-family baseline gates already in place:

- [tests/models/test_model_family_contract_suite.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/tests/models/test_model_family_contract_suite.py)
- [tests/workflows/forecasting/test_model_config_schema_loader.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/tests/workflows/forecasting/test_model_config_schema_loader.py)
- [tests/workflows/forecasting/test_config_schema_artifact_generation.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/tests/workflows/forecasting/test_config_schema_artifact_generation.py)

Family-specific regression coverage currently present:

- TTM: [test_ttm_preprocessor_roundtrip.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/tests/models/test_ttm_preprocessor_roundtrip.py), [test_ttm_preprocessor_schema_contract.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/tests/models/test_ttm_preprocessor_schema_contract.py)
- TimesFM: [test_timesfm_config.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/tests/models/test_timesfm_config.py), [test_timesfm_loss_and_callback.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/tests/models/test_timesfm_loss_and_callback.py), [test_timesfm_patient_split.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/tests/models/test_timesfm_patient_split.py)
- Chronos2: [test_chronos2.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/tests/models/test_chronos2.py), [gpu_smoke_test_chronos2_inference.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/tests/models/gpu_smoke_test_chronos2_inference.py), [test_chronos2_known_covariates.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/tests/data/test_chronos2_known_covariates.py)

Coverage gaps to close in PR-C1..PR-C4:

- Moirai, Toto, and Tide currently rely mostly on shared contract + schema loader coverage with limited direct family-specific regression tests.
- Moment has sweep-config coverage but should gain explicit runtime regression checks alongside consolidation.

### Constructor-path call-site baseline

Current maintained call-site inventory confirms mixed constructor lanes:

- `create_model_and_config` call sites in:
  - [src/workflows/evaluation/nocturnal_hypo_eval.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/workflows/evaluation/nocturnal_hypo_eval.py)
  - [src/workflows/evaluation/sliding_window_eval.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/workflows/evaluation/sliding_window_eval.py)
  - [src/workflows/evaluation/validate_predict_batch.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/workflows/evaluation/validate_predict_batch.py)
  - [src/workflows/personalization/per_patient_finetune.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/workflows/personalization/per_patient_finetune.py)
- `ModelFactory` path is already primary in:
  - [src/workflows/forecasting/pipeline.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/workflows/forecasting/pipeline.py)
  - [src/workflows/forecasting/modeling.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/workflows/forecasting/modeling.py)

### Known edge cases to preserve through consolidation

- TTM checkpoint preprocessor schema guardrails must remain fail-fast and actionable.
- TimesFM single-patient split/window behavior must retain zero-window protections and training-window retention semantics.
- Chronos2 known-covariate and checkpoint-materialization behavior remains high-risk and needs parity tests during Tier B waves.
- Moirai probabilistic output/evaluation pathways need explicit parity checks in family-specific tests before broad helper extraction.

### PR-C1 handoff note

PR-C1 should target only TTM consolidation (`WS2`) plus missing TTM-focused regression coverage needed to prove behavior parity after helper extraction.

### 8B) Current checkpoint (2026-08-26)

- Housekeeping closeout from PR #463 and P1-39 wrap-up was completed on working branch
  [chore/pr-463-closeout-p1-38-kickoff](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast).
- All-model Aleppo smoke suite runner and comparator are in place:
  - [run_aleppo_model_regression_smoke_suite.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/scripts/workflows/forecasting/run_aleppo_model_regression_smoke_suite.py)
  - [compare_regression_smoke_suites.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/scripts/workflows/forecasting/compare_regression_smoke_suites.py)
- Fast/observable smoke profiles and ultra-smoke Aleppo holdout are in place:
  - [holdout_smoke_aleppo_ultra/](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/configs/data/holdout_smoke_aleppo_ultra)
  - per-family `97_regression_smoke_balanced.yaml` configs under
    [configs/models/](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/configs/models)
- Suite-manifest nested-path resolution is fixed so per-model status resolves
  timestamped run manifests correctly.
- TimeGrad smoke-lane compatibility is resolved with a v2-compatible runtime lane
  (`pydantic>=2`, modern `gluonts`, compatibility shims in
  [src/models/timegrad/](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/timegrad)).
- Latest pre-change baseline run is
  `pre_refactor_20260827v1` under
  [trained_models/artifacts/regression_smoke/all_models_aleppo/](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/trained_models/artifacts/regression_smoke/all_models_aleppo)
  with all maintained models succeeding.

### Immediate next actions

1. Start PR-C1 (TTM-only consolidation slice) with no behavior change outside TTM lane.
2. Run pre/post smoke comparator around PR-C1 using `pre_refactor_20260827v1` as baseline.
3. Record PR-C1 drift/artifact parity outcomes in this plan and `project_tracking.csv`.

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
6. Maintained evaluation entrypoints (including canonical wrappers that route to
   [`src/workflows/evaluation/`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/workflows/evaluation/))
   use schema-routed constructor behavior.
7. `create_model_and_config` has no remaining maintained entrypoint callers.
8. The final consolidation PR removes the `create_model_and_config` shim after
   test-backed parity validation.
9. Constructor-path ownership is explicit in docs: `ModelFactory` for runtime
   construction and `ModelRegistry` for registration/discovery support only.

---

## 11) Definition of readiness for downstream P1 work

`optuna-integration`, `mlflow-integration`, `stats-rigor-module`, and
`experiments-collapse` should only proceed once this task reaches done (or an
explicitly accepted partial with residuals tracked as separate blocking rows).
