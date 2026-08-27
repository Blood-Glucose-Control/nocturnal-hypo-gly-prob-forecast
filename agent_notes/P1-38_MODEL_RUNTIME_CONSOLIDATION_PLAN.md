# P1-38 Model Runtime Consolidation Plan

**Date:** 2026-08-23
**Update Date:** 2026-08-27
**Status:** In progress (all-model pre-change baseline is green; PR-C1 is active with WS1-TTM-01 and WS1-TTM-02 completed)
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

### WS1A — Objective matrix (detailed, itemized task list) — 2026-08-27

| Family | Clean-up targets (itemized) | Common data input contracts | Shared method contracts | Consolidation tasks (itemized) | In-scope files/folders | Out-of-scope (family-local) | Acceptance checks | Targeted validation commands | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TTM | 1) Decompose `_predict` / `_predict_batch` / checkpoint preprocessor paths.<br>2) Decompose `_prepare_training_data`, `_train_model`, `_compute_trainer_metrics`, `_create_training_arguments`.<br>3) Audit `src/models/ttm/_deprecated/` with no-caller proof prior to retention/removal decisions.<br>4) Remove stale TODO-heavy comments from retained legacy surfaces. | 1) Training accepts panel DataFrame or patient dict and normalizes to DataFrame.<br>2) `predict` remains single-episode contract.<br>3) `predict_batch` remains `episode_col` -> `Dict[str, np.ndarray]` contract. | Preserve canonical base lifecycle behavior for `_initialize_model`, `_prepare_training_data`, `_train_model`, `_predict`, `_predict_batch`, `_save_checkpoint`, `_load_checkpoint`; preserve preprocessor schema fail-fast behavior. | 1) Extract quantile warning helper.<br>2) Extract zero-shot pipeline builder helper.<br>3) Extract preprocessor save/load helper pair.<br>4) Follow with training-path helper extraction.<br>5) Add/keep explicit regression checks for checkpoint + scaling + batch semantics. | [src/models/ttm/model.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/ttm/model.py)<br>[src/models/ttm/config.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/ttm/config.py)<br>[src/models/ttm/_deprecated/](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/ttm/_deprecated) | TTM architecture redesign or TSFM external package behavior changes. | 1) TTM fitted/zero-shot inference parity preserved.<br>2) Preprocessor checkpoint behavior preserved.<br>3) Contract/schema gates remain green. | `pytest -q tests/models/test_ttm_preprocessor_schema_contract.py tests/models/test_ttm_preprocessor_roundtrip.py`<br>`pytest -q tests/models/test_model_family_contract_suite.py`<br>`pytest -q tests/workflows/forecasting/test_model_config_schema_loader.py -k "ttm"` | Completed (WS1-TTM-01/02/03 complete on PR-C1 branch). |
| TimesFM | 1) Split `_prepare_training_data` and `_train_model` hotspots.<br>2) Reduce duplicated alias/window setup logic.<br>3) Preserve horizon/forecast alias synchronization. | Flat panel -> patient windows with deterministic split/window semantics and stable output shape contracts. | Preserve predict/load/save behavior and `forecast_length`/`horizon_length` semantics. | 1) Extract patient split/window helpers.<br>2) Extract checkpoint metadata/path normalization helper.<br>3) Keep schema-route adapter behavior unchanged. | [src/models/timesfm/model.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/timesfm/model.py)<br>[src/models/timesfm/config.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/timesfm/config.py) | TimesFM algorithm/loss redesign outside behavior-preserving refactor scope. | Existing TimesFM tests + schema/contract gates stay green; no runtime payload drift. | `pytest -q tests/models/test_timesfm_config.py tests/models/test_timesfm_loss_and_callback.py tests/models/test_timesfm_patient_split.py` | Completed (WS1-TFM-01 fully validated in TimesFM dependency lane). |
| Moirai | 1) Replace `_prepare_training_data` placeholder compatibility path with explicit contract-safe behavior.<br>2) Split `_train_model` and training tensor prep logic.<br>3) Reduce duplicate dataframe/episode conversion branches. | DataFrame/list/dict inputs normalize deterministically; `past_covariate_dim`/`covariate_cols` parity enforced. | Preserve probabilistic outputs and checkpoint load/save behavior (`.ckpt` + base format). | 1) Extract input normalization helpers.<br>2) Extract sample->quantile helper.<br>3) Add direct runtime parity tests for probabilistic and covariate-dimension behavior. | [src/models/moirai/model.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/moirai/model.py)<br>[src/models/moirai/config.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/moirai/config.py) | Moirai objective/architecture changes. | Schema route behavior unchanged; probabilistic parity proven by tests. | `pytest -q tests/workflows/forecasting/test_model_config_schema_loader.py -k "moirai"` (+ family runtime tests in C2) | Completed (WS1-MOI-01 helper extraction + runtime helper tests validated in Moirai dependency lane). |
| Moment | 1) Split `_train_model`, `_get_context_target_pairs`, `_forecast_batch`.<br>2) Remove stale behavior comments/docstrings adjacent to touched logic.<br>3) Consolidate target/covariate column selection branches. | Panel context/target extraction and covariate stacking remain deterministic and default-compatible. | Preserve predict/predict_batch API behavior and wrapper-normalization semantics. | 1) Extract context-target prep helpers.<br>2) Extract reusable column selectors.<br>3) Add runtime parity tests beyond sweep-config coverage. | [src/models/moment/model.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/moment/model.py)<br>[src/models/moment/config.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/moment/config.py) | Unapproved normalization behavior changes. | No API drift; tests + shared gates green. | `pytest -q tests/models/test_moment_sweep_configs.py` (+ runtime tests in C3) | Completed (WS1-MOM-01 helper extraction + validation gates complete; targeted runtime-test expansion remains queued for C4 parity hardening). |
| Chronos2 | 1) Split `_materialize_intermediate_checkpoints` and `_predict_batch`.<br>2) Re-home/remove stale helper surfaces in `chronos2/utils.py` and `chronos2/visualization.py` only with no-caller proof.<br>3) Keep known-covariate boundaries explicit and tested. | Panel->AutoGluon conversion with strict `target_col/patient_col/time_col`; known future covariates separate from past-only covariates. | Preserve fine-tune/zero-shot behavior and checkpoint materialization behavior. | 1) Extract known-covariate context build helper.<br>2) Extract checkpoint materialization primitives.<br>3) Add direct parity tests for known covariates and checkpoint materialization. | [src/models/chronos2/model.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/chronos2/model.py)<br>[src/models/chronos2/config.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/chronos2/config.py)<br>[src/models/chronos2/utils.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/chronos2/utils.py)<br>[src/models/chronos2/visualization.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/chronos2/visualization.py) | Dropping required Chronos2-specific behavior deviations. | Known-covariate + checkpoint parity proven; no shared gate regressions. | `pytest -q tests/models/test_chronos2.py tests/data/test_chronos2_known_covariates.py` | Pending |
| Toto | 1) Split `_predict_batch` and `_train_model`.<br>2) Deduplicate timestamp/variate preparation helpers.<br>3) Keep alias semantics (`lr`/`learning_rate`, `max_steps`/`num_epochs`) consistent. | DataFrame->variate tensor conversion deterministic; batch mapping and sample semantics preserved. | Preserve base lifecycle and probabilistic behavior compatibility. | 1) Extract timestamp conversion helper.<br>2) Extract shared episode-batching helper.<br>3) Add dedicated Toto runtime regression tests. | [src/models/toto/model.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/toto/model.py)<br>[src/models/toto/config.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/toto/config.py) | Toto architecture/training objective redesign. | Schema/factory and runtime behavior parity maintained. | `pytest -q tests/workflows/forecasting/test_model_config_schema_loader.py -k "toto"` (+ runtime tests in C4) | Pending |
| TiDE | 1) Reduce duplicated AG data prep/checkpoint/inference scaffolding.<br>2) Preserve strict constraints (`from_scratch`, `scaling='mean'`, encoder/decoder parity). | Flat panel segmentation + AG inference context remain behavior-equivalent. | Preserve strict config validation and checkpoint behavior. | 1) Consolidate shared AG helper usage where equivalent.<br>2) Add TiDE runtime parity tests beyond schema/factory checks.<br>3) Validate no drift in hard-constraint enforcement. | [src/models/tide/model.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/tide/model.py)<br>[src/models/tide/config.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/tide/config.py) | Relaxing TiDE hard constraints. | Constraint checks preserved; runtime behavior unchanged. | `pytest -q tests/workflows/forecasting/test_model_config_schema_loader.py -k "tide"` (+ runtime tests in C4) | Pending |

### WS1B — Itemized execution task list (ordered)

| Task ID | Objective | Primary files | Exit condition |
| --- | --- | --- | --- |
| WS1-TTM-01 | Extract TTM inference/checkpoint shared helpers and remove duplicated branches without behavior drift. | [src/models/ttm/model.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/ttm/model.py) | TTM targeted tests + contract/schema gates pass. |
| WS1-TTM-02 | Decompose TTM training path helpers (`_prepare_training_data`, `_train_model`, metrics, training args). | [src/models/ttm/model.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/ttm/model.py) | Same gates pass with no behavior drift evidence. |
| WS1-TTM-03 | Produce no-caller and parity evidence for [src/models/ttm/_deprecated/](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/ttm/_deprecated) and stage keep/remove decision. | [src/models/ttm/_deprecated/](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/ttm/_deprecated) | Explicit keep/remove decision documented with caller proof. |
| WS1-TFM-01 | Extract TimesFM windowing + checkpoint normalization helpers and reduce long-method complexity. | [src/models/timesfm/model.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/timesfm/model.py) | TimesFM targeted tests + shared gates pass. |
| WS1-MOI-01 | Normalize Moirai multi-input adaptation and probabilistic extraction helpers. | [src/models/moirai/model.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/moirai/model.py) | Moirai schema tests + runtime parity tests pass. |
| WS1-MOM-01 | Refactor Moment context/target/training helper boundaries. | [src/models/moment/model.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/moment/model.py) | Moment tests + shared gates pass. |
| WS1-CHR-01 | Refactor Chronos2 known-covariate + checkpoint-materialization helper surfaces. | [src/models/chronos2/model.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/chronos2/model.py) | Chronos2 targeted tests and smoke parity pass. |
| WS1-CHR-02 | Re-home/retire stale Chronos2 utility/visualization surfaces only with no-caller proof. | [src/models/chronos2/utils.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/chronos2/utils.py), [src/models/chronos2/visualization.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/chronos2/visualization.py) | Caller inventory + parity evidence captured. |
| WS1-TOT-01 | Decompose Toto batch/training helper logic and add direct regression coverage. | [src/models/toto/model.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/toto/model.py) | Toto runtime tests + shared gates pass. |
| WS1-TID-01 | Consolidate TiDE with shared AG helper patterns while preserving hard constraints. | [src/models/tide/model.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/tide/model.py) | TiDE schema/runtime tests pass; constraints unchanged. |

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

1. Run pre/post smoke comparator around PR-C3 using `pre_refactor_20260827v1` as baseline.
2. Record PR-C3 drift/artifact parity outcomes in this plan and `project_tracking.csv`.
3. Begin PR-C4 by starting WS1-CHR-01 / WS1-TOT-01 / WS1-TID-01.

### PR-C1 status checkpoint (2026-08-27)

- Branch created: `feat/p1-38-ws1-helper-extraction-c1`.
- WS1-TTM-01 completed in [src/models/ttm/model.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/ttm/model.py):
  - extracted quantile warning helper,
  - extracted zero-shot pipeline builder helper,
  - extracted preprocessor checkpoint save/load helper pair.
- WS1-TTM-02 completed in [src/models/ttm/model.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/ttm/model.py):
  - decomposed `_prepare_training_data` into normalized-input, preprocessor, dataset, dataloader, and dataset-logging helpers,
  - decomposed `_train_model` into training-environment, trainer-construction, training-history, and test-evaluation helpers,
  - preserved existing training/inference contracts and preprocessor schema guard behavior.
- WS1-TTM-03 completed for [src/models/ttm/_deprecated/](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/ttm/_deprecated):
  - no maintained runtime callers found (`rg` path/import scans show no references outside this note),
  - [src/models/ttm/__init__.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/ttm/__init__.py) exports only `TTMConfig` and `TTMForecaster`,
  - `_deprecated` helper symbols are self-contained to the `_deprecated` directory,
  - decision staged: retain `_deprecated` as non-maintained archival reference during the consolidation wave (no active runtime dependency), and avoid behavior-risking removals in PR-C1.
- Targeted validations completed:
  - `pytest -q tests/models/test_ttm_preprocessor_schema_contract.py tests/models/test_ttm_preprocessor_roundtrip.py tests/models/test_model_family_contract_suite.py`
  - `pytest -q tests/workflows/forecasting/test_model_config_schema_loader.py -k "ttm"`
  - `SKIP=pyright pre-commit run --files src/models/ttm/model.py`
  - `ruff check src/models/ttm/model.py`
  - Pylance diagnostics clean for touched file.

### PR-C2 status checkpoint (2026-08-27)

- WS1-TFM-01 pass 1 landed in [src/models/timesfm/model.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/timesfm/model.py):
  - decomposed `_prepare_training_data` into patient extraction, datetime-index normalization, gap segmentation, segment regrouping, split-array generation, dataset building, and temporal-eval dataset helpers,
  - decomposed `_train_model` into input-dtype, trainer-wrapper, training-args, callback, and trainer-construction helpers.
- Validation status for this pass:
  - `pytest -q tests/models/test_timesfm_config.py tests/models/test_timesfm_loss_and_callback.py tests/models/test_timesfm_patient_split.py` (all skipped in current environment),
  - `pytest -q tests/models/test_model_family_contract_suite.py` (pass),
  - `pytest -q tests/workflows/forecasting/test_model_config_schema_loader.py -k "timesfm"` (pass),
  - `SKIP=pyright pre-commit run --files src/models/timesfm/model.py` (pass),
  - Pylance diagnostics clean for touched file.
- WS1-TFM-01 pass 2 landed in [src/models/timesfm/model.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/timesfm/model.py):
  - extracted checkpoint path/config serialization helpers and unified save/load checkpoint path handling.
- WS1-MOI-01 pass 1 landed in [src/models/moirai/model.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/moirai/model.py):
  - replaced `_prepare_training_data` placeholder stub with real patched-loader construction from normalized training tensors,
  - rewired `_train_model` to consume `_prepare_training_data` (single contract path),
  - extracted predictor lifecycle / input-normalization / forecast-extraction helpers,
  - consolidated dict/list episode normalization and covariate-column validation inside training tensor prep.
- WS1-MOI-01 pass 2 landed:
  - added helper-parity tests in [test_moirai_runtime_contract_helpers.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/tests/models/test_moirai_runtime_contract_helpers.py),
  - applied explicit optional-dependency Pyright import guards for `gluonts`/`uni2ts` imports in [model.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/moirai/model.py).
- Validation status for C2 passes:
  - `pytest -q tests/models/test_moirai_runtime_contract_helpers.py tests/models/test_model_family_contract_suite.py tests/workflows/forecasting/test_model_config_schema_loader.py -k "moirai"` (pass; helper test file is skipped when optional deps are unavailable),
  - `pytest -q tests/models/test_model_family_contract_suite.py` (pass),
  - `pytest -q tests/workflows/forecasting/test_model_config_schema_loader.py -k "moirai"` (pass),
  - `SKIP=pyright pre-commit run --files src/models/timesfm/model.py src/models/moirai/model.py tests/models/test_moirai_runtime_contract_helpers.py` (pass),
  - Pylance diagnostics clean for touched files.
- Dependency-lane closure evidence:
  - `.venvs/timesfm/bin/python -m pytest -q tests/models/test_timesfm_config.py tests/models/test_timesfm_loss_and_callback.py tests/models/test_timesfm_patient_split.py` (20 passed),
  - `.venvs/moirai/bin/python -m pytest -q tests/models/test_moirai_runtime_contract_helpers.py` (3 passed),
  - `.venvs/moirai/bin/python -m pytest -q tests/workflows/forecasting/test_model_config_schema_loader.py -k "moirai"` (6 passed).
- C2 status: ready to close and move to C3 (Moment), with C2 code paths validated in family dependency lanes.

### PR-C3 status checkpoint (2026-08-27)

- WS1-MOM-01 landed in [src/models/moment/model.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/moment/model.py):
  - extracted helper boundaries for pair split normalization, dataset/loader construction, train-loop metrics accumulation, validation-loss evaluation, and state-dict save/load orchestration,
  - preserved wrapper-normalization behavior and single/batch prediction contracts,
  - resolved touched-file typing issues surfaced by the final diagnostics gate (explicit array coercions, model optionality narrowing, and initialized loss variables).
- Validation status for C3:
  - `pytest -q tests/models/test_moment_sweep_configs.py` (pass),
  - `pytest -q tests/models/test_model_family_contract_suite.py` (pass),
  - `pytest -q tests/workflows/forecasting/test_model_config_schema_loader.py -k "moment"` (pass),
  - `SKIP=pyright pre-commit run --files src/models/moment/model.py` (pass),
  - Pylance diagnostics clean for touched file (warnings only; no error-severity diagnostics).
- C3 status: helper extraction and validation gates complete; proceed to C4 families.

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
