# P1-30 Cleanup Validation Matrix Kickoff

Date: 2026-08-20
Status: Complete (with follow-up tasks opened)

## Why this note exists

No prior dedicated `agent_notes` file existed for `cleanup-validation-matrix` (P1-30). This is now the canonical kickoff/status note for that task.

## Work just completed before kickoff

1. Fast-forwarded local `main` to merged PR #447 commit (`cecf08b`).
2. Pruned stale remote refs and deleted the local merged feature branch.
3. Reconfirmed tracking priority and selected `cleanup-validation-matrix` as the next P1 execution task.

## Task definition (from `project_tracking.csv`)

`cleanup-validation-matrix` requires a focused post-cleanup matrix covering:

- active-model unit tests,
- key experiment/evaluation entrypoints (`per_patient_finetune`, `nocturnal_hypo_eval`, `sliding_window_eval`),
- lint/compile checks,
- Pylance diagnostic sweep on runtime-core files.

## Verified current state (done vs not done)

### Already done (prerequisites and partial evidence)

- Dependency tasks are complete:
  - `prune-empty-scaffold-packages`
  - `scripts-reorg-and-prune`
  - `dormant-model-family-disposition`
- Recent merged runtime validation exists for TSMixer integration:
  - Aleppo IOB/COB smoke run completed after training and after checkpoint load.
  - Targeted regression tests were added for Darts gap splitting, forecast validation, output-dir behavior, and TSMixer wiring.

### Matrix execution log (this kickoff)

1. **Active-model tests**
   Command: `pytest tests/models`
   Result: `136 passed, 14 skipped, 1 skipped file, 0 failed` (autogluon env).

2. **Canonical entrypoint sanity checks**
   Commands:
   - `python scripts/workflows/personalization/per_patient_finetune.py --help`
   - `python scripts/evaluation/nocturnal_hypo_eval.py --help`
   - `python scripts/evaluation/sliding_window_eval.py --help`
   Result: all three completed successfully.

3. **Compile + targeted lint checks**
   Commands:
   - `python -m compileall src scripts/evaluation scripts/workflows/personalization`
   - `ruff check src/models src/workflows src/evaluation scripts/evaluation scripts/workflows/personalization --select E9,F63,F7,F82`
   Result: compile pass successful; targeted lint pass successful (`All checks passed!`).

4. **Pylance diagnostics (runtime-core targeted sweep)**
   `textDocument/diagnostic` run on canonical runtime files.
   Result split:
   - No diagnostics: `src/models/factory.py`, `src/workflows/forecasting/modeling.py`, `src/workflows/evaluation/nocturnal_hypo_eval.py`, `scripts/workflows/personalization/per_patient_finetune.py`, `scripts/evaluation/nocturnal_hypo_eval.py`.
   - Diagnostics present:
     - `reportMissingImports` on several `src.*` imports in:
       - `src/models/darts_base.py`
       - `src/workflows/forecasting/evaluation.py`
       - `src/workflows/forecasting/pipeline.py`
       - `scripts/evaluation/sliding_window_eval.py`
     - Additional non-import diagnostics (`reportOptionalSubscript`, `reportOperatorIssue`, `reportOptionalMemberAccess`) in `src/workflows/forecasting/pipeline.py`.

5. **Deep follow-up: lightweight functional runs (post-fast scope)**
   A temporary mini holdout config was created at `/tmp/p1_30_holdout_patient_single/brown_2019.yaml` using `holdout_type=patient_based` with one holdout patient (`bro_92`) to keep runtime bounded while executing real script paths.

   - `scripts/evaluation/nocturnal_hypo_eval.py`
     - Zero-shot `naive_baseline` failed as expected (`requires training before prediction`).
     - Re-run with trained checkpoint succeeded:
      `--checkpoint trained_models/artifacts/naive_baseline/2026-04-28_01:37_RID20260428_013743_836040_holdout_workflow`
      Result: completed, RMSE `4.5259`, `130` midnight episodes, run manifest written.

   - `scripts/evaluation/sliding_window_eval.py`
     - Run with same trained `naive_baseline` checkpoint succeeded.
      Result: completed, RMSE `4.185`, `435` episodes, results/plots written.

   - `scripts/workflows/personalization/per_patient_finetune.py`
     - `ttm` path failed in stage-2 fit with:
      `AttributeError: 'TimeSeriesPreprocessor' object has no attribute 'other_columns_to_scale'`.
     - `timesfm` path failed in stage-2 fit with:
      `ValueError: num_samples should be a positive integer value, but got num_samples=0`.
     - `chronos2` path succeeded (1 patient, `fine_tune_steps=1`, `forecast_length=24`) after preparing a temporary checkpoint overlay that adds top-level config metadata for loader compatibility:
      - Stage-1 RMSE: `1.1133` (4 midnight episodes)
      - Stage-2 RMSE: `1.2495` (4 midnight episodes)
      - Script completed with `Succeeded: 1 / 1`.

### Remaining work before P1-30 closeout

- P1-30 itself is now closed by operator decision after documenting matrix outcomes.
- Follow-up work has been split into new tracked tasks:
  - `per-patient-stage2-regression-triage`
  - `pylance-runtime-diagnostics-triage`

## Proposed execution plan

1. **Model tests pass**
   Run focused `tests/models/` matrix (and add minimal targeted evaluation tests if failures point to workflow boundaries).
2. **Entrypoint sanity pass**
   Execute canonical entrypoint checks (at minimum `--help`; then lightweight smoke where available).
3. **Lint/compile pass**
   Run targeted lint/compile checks over runtime-core modules touched by cleanup waves.
4. **Pylance diagnostics pass**
   Run and triage targeted runtime-core diagnostics, separating interpreter/workspace config issues from true code regressions.
5. **Closeout artifacts**
   Update this note with results + outcomes, then mark `project_tracking.csv` P1-30 complete if all matrix gates pass.

## Operator note

`project_tracking.csv` status was moved `pending -> in_progress -> done` for `cleanup-validation-matrix` in this execution window.
Execution scope selected with operator: **Fast matrix** (tests/models + entrypoint help checks + targeted lint/compile + targeted Pylance diagnostics).
Deep follow-up execution has now been run in the same task window (lightweight functional script runs on a bounded mini-holdout).
Closeout policy selected by operator: close P1-30 now and track the per-patient model regressions + Pylance triage as separate follow-up tasks.

## Follow-up completion: per-patient-stage2-regression-triage (2026-08-20)

### Fixes implemented

1. **TTM preprocessor schema contract**
   - Added `_validate_preprocessor_schema` in `src/models/ttm/model.py`.
   - Runtime now fails fast with an actionable error if checkpoint preprocessor schema is unsupported (missing `other_columns_to_scale`).
   - Policy: legacy checkpoint shimming is intentionally not maintained; retrain Stage-1 with current runtime when needed.

2. **TimesFM Stage-2 single-patient split robustness**
   - Added `_split_train_val_patients` in `src/models/timesfm/model.py`.
   - Guarantees at least one train patient (single-patient runs now stay in train).
   - Added explicit guardrail error when train windows are zero after split/windowing.

3. **TimesFM Stage-2 training stability fixes surfaced during triage**
   - Ensure Stage-2 output directory exists before callback init (`epoch_metrics.csv` path).
   - Hardened mid-training callback to create output dirs.
   - Added explicit input dtype wiring for trainer forward pass to avoid BF16/FP32 mismatches.
   - Updated collate path to return dense `past_values` tensor so multi-GPU `DataParallel` scatters `past_values` and `freq` consistently.

### Regression tests added/updated

- Added `tests/models/test_ttm_preprocessor_schema_contract.py`.
- Added `tests/models/test_timesfm_patient_split.py`.
- Updated `tests/models/test_timesfm_config.py` for `val_patient_ratio` validation.
- Updated `tests/models/test_timesfm_loss_and_callback.py`:
  - callback creates missing output dir,
  - forward casts inputs to model dtype safely.

### Validation run results

- Targeted tests:
  - TimesFM suite: `20 passed`
  - TTM schema-contract test: `1 passed`
- Targeted lint:
  - `ruff check` on changed model/test files: passed.
- End-to-end one-patient Stage-2 smoke (brown_2019, holdout `bro_92`, IOB/COB):
  - **TimesFM**: succeeded (`summary.csv` status `OK`; Stage-1 RMSE 2.5811 -> Stage-2 RMSE 2.0947).
  - **TTM**: succeeded (`summary.csv` status `OK`; no preprocessor attribute error).
