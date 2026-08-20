# P1-32 Runtime Diagnostics Triage Handoff

Date: 2026-08-20
Status: Runtime-core scope completed; repo-wide diagnostics deferred and tracked

## Outcome summary

Completed **`pylance-runtime-diagnostics-triage`** from [project_tracking.csv](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/project_tracking.csv).

## What was resolved

- Fixed runtime-core `reportMissingImports` by converting absolute `src.*` imports to package-relative imports in:
  - [darts_base.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/darts_base.py)
  - [evaluation.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/workflows/forecasting/evaluation.py)
- Resolved follow-on `reportAttributeAccessIssue` diagnostics in [darts_base.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/darts_base.py) by introducing a typed runtime config protocol (`DartsRuntimeConfig`) and narrowing config access via `_cfg`.
- Re-ran targeted diagnostics on the agreed runtime-core set and verified zero remaining diagnostics:
  - [darts_base.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/darts_base.py)
  - [evaluation.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/workflows/forecasting/evaluation.py)
  - [pipeline.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/workflows/forecasting/pipeline.py)
  - [sliding_window_eval.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/scripts/evaluation/sliding_window_eval.py)
- Fixed follow-on `reportCallIssue` and override contract diagnostics in [darts_base.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/darts_base.py) by:
  - adding explicit Darts model/forecast protocols for `fit/predict/save` callsites
  - aligning shared `_prepare_training_data` typing in [base_model.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/base/base_model.py) with custom-backend behavior

## Validation evidence

- Targeted tests:
  - `pytest -q tests/models/test_darts_base_gap_split.py tests/models/test_tsmixer_darts_wiring.py tests/evaluation/test_forecast_prediction_validation.py`
  - Result: `6 passed, 1 skipped`
- Targeted lint:
  - `ruff check src/models/darts_base.py src/models/base/base_model.py src/workflows/forecasting/evaluation.py`
  - Result: `All checks passed`

## Scope caveat / root-cause correction

An earlier verification pass used real-path URIs (`.../nocturnal-hypo-gly-prob-forecast/...`) instead of canonical workspace-path URIs (`.../nocturnal/...`), while the editor workspace root is mounted at `/data/home/cjrisi/nocturnal`. Revalidation was rerun on workspace-path URIs and used as final evidence.

## Deferred items (explicit future tasks)

- **Repo-wide diagnostics sweep** is required and is now tracked as:
  - `pylance-workspace-diagnostics-sweep` in [project_tracking.csv](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/project_tracking.csv)
- Seed evidence for deferred scope already confirmed outside runtime-core:
  - [cache_manager.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/data/cache_manager.py) has `reportMissingImports` (`src.*` import style) plus deprecation-use diagnostics.
