# P1-32 Workspace Remount Handoff

Date: 2026-08-20
Branch: `p1-stage2-triage-clean-pr`
Goal: continue P1-32 in a new chat after reopening workspace at canonical path.

## Why this handoff exists

Current chat is attached to a session whose Pylance workspace roots resolve under the symlink path:

- `file:///data/home/cjrisi/nocturnal/...`

You want canonical roots under:

- `file:///data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/...`

Reopening [nocturnal-forecast.code-workspace](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/nocturnal-forecast.code-workspace) from the canonical path is expected to fix this.

## Current in-progress diff (not committed yet)

- [src/models/darts_base.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/darts_base.py)
- [src/models/base/base_model.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/base/base_model.py)
- [src/workflows/forecasting/evaluation.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/workflows/forecasting/evaluation.py)
- [agent_notes/P1-32_PYLANCE_RUNTIME_DIAGNOSTICS_TRIAGE_HANDOFF.md](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/agent_notes/P1-32_PYLANCE_RUNTIME_DIAGNOSTICS_TRIAGE_HANDOFF.md)
- [project_tracking.csv](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/project_tracking.csv)

Untracked local dirs (intentionally untouched):

- [/.stash-backups/](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/.stash-backups/)
- [/logs/](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/logs/)

## Validation already run in this chat

- `ruff check src/models/darts_base.py src/models/base/base_model.py src/workflows/forecasting/evaluation.py` ✅
- `pytest -q tests/models/test_darts_base_gap_split.py tests/models/test_tsmixer_darts_wiring.py tests/evaluation/test_forecast_prediction_validation.py` ✅ (`6 passed, 1 skipped`)
- Pylance diagnostics (workspace-path URIs using `/data/home/cjrisi/nocturnal/...`):
  - [src/models/darts_base.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/darts_base.py): clean
  - [src/workflows/forecasting/evaluation.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/workflows/forecasting/evaluation.py): clean
  - [src/workflows/forecasting/pipeline.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/workflows/forecasting/pipeline.py): clean
  - [scripts/evaluation/sliding_window_eval.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/scripts/evaluation/sliding_window_eval.py): clean

## First steps in the new chat

1. Verify workspace roots with Pylance and confirm they now use `/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/...`.
2. Re-run Pylance diagnostics on:
   - [src/models/darts_base.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/darts_base.py)
   - [src/workflows/forecasting/evaluation.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/workflows/forecasting/evaluation.py)
   - [src/workflows/forecasting/pipeline.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/workflows/forecasting/pipeline.py)
   - [scripts/evaluation/sliding_window_eval.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/scripts/evaluation/sliding_window_eval.py)
3. If clean, commit current diff and open/update PR for P1-32.
4. Keep deferred repo-wide diagnostics tracking entry (`pylance-workspace-diagnostics-sweep`) in [project_tracking.csv](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/project_tracking.csv).

## Paste this as your first message in the new chat

```
Continue from P1-32 workspace remount handoff in agent_notes/P1-32_WORKSPACE_PATH_REMOUNT_HANDOFF.md.
We are on branch p1-stage2-triage-clean-pr with uncommitted changes in darts_base/base_model/evaluation plus tracking notes.
First, verify Pylance workspace roots now use /data/home/cjrisi/nocturnal-hypo-gly-prob-forecast (not /data/home/cjrisi/nocturnal symlink path), then rerun diagnostics on the 4 runtime-core files and proceed to commit/PR if clean.
```
