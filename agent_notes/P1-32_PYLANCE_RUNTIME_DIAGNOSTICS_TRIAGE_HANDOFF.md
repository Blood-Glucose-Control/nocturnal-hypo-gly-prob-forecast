# P1-32 Runtime Diagnostics Triage Handoff

Date: 2026-08-20
Status: Ready to start (next PR recommendation)

## Recommended next task

**`pylance-runtime-diagnostics-triage`** from [project_tracking.csv](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/project_tracking.csv).

## Why this is next

- It is the remaining high-priority follow-up from the cleanup validation matrix.
- It is already scoped and pending.
- It directly improves runtime reliability and contributor signal quality.

## What is already known

From [P1-30_CLEANUP_VALIDATION_MATRIX_KICKOFF.md](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/agent_notes/P1-30_CLEANUP_VALIDATION_MATRIX_KICKOFF.md), targeted diagnostics showed:

- `reportMissingImports` in:
  - [darts_base.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/darts_base.py)
  - [evaluation.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/workflows/forecasting/evaluation.py)
  - [pipeline.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/workflows/forecasting/pipeline.py)
  - [sliding_window_eval.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/scripts/evaluation/sliding_window_eval.py)
- Additional optionality/operator diagnostics in:
  - [pipeline.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/workflows/forecasting/pipeline.py)

## Proposed PR scope

1. Re-run targeted Pylance diagnostics on runtime-core files.
2. Separate environment-resolution noise vs real code issues.
3. Fix true positives (especially optional access/operator paths in forecasting pipeline).
4. Re-run targeted checks:
   - diagnostics on touched files
   - focused runtime tests or script smoke checks for touched paths
   - `ruff check` on changed files
5. Update tracking/note with exact resolved vs deferred diagnostics.

## Out of scope

- Broad repo-wide typing cleanup.
- New architecture changes unrelated to diagnostic findings.

## Suggested acceptance criteria

- No remaining actionable Pylance diagnostics in the agreed runtime-core target set.
- All code changes validated with targeted tests/checks.
- Tracking and handoff notes updated with evidence.

## First command set for next session

1. Run targeted diagnostics on the four known files plus [pipeline.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/workflows/forecasting/pipeline.py).
2. Patch only confirmed true-positive issues.
3. Validate and open PR.
