# P1-37 Canonical Handoff and Status

Date: 2026-08-18
Status: Complete (cleanup execution closed; follow-ups tracked separately)

## Scope this note covers

This note is the single canonical handoff/status artifact for P1-37 scripts reorg/prune execution after merge completion.

## Outcome summary

The multi-wave scripts cleanup is complete and merged:

1. Wave 1: canonical script-surface rehome (`scripts/workflows/`, `scripts/orchestration/sweeps/`).
2. Wave 2: heavy runtime migration into `src/workflows/*` with thin maintained script entrypoints.
3. Wave 2b: profile-driven sweep wrapper dedup across maintained model families and context-ablation surfaces.
4. Wave 3: legacy script quarantine into `scripts/scratch/` (recoverable, non-canonical).
5. Wave 4: retained visualization/analysis consistency pass (model-agnostic plotting surfaces, shared `src/visualization/nocturnal.py` helpers, docstring interpretation upgrades, and docs example gallery alignment).

## Branch/PR closeout

- Cleanup branch was rebased onto updated `main` and merged.
- Local `main` was fast-forwarded.
- Obsolete local work branches and temporary rebase backup branches were removed.
- New post-merge maintenance branch: `p1-37-tracking-reconcile-notes-consolidation`.

## Tracking reconciliation actions (this pass)

- `project_tracking.csv` P1-37 wave tasks were reconciled and closed where complete.
- A new low-priority follow-up task was added:
  - `visualization-quantile-default-centralization` (P3, pending).

## Low-priority quantile-default centralization proposal

Proposed implementation in `src/visualization/nocturnal.py`:

1. Introduce a single exported defaults object (e.g., dataclass or named constants bundle) for:
   - boxplot quantiles,
   - IQR quantiles,
   - probabilistic interval defaults used by plotting scripts.
2. Keep existing validator functions as the authoritative shape/range checks.
3. Rewire script CLI default values to read from that shared defaults surface.
4. Preserve existing CLI override flags so current workflows remain customizable and backward compatible.
5. Add focused tests that assert:
   - helper functions and script parsers share the same defaults,
   - override behavior is unchanged,
   - invalid quantile settings still fail fast with clear errors.

## Archived superseded notes

The following overlapping P1-37 planning/handoff variants are archived under:

`agent_notes/archive_2026-08-18/p1-37/`

- `P1-37_HANDOFF_2026-08-14.md`
- `P1-37_HANDOFF_2026-08-16_EOD.md`
- `P1-37_PHASE_F_RATIFICATION_CHECKLIST_PR_PLAN.md`
- `P1-37_SCRIPTS_REORG_AND_PRUNE_EXECUTION_PLAN.md`
- `P1-37_SCRIPTS_UPDATED_CLEANUP_PLAN.md`
