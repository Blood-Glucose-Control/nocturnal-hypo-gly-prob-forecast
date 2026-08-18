# P1-37 Script Taxonomy Reset + Cleanup Plan (Draft)

## Problem and approach

The current `scripts/` surface still mixes runtime workflow entrypoints, evaluation CLIs, sweep orchestration, one-off experiment scripts, and legacy cluster wrappers. This blurs boundaries for new researchers and keeps too much model-specific logic in shell scripts.

Approach:

1. Ratify a strict taxonomy contract first (especially **evaluation vs experiment**).
2. Apply a complete per-script decision matrix (all 113 script/doc entrypoints audited).
3. Execute cleanup in waves so canonical runtime paths remain stable while legacy surfaces are moved to `scripts/scratch/` quarantine (no permanent deletion in this phase).
4. Keep Python-first logic in `src/` and keep `scripts/` as thin, discoverable entrypoints.

---

## Target taxonomy contract

### Directory role definitions

- `scripts/examples/`: onboarding/tutorial entrypoints only.
- `scripts/data_processing/`: generic holdout/data-config utilities only.
- `scripts/workflows/` (new): single-run workflow launchers (thin wrappers over `src/workflows/*`).
- `scripts/training/`: training launch surfaces (generic sweep launchers + SLURM wrappers).
- `scripts/evaluation/`: evaluation launch surfaces (nocturnal/sliding-window/etc).
- `scripts/orchestration/`: multi-run chains and sweep coordinators.
- `scripts/experiments/`: optional research protocol wrappers only (not default runtime surface).
- `scripts/analysis/`: post-run tables/summary scripts.
- `scripts/visualization/`: plotting/report scripts.
- `scripts/scratch/`: explicitly non-canonical legacy/exploratory scripts.

### Evaluation vs experiment

- **Evaluation**: a metric protocol that scores model outputs on a dataset/task split (e.g., nocturnal anchoring, sliding-window RMSE, probabilistic diagnostics).
- **Experiment**: a hypothesis-driven protocol that may include training + one or more evaluations + analysis choices.

Rule: evaluation implementations are first-class and reusable; experiment scripts should compose them, not duplicate them.

---

## Inventory results and per-script decisions

I generated a full action matrix for every script/doc entrypoint:

- [`script_inventory_action_matrix_2026-08-17.tsv`](/data/home/cjrisi/.copilot/session-state/f7693354-6f4e-43a2-a6b2-a901cfef5843/files/script_inventory_action_matrix_2026-08-17.tsv)

Action totals (113 files):

- `KEEP_CANONICAL`: 15
- `KEEP_AND_CLEAN`: 31
- `CLEAN_AND_DEDUP`: 24
- `MIGRATE_TO_CANONICAL_PATH`: 6
- `MIGRATE_TO_SRC_PLUS_THIN_SCRIPT`: 4
- `MOVE_TO_SCRATCH`: 33

This matrix is the script-by-script source of truth for keep/clean/migrate decisions, with legacy quarantine preferred over deletion.

---

## Planned implementation waves

### Wave 0 — lock canonical entrypoints

- Preserve currently stable generic sweep launchers and onboarding examples.
- Keep existing canonical paths working while migration PRs land.

### Wave 1 — namespace correction + docs + flow diagram

- Introduce `scripts/workflows/` for single-run forecasting workflow wrappers currently in `scripts/experiments/`.
- Move generic sweep dispatch CLIs to orchestration namespace.
- Update docs and add a mermaid flowchart showing data -> workflow -> eval -> analysis/orchestration flow.

### Wave 2 — Python-first migration of heavy runtime scripts

- Move heavy evaluation/workflow logic from `scripts/experiments/*.py` into `src/workflows/*` modules.
- Keep only thin script wrappers.
- Deduplicate heavy model sweep shells to profile-driven wrappers and generic launchers.
- Merge ctx-ablation eval variant behavior into one canonical nocturnal eval CLI.

### Wave 3 — legacy quarantine

- Move explicitly non-canonical scripts to `scripts/scratch/` (model-specific legacy SLURM wrappers, one-off experiment scripts, hardcoded data loader shells, old ad-hoc utilities).
- Remove stale references from docs and launchers.

### Wave 4 — analysis/visualization consistency pass

- Keep maintained analysis/plot scripts but standardize naming, IO contracts, and shared helper usage.
- Migrate shared plotting/analysis helper logic into reusable `src/` modules where duplicated.

---

## Project tracking tasks to add/update

Proposed new tracking tasks (to be added in `project_tracking.csv` when implementation begins):

1. `scripts-taxonomy-contract-ratification` — finalize role definitions and naming contract.
2. `scripts-surface-rehome-wave1` — move workflow/sweep entrypoints to canonical namespaces.
3. `scripts-runtime-migration-wave2` — migrate heavy script logic into `src/workflows` + thin wrappers.
4. `scripts-legacy-quarantine-wave3` — move designated legacy surfaces into `scripts/scratch`.
5. `scripts-sweep-dedup-wave2b` — convert remaining heavy model/context-ablation sweeps to profile-driven wrappers.
6. `scripts-analysis-visualization-consistency-wave4` — standardize retained post-run scripts.
7. `scripts-taxonomy-docs-and-mermaid` — update docs and publish architecture flowchart.

---

## Validation gates per wave

- `ruff check` on touched Python files.
- `bash -n` on touched shell scripts.
- Targeted dry-runs for changed launchers/orchestrators.
- No stale path references in docs for moved/deleted scripts.

---

## Decision captured

Wave 3 will use an **aggressive quarantine policy**:

- Move every script marked `MOVE_TO_SCRATCH` in the action matrix now (33 files).
- Keep canonical surface intentionally small and clean for new researchers.
- Preserve recoverability through git history while avoiding legacy clutter in public-facing paths.
- Do not permanently delete scripts during this phase; quarantine first.
