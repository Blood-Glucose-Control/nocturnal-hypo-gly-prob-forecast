# GlucoFoundry Four-Repo Roadmap

**Date:** 2026-10-21
**Status:** Initial plan (ratified direction; execution not yet started)
**Supersedes scope framing of:** `00_P0-P3_MASTER_REORG_PLAN.md` (that plan targeted
cleaning a single repo; this roadmap reframes the cleanup as *toolkit extraction*).

---

## Vision

Turn this repository lineage into a reusable research toolkit plus a set of
experiment forks. The toolkit is the canonical upstream; each experiment is a
fork that carries only its own experiment layer. Reproducible core flows **up**
to the toolkit; experiment artifacts stay **down** in the forks.

### End state (four repos)

| # | Repo | Role | Scope |
|---|------|------|-------|
| 0 | **GlucoFoundry** | Toolkit / canonical upstream | Reusable, task-general TS-foundation-model core + reproduction recipes. No experiment artifacts. |
| 1 | **Nocturnal-Hypoglycemia** | Fork of GlucoFoundry | Overnight hypoglycemia risk (project #1, NeurIPS lineage). Also retains full historical record. |
| 2 | **Metabonet** | Fork of GlucoFoundry | General 2h-ahead forecasting across all time windows (competition, project #2). |
| 3 | **Meal-Image Forecast** | Fork of GlucoFoundry | Postprandial forecasting from food images; multimodal; horizon up to ~3h (project #3, new). |

### Naming

- Toolkit name: **GlucoFoundry** — a "foundry" forges/casts many artifacts
  (task-general), "found-" nods to **found**ation models, and it is the canonical
  base the forks are cast from.
- Task-general by design: supports forecasting and non-forecasting tasks
  (classification, event/anomaly detection, representation learning, imputation).

---

## Why merge private -> public first (the safety net)

Project #2 (Metabonet) was developed in the private fork
(`private/main`, including `private/metabonet-integration`). As of planning,
`private/main` is **96 commits ahead** of `origin/main` with **0 divergence**
(`origin/main` is the exact merge base). That delta is a mix of genuinely
reusable framework code (sweep/Optuna scaffolding, model adapters, tests) and
experiment artifacts (SUBMISSIONS registries, champion configs, competition
metrics).

Merging private -> public **first** consolidates the complete project record in
one place. Only then is toolkit extraction **risk-free**: GlucoFoundry can strip
aggressively because nothing is ever lost (the full record remains in the
nocturnal repo and its fork).

Note: there are also 4 project-#2 commits on local branch
`feat/moirai2-moe-lora-sweep` not yet in `private/main` (stacker/champion work)
that need a land-or-exclude decision before the merge.

---

## Phases

### Phase A - Merge private -> public nocturnal (safety net)

- `a1-land-feature-commits` - Decide fate of the 4 unmerged `feat/moirai2-moe-lora-sweep` commits (land into `private/main` or exclude).
- `a2-premerge-safety-audit` - PII/PHI, secrets, local-path, raw-data, large-file audit on the 96-commit delta (reuse `pii-secrets-audit` tooling). Gate before public exposure.
- `a3-merge-mechanism-decision` - Choose history-preserving single reviewed PR (recommended for a record repo) vs squash.
- `a4-merge-private-to-public` - Open `private/main -> origin/main` PR, run CI, merge, fast-forward public `main`.

### Phase B - Extract GlucoFoundry (risk-free)

- `b1-fork-snapshot-glucofoundry` - Create GlucoFoundry repo from merged nocturnal (carry-history vs clean init).
- `b2-toolkit-charter` - Charter/RFC drawing the core-vs-artifact line.
- `b3-strip-artifacts` - Strip artifacts; finish toolkit-ification items (runtime-core-boundary-lock, repo-root-file-governance-audit, readme-rewrite, experiments-collapse, model-runtime consolidation, reproducibility-lock, license-migration -> Apache-2.0, docs-consolidation).
- `b4-repro-recipes` - Reproduction recipes for experiments #1 and #2 (point at record forks for artifacts).
- `b5-toolkit-validate` - Fresh clone -> install -> run one smoke experiment end-to-end.

### Phase B2 - Carve experiment forks

- `c1-nocturnal-as-fork` - Re-express nocturnal as GlucoFoundry core + nocturnal experiment layer.
- `c2-metabonet-carveout` - Carve Metabonet into its own GlucoFoundry fork.
- `c3-governance-contract` - Encode core->toolkit / artifacts->fork contract in contributor docs + lightweight CI check.

### Phase C - New Meal-Image project (#3)

- `d1-fork-meal-image` - Fork GlucoFoundry into the meal-image forecast repo.
- `d2-meal-image-domain-gap` - Domain-gap analysis: vision encoder + fusion, image+nutrition+timestamp schema, ~3h postprandial horizon, new metrics; separate reused-harness vs net-new.
- `d3-meal-image-exp-data-plan` - Draft experiment + data plan (datasets, labeling, augmentation, evaluation).
- `d4-upstream-multimodal` - Contribute reusable multimodal harness pieces back up to GlucoFoundry.

---

## Critical path and open decisions

- Critical path: **A -> B -> (B2 parallel with C)**. The only immediately ready
  task is `a1-land-feature-commits`, which gates the whole merge.
- Open decisions to confirm during execution:
  1. Land vs exclude the 4 `feat/moirai2-moe-lora-sweep` commits.
  2. History-preserving vs squash for the private -> public merge.
  3. GlucoFoundry: carry full history vs clean `init`.

---

## Tracking

- Task rows mirrored in `project_tracking.csv` (ids `glucofoundry-roadmap`,
  `name-lock-glucofoundry`, `a1..a4`, `b1..b5`, `c1..c3`, `d1..d4`).
- Live structured status is also seeded in the session task DB during planning.
