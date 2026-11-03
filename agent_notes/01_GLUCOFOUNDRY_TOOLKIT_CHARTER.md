# GlucoFoundry Toolkit Charter

**Date:** 2026-11-02
**Status:** Ratified design (Phase B foundation). Execution tracked separately.
**Related:** `00_GLUCOFOUNDRY_FOUR_REPO_ROADMAP.md`

---

## 1. Purpose

Define the boundary that turns this repository lineage into **GlucoFoundry** — a
task-general, reusable research toolkit — versus the experiment-specific and
artifact material that belongs in the downstream forks. This charter is the
design contract for the Phase B extraction and the strip plan.

## 2. Core design principle

GlucoFoundry is **toolkit core + droppable example experiments + empty artifact
scaffolding**. Reproducible code flows *up* to the toolkit; experiment artifacts
stay *down* in the forks.

Concretely, GlucoFoundry ships:

- the reusable machinery (data, models, sweeps, evaluation metrics, workflows),
- a set of **example experiment pipelines** that were never run here but can be
  run by a user to generate *their own* artifacts, and
- **empty, scaffolded** artifact directories.

A fork of GlucoFoundry keeps the experiments it needs, **drops the directories it
doesn't**, runs the pipelines, and accumulates its own artifacts.

## 3. Droppable per-experiment architecture (the central design)

### 3.1 Consistent per-experiment layout

Each experiment is a **single self-contained, droppable directory** at the top
level, importing the pure `src/` core (which never imports experiments). The
installed library (`src/`) stays experiment-agnostic; experiments are runnable
**recipes**, not installed package weight — the standard library+examples pattern
used by mature ML projects.

```
experiments/<experiment>/
    eval.py          # experiment-specific evaluation / metrics / anchoring
    viz.py           # experiment-specific visualization
    configs/         # experiment-tuned sweep/eval configs
    run.py           # entrypoint
    register.py      # registration hook (auto-discovered)
    README.md        # how to reproduce this experiment
```

For example: `experiments/nocturnal/`, `experiments/metabonet/`. Cloning
GlucoFoundry and deleting `experiments/metabonet/` removes that experiment
cleanly, with no dangling imports. The repo's existing top-level `experiments/`
directory is **repurposed** as the home for these recipes (not stripped).

### 3.2 Core must never import experiments (the key enabler)

Today the core hardcodes experiment imports, so experiments are **not** actually
droppable — deleting an experiment module breaks package import. Known coupling
sites to remove/redirect:

1. `src/evaluation/__init__.py` — imports `.episode_builders` and `.nocturnal`.
2. `src/experiments/__init__.py` — imports `.nocturnal.summarize`.
3. `src/visualization/__init__.py` — imports `.nocturnal`.
4. `src/models/chronos2/utils.py` — imports `evaluation.episode_builders`.
5. `src/workflows/evaluation/validate_predict_batch.py` — imports `episode_builders`.
6. `src/workflows/forecasting/evaluation.py` — imports `episode_builders`.

Note that `episode_builders` (midnight/nocturnal episode construction) is
entangled into workflow and model code. Decoupling requires either promoting a
**generic** episode-construction abstraction into core and keeping only the
nocturnal-specific policy in the nocturnal experiment, or redirecting these
callers through the registration layer.

**Rule:** core packages (`data`, `models`, `config`, `workflows`, generic
`evaluation`, generic `visualization`, `utils`) must not import from any
experiment package. Dependency direction is one-way: experiment → core.

### 3.3 Registration: auto-discovery

Experiments self-register via **auto-discovery**: the core scans the top-level
`experiments/*/` directories for a registration hook (`register.py`) at
startup, so dropping a directory simply removes that experiment with no edits
elsewhere. This mirrors the existing data-loader registry pattern already used in
`src/data`, kept consistent across the codebase.

A registration hook per experiment declares its summarizer, evaluation entry
point, visualization surfaces, and any config schema, so generic core
(including `workflows`) can resolve experiments by name without importing them
directly. In particular, `workflows/` stays generic core and calls the
registered experiment's evaluation/anchoring through this interface rather than
importing experiment modules.

### 3.4 Empty artifact scaffolding

Artifact directories ship **empty but structured**:

- `results/` and `trained_models/` contain only a `.gitkeep` and a short README
  explaining that experiment runs populate them.
- No actual artifacts (submissions, study DBs, prediction parquets, trained
  weights) ship in GlucoFoundry.

## 4. Content classification

### A. Core — KEEP in GlucoFoundry
- `src/models/` (ModelAdapter protocol, `base/`, `factory.py`, all adapters)
- `src/workflows/` (sweep framework, forecasting, runtime, personalization)
- `src/data/` (loaders, schemas, preprocessing, transformers, datasets, versioning)
- `src/config/`; generic `src/evaluation/` (`metrics/`, `evaluator.py`,
  `storage.py`, `stats/`); `src/utils/` (incl. `dilate`); generic
  `src/visualization/`; `src/experiments/base/`
- `tests/` for the above; `configs/data|evaluation|models` templates
- `docs/` toolkit usage; root infra (`pyproject`, `Makefile`, `.pre-commit`,
  `mkdocs`, `LICENSE`, `CONTRIBUTING`, `pyrightconfig`, README)

### B. Example experiments — KEEP as runnable recipes (no artifacts)
Consolidated into top-level `experiments/<experiment>/` recipes that import the
pure `src/` core and self-register via auto-discovery:
- nocturnal: `experiments/nocturnal/` (`eval.py`, `viz.py`, `configs/`, `run.py`,
  `register.py`, `README.md`) + its tests
- (future) metabonet and others follow the same shape

These are the reproducibility recipes: code only, run to produce artifacts. The
experiment-specific logic currently in `src/evaluation/nocturnal.py`,
`src/visualization/nocturnal.py`, `src/experiments/nocturnal/`, and
`src/workflows/evaluation/nocturnal_hypo_eval.py` moves here.

### C. Artifacts / record / meta — STRIP from GlucoFoundry
- `results/`, `trained_models/` → empty scaffolds only (see 3.4)
- `mlflow/`, `demo/`, `images/`, `docs-internal/` notebooks
- `agent_notes/`, `project_tracking.csv` (planning meta)
- Competition artifacts (already excluded from public; live in private → the
  Metabonet fork)

## 5. Ratified nuanced decisions

- **Sweep configs:** keep the sweep *framework*; **move experiment-tuned search
  spaces** into the owning `experiments/<experiment>/configs/`.
- **Top-level `experiments/`:** **repurpose** as the home for droppable
  experiment recipes (not stripped). The current experiment-output dirs
  (`nocturnal_forecasting*`, `standard_forecasting`, `gpu_optimization`) are
  removed/replaced by the clean recipe structure.
- **`results/`, `trained_models/`:** ship as empty scaffolded directories.
- **`workflows/` stays generic core:** experiments plug in their
  evaluation/anchoring via the registration interface; `workflows` never imports
  experiment modules.

## 6. Reproducibility model

GlucoFoundry = toolkit + **example pipelines that were never run here**. A user:

1. forks GlucoFoundry,
2. drops experiments they don't need,
3. runs an experiment pipeline to produce their own artifacts,
4. adjusts/extends as needed, and
5. contributes genuinely reusable improvements back up to the toolkit.

## 7. Derived Phase B workstreams

All executed on the **GlucoFoundry** fork (the old nocturnal repo stays a frozen
record and is not refactored):

1. **Decouple + register** (`b3a`) — remove the 6 core→experiment imports; add
   auto-discovery registration scanning `experiments/*/register.py`. *Central
   enabler.*
2. **Per-experiment recipes** (`b3b`) — move nocturnal's eval/viz/experiment/
   workflow-eval logic into a self-contained top-level `experiments/nocturnal/`
   recipe importing the pure `src/` core.
3. **Scaffold artifact dirs** (`b3c`) — empty `results/`, `trained_models/`.
4. **Strip artifacts + experiment-tuned configs** (`b3d`) — move experiment-tuned
   sweep search spaces into the owning recipe; remove artifact/record/meta dirs.
