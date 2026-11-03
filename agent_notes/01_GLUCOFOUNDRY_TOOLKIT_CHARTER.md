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

Every experiment is a **self-contained, droppable subdirectory** under a uniform
`<domain>/<experiment>/` convention:

```
src/experiments/<experiment>/<experiment>.py
src/evaluation/<experiment>/<experiment>.py
src/visualization/<experiment>/<experiment>.py
configs/experiments/<experiment>/...
```

For example: `src/evaluation/nocturnal/nocturnal.py`,
`src/evaluation/metabonet/metabonet.py`. Flat files such as the current
`src/evaluation/nocturnal.py` are restructured into this form.

Goal: cloning GlucoFoundry and deleting, say, `*/metabonet/` directories should
remove that experiment cleanly, with no dangling imports or broken package load.

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

Experiments self-register via **auto-discovery**: the core scans
`src/experiments/*/` (and parallel evaluation/visualization experiment packages)
for a registration hook at import/startup, so dropping a directory simply removes
that experiment with no edits elsewhere. This mirrors the existing data-loader
registry pattern already used in `src/data`, kept consistent across the codebase.

A registration hook per experiment declares its summarizer, evaluation entry
point, visualization surfaces, and any config schema, so the rest of the system
can resolve experiments by name without importing them directly.

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

### B. Example experiments — KEEP as runnable pipelines (no artifacts)
Restructured into the per-experiment layout and decoupled via registration:
- nocturnal: `src/experiments/nocturnal/`, `src/evaluation/nocturnal/`,
  `src/visualization/nocturnal/`, nocturnal `configs/experiments/`, tests
- (future) metabonet and others follow the same shape

These are the reproducibility recipes: code only, run to produce artifacts.

### C. Artifacts / record / meta — STRIP from GlucoFoundry
- `results/`, `trained_models/` → empty scaffolds only (see 3.4)
- `mlflow/`, `demo/`, `images/`, `docs-internal/` notebooks
- `agent_notes/`, `project_tracking.csv` (planning meta)
- Competition artifacts (already excluded from public; live in private → the
  Metabonet fork)

## 5. Ratified nuanced decisions

- **Sweep configs:** keep the sweep *framework*; **strip experiment-tuned search
  spaces** to the forks.
- **Top-level `experiments/`:** strip entirely (`standard_forecasting`,
  `gpu_optimization`) — `experiments/` is experiment-only, not toolkit material.
- **`results/`, `trained_models/`:** ship as empty scaffolded directories.

## 6. Reproducibility model

GlucoFoundry = toolkit + **example pipelines that were never run here**. A user:

1. forks GlucoFoundry,
2. drops experiments they don't need,
3. runs an experiment pipeline to produce their own artifacts,
4. adjusts/extends as needed, and
5. contributes genuinely reusable improvements back up to the toolkit.

## 7. Derived Phase B workstreams

1. **Decouple + register** (`b3a`) — remove the 6 core→experiment imports; add
   auto-discovery registration. *Central enabler.*
2. **Per-experiment layout** (`b3b`) — restructure to `<domain>/<experiment>/`.
3. **Scaffold artifact dirs** (`b3c`) — empty `results/`, `trained_models/`.
4. **Strip experiment configs/dirs** (`b3d`) — experiment-tuned sweeps and
   top-level `experiments/`.
