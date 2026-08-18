# P1-34 Workspace Triage and Simplification Plan

**Date:** 2026-08-13
**Scope:** `src/*`, `scripts/*`, Pylance + static reachability evidence

---

## 1) Workspace diagnostics triage (current state)

### 1.1 Active model runtime files

After the second-pass diagnostics cleanup, these active runtime model files were rechecked and are clean in current Pylance diagnostics:

- [`src/models/ttm/model.py`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/ttm/model.py)
- [`src/models/timesfm/model.py`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/timesfm/model.py)
- [`src/models/patchtst/model.py`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/patchtst/model.py)
- [`src/models/chronos2/model.py`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/chronos2/model.py)

### 1.2 Remaining unresolved-import pressure (workspace-level)

`pylanceImports` still reports unresolved top-level modules (`src`, `autogluon`, `chronos`, `gluonts`, `uni2ts`, `momentfm`, `pts`, `toto`, `lightning`) in the workspace aggregate, but this is no longer concentrated in the active model runtime path above.

---

## 2) `src/*` reachability triage

AST-based import graph scan across [`src/`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src) and [`scripts/`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/scripts):

- `data`: **177** imports
- `models`: **177** imports
- `utils`: **55** imports
- `evaluation`: **30** imports
- `experiments`: **29** imports
- `training`: **1** import
- `tuning`: **2** imports
- `registry`: **0** imports

### 2.1 High-confidence dead/stub surfaces

The following package surfaces are effectively non-runtime today:

- [`src/registry/`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/registry) — all Python files zero-byte stubs.
- [`src/training/`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/training) — mostly zero-byte stubs; not used by runtime scripts.
- [`src/tuning/`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/tuning) — only minimal package marker.

### 2.2 Empty-module burden

There are **50 zero-byte Python files** in [`src/`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src), concentrated in:

- `experiments` (19)
- `training` (11)
- `models` (10; mostly dormant model families)
- `registry` (7)
- `evaluation` (2)
- `data` (1)

This is high-confidence confusion debt for humans and coding agents.

### 2.3 Runtime-core imports from experiment entrypoints

Current primary runtime scripts under [`scripts/experiments/`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/scripts/experiments) consistently use:

- [`src.data.versioning.dataset_registry`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/data/versioning/dataset_registry.py)
- [`src.data.utils`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/data/utils.py)
- [`src.models.create_model_and_config`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/factory.py:17)
- [`src.evaluation.nocturnal`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/evaluation/nocturnal.py) / [`src.evaluation.storage`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/evaluation/storage.py) / metrics
- [`src.utils`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/utils)

This supports a clean “runtime-core first” pruning strategy.

---

## 3) `scripts/*` inventory and organization pressure

Current counts:

- Python scripts: **56**
- Shell scripts: **64**

Python distribution:

- `visualization`: 15
- `analysis`: 14
- `examples`: 12
- `data_processing_scripts`: 5
- `experiments`: 5
- root-level: 2
- `training`: 2
- `code_benchmarking`: 1

Shell distribution:

- `scripts/experiments`: 36
- `scripts/training`: 12
- `scripts/data_processing_scripts`: 8
- `scripts/examples`: 4
- root-level: 2

Existing cleanup summary/status now lives in [`P1-37_CANONICAL_HANDOFF_AND_STATUS.md`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/agent_notes/P1-37_CANONICAL_HANDOFF_AND_STATUS.md).

---

## 4) Decision guidance: remove vs migrate

### 4.1 `registry`, `training`, `tuning` modules

Given current project goals and measured usage:

- **Do not migrate runtime code into these now** (would add abstraction cost without immediate value).
- **Prefer pruning/deleting empty scaffolds** and documenting a small, explicit runtime architecture.
- Reintroduce these modules only when concrete implementation and call sites are ready in the same PR.

### 4.2 Clean target architecture (near-term)

Keep runtime shape small and explicit:

- Data lifecycle in [`src/data/`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/data)
- Model lifecycle in [`src/models/`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models) via [`create_model_and_config`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/factory.py:17)
- Evaluation lifecycle in [`src/evaluation/`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/evaluation)
- Thin orchestration scripts in [`scripts/experiments/`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/scripts/experiments)

No parallel “shadow architecture” directories should remain half-populated.

---

## 5) Execution order (safe)

1. Freeze runtime core boundaries in docs/tests.
2. Delete zero-byte and non-imported scaffolds in `src/registry`, `src/training`, `src/tuning` (single PR, reversible tag first).
3. Execute script reorg from [`P1-37_CANONICAL_HANDOFF_AND_STATUS.md`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/agent_notes/P1-37_CANONICAL_HANDOFF_AND_STATUS.md) and linked archive notes as needed.
4. Remove dormant model-family placeholders with no runtime wiring (after explicit check against planned P1/P2 scope).
5. Re-run focused runtime smoke flows (`per_patient_finetune`, `nocturnal_hypo_eval`, `sliding_window_eval`) plus diagnostics.
