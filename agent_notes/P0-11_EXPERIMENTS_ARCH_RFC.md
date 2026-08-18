# P0-11 Experiments Architecture RFC

**Date:** 2026-08-13
**Status:** Accepted (decision-only; no code changes in this task)
**Tracking ID:** `experiments-arch-rfc`

---

## 1) Decision

Use a single, explicit meaning of **experiment**:

> An experiment is an immutable run instance produced by a typed run specification (`RunSpec`) executed by a pipeline runner, with outputs written to a canonical runs store and indexed by a run manifest.

Canonical flow:

`configs/` (declarative run specs) → `src/pipelines/` (execution) → `runs/` (artifacts + manifest)

---

## 2) Why this decision

Current repo has four partially overlapping experiment surfaces:

- runtime artifacts in [`experiments/`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/experiments)
- code in [`src/experiments/`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/experiments)
- sparse placeholders in [`configs/experiments/`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/configs/experiments)
- many launcher scripts in [`scripts/experiments/`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/scripts/experiments)

This creates ownership ambiguity (what is source-of-truth config vs execution vs output).
The selected architecture enforces one direction of truth and sets up P1 tasks (`run-manifest-schema`, `mlflow-integration`, `optuna-integration`, `experiments-collapse`).

---

## 3) Target architecture (post-P1 collapse)

```text
configs/
  experiments/
    <experiment_family>/
      <run_spec>.yaml

src/
  pipelines/
    run_experiment.py                # generic entrypoint
    experiments/
      <experiment_family>.py         # orchestration runners
  evaluation/
    stats/                           # long-lived reusable stats modules
    analysis/                        # experiment-level analysis utilities

runs/                                # canonical runtime artifacts (new)
  <experiment_family>/
    <ctx_fh_or_variant>/
      <model>/
        <run_id_timestamp>/
          run_manifest.json
          metrics*.json
          plots/
          logs/
          checkpoints/ (optional)
```

---

## 4) Mapping from current four folders

### A. [`configs/experiments/`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/configs/experiments) (declarative surface)

- Keep as the canonical location for experiment run specs.
- Populate currently empty placeholders (`point_forecasts.yaml`, `probabilistic_forecast.yaml`) with typed run specs in P1 after schema work.
- Run specs should reference model/data config IDs, not hardcoded local paths.

### B. [`src/experiments/`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/experiments) (logic surface)

- Migrate orchestration responsibilities to `src/pipelines/experiments/*`.
- Keep reusable summarization/analysis logic but relocate by role:
  - execution/orchestration → `src/pipelines/`
  - reusable analysis/statistics → `src/evaluation/analysis/` or `src/evaluation/stats/`
- Preserve compatibility import shims for one release.

### C. [`scripts/experiments/`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/scripts/experiments) (operator interface)

- Keep as thin operational wrappers only.
- Wrappers should call stable Python entrypoints (`python -m src.pipelines.run_experiment ...`).
- Remove local-path-coupled one-off scripts from public surface during collapse.

### D. [`experiments/`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/experiments) (artifact surface)

- Treat as current artifact store to be migrated to canonical `runs/`.
- During transition:
  1. write new runs to `runs/`,
  2. keep read support for existing `experiments/`,
  3. maintain compatibility shim/symlink for one release window.

---

## 5) Constraints and guardrails

1. No destructive movement of historical artifacts without backup verification.
2. Keep training/eval data paths under existing read-only safety conventions.
3. Every pipeline run must emit `run_manifest.json` (P1 dependency).
4. No hardcoded absolute local paths in public-facing run specs or wrappers.
5. Compatibility period: one release with shims for old paths/imports.

---

## 6) Definition of done for this RFC task

This P0 task is complete when:

- a single experiment architecture decision is documented,
- the four existing experiment folders are explicitly mapped to target roles,
- migration constraints for `experiments-collapse` are clear.

All three are satisfied by this document.

---

## 7) Immediate handoff into P1

`experiments-collapse` must execute this RFC in implementation form, after:

- `pydantic-config-schemas` (typed config layer),
- `run-manifest-schema` (artifact contract),
- `mlflow/optuna` integration direction is stable.
