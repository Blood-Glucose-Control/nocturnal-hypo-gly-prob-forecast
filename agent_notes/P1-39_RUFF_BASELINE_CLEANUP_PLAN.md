# P1-39 Ruff Baseline Cleanup Plan

**Date:** 2026-08-23
**Status:** Proposed (pending execution)
**Tracking row:** `ruff-baseline-cleanup-pass` in [project_tracking.csv](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/project_tracking.csv)

---

## 1) Why this task exists

After recent pre-commit updates, we intentionally scoped Ruff checks to changed
files to avoid pulling in large amounts of unrelated legacy lint churn inside
feature PRs.

That tactical move unblocks delivery, but the desired steady state remains:

- repo-wide Ruff/Ruff-format hygiene in CI
- predictable lint semantics pinned in config
- no surprise global rewrites inside unrelated PRs

This task creates a dedicated P1 lane for that baseline cleanup so we can
re-enable strict all-files Ruff gates cleanly.

---

## 2) Placement in sequence

This task should start **immediately after P1-15 model-family rollout closes**
(`pydantic-model-family-rollout`) and before deeper refactor/integration waves.

It should complete before:

- [P1-38 model runtime consolidation plan](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/agent_notes/P1-38_MODEL_RUNTIME_CONSOLIDATION_PLAN.md)
- `optuna-integration`
- `mlflow-integration`
- `stats-rigor-module`
- `experiments-collapse`

---

## 3) Goals

1. Establish a stable Ruff baseline across maintained repository surfaces.
2. Remove existing repo-wide Ruff violations in controlled slices.
3. Align CI/pre-commit behavior so Ruff can run all-files again without noisy,
   unrelated failures in model-feature PRs.
4. Preserve the current Pyright diff-scoped policy for multi-venv model stacks.

---

## 4) Non-goals

- No broad behavioral refactors unrelated to lint findings.
- No model-family architecture redesign.
- No expansion of Pyright to all-files in a single shared CI environment.

---

## 5) Execution plan (PR slices)

### PR-R0 — Configuration freeze

- Confirm/pin Ruff hook versions and rule selectors in
  [.pre-commit-config.yaml](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/.pre-commit-config.yaml).
- Explicitly pin Ruff target Python semantics in
  [pyproject.toml](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/pyproject.toml)
  (avoid parser drift surprises).

### PR-R1 — Core `src/` cleanup

- Run repo-wide Ruff on `src/` and fix violations.
- Keep changes mechanical (lint-driven), no behavior drift.

### PR-R2 — Tests cleanup

- Run repo-wide Ruff on `tests/` and fix violations.
- Keep fixtures and assertions semantically identical.

### PR-R3 — Scripts/notebooks policy slice

- Decide and document whether notebooks are Ruff-linted or excluded.
- Apply consistent policy in pre-commit config.

### PR-R4 — CI policy restore

- Restore CI Ruff/Ruff-format to all-files execution once R0–R3 are clean.
- Keep Pyright as changed-files scoped in
  [.github/workflows/cicd.yml](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/.github/workflows/cicd.yml).

---

## 6) Validation gates

Per slice:

```bash
pre-commit run ruff --all-files
pre-commit run ruff-format --all-files
```

Plus targeted regression tests for touched runtime surfaces.

Final gate:

```bash
pre-commit run --all-files
```

with the intended split policy preserved only where documented (Pyright).

---

## 7) Exit criteria

Task completes when:

1. Repo-wide Ruff + Ruff-format pass in CI with stable pinned semantics.
2. All temporary diff-scoped Ruff workarounds are removed.
3. Pyright remains intentionally diff-scoped (documented rationale retained).
4. Contributor docs reflect the final lint policy and workflow.
