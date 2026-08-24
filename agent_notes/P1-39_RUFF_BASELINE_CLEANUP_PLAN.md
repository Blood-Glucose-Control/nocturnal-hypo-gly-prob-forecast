# P1-39 Ruff Baseline Cleanup Plan

**Date:** 2026-08-23
**Update Date:** 2026-08-23
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
4. Preserve a runtime-core Pyright diff-scoped policy for multi-venv model stacks
   (`src/models`, `src/workflows`, `src/configs`, and matching `tests/*` lanes).
5. Complete targeted utility-module cleanup for Chronos2/Tide surfaces that are
   currently messy or misplaced so P1-38 starts from a cleaner `src/models/` baseline.

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
- Decision (2026-08-23): when Chronos2/Tide helper logic is clearly shared or
  misplaced, move it to a common maintained location under
  [src/models/](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/)
  and rewire imports now (do not defer this class of cleanup to a later phase).
- Include targeted hygiene/placement cleanup for:
  - [chronos2/utils.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/chronos2/utils.py)
  - [tide/utils.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/tide/utils.py)
  - [tide/visualization.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/tide/visualization.py)
  where these helpers violate current lint/organization expectations.

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
5. Chronos2/Tide utility-module cleanup from PR-R1 is completed and reflected in
   P1-38 handoff notes.
6. Non-runtime paths narrowed out of the Pyright pre-commit hook are tracked with
   an explicit follow-up task and triage plan.

---

## 8) Explicit configuration items to assess in P1-39

As part of PR-R0, explicitly evaluate and either ratify or revise the current
pre-commit entries in
[.pre-commit-config.yaml](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/.pre-commit-config.yaml):

```yaml
      - id: ruff
        args: ["--fix", "--select=E4,E7,E9,F,I"]
        types_or: [python, pyi]
      - id: ruff-format
        types_or: [python, pyi]

  - repo: https://github.com/RobertCraigie/pyright-python
    rev: v1.1.411
    hooks:
      - id: pyright
        args: [--project=pyrightconfig.json]
        files: ^(src|tests)/.*\.py$
```

Assessment checklist:

1. Confirm whether `--select=E4,E7,E9,F,I` remains the intended long-term Ruff
   policy or is a temporary baseline clamp during cleanup.
2. Confirm `types_or: [python, pyi]` stays in place to prevent notebook/content
   parsing churn during repo-wide runs.
3. Confirm Pyright `files` scope remains runtime-core focused:
   `^((src|tests)/(models|workflows|configs)/.*\.py)$`
   (with temporary exclusion for deprecated model subpaths),
   and that narrowed-out paths are tracked for follow-up.
4. Lock Ruff Python semantics to 3.9+ and align
   [pyproject.toml](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/pyproject.toml)
   minimum Python accordingly (decision captured 2026-08-23).
5. Document final keep/change decisions in this plan and mirror them in
   contributor-facing docs once P1-39 closes.
