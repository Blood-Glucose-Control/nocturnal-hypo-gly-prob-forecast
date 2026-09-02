---
name: noctprob-python-quality-gates
description: Repository quality gate for Python edits. Use whenever changing Python files, especially src/models, schema adapters, or workflow model wiring.
---

# NoctProb Python Quality Gates

Use this skill for Python changes in this repository.

## 1) Interpreter expectations

- Use the shared `.noctprob-venv` for general/core work.
- For model-family (`src/models/*`) work and model training/inference flows, use the model-specific
  environment in `.venvs/<family>/bin/python` when that family has isolated deps.
- If imports are unexpectedly unresolved in model code, verify the selected
  interpreter before concluding the code is broken.

Reference:
- `docs/contributing.md` model-family interpreter section and mapping template.

## 2) Model-change checklist requirement

If changes touch model-family code or model wiring (for example `src/models/*`,
`src/workflows/forecasting/modeling.py`, or `src/config/schemas/model_configs.py`):

1. Open and follow `docs/architecture/model-implementation-quality-checklist.md`.
2. Use its PR acceptance items as a merge gate.
3. Include checklist-driven validation in the final report.

## 3) Mandatory Pylance final gate

Before completing Python work:

1. Run Pylance diagnostics on every touched Python file.
2. Apply fixes for all error-severity diagnostics (red squiggles).
3. If formatters or pre-commit modify files, rerun Pylance diagnostics on the
   modified files.
4. Do not mark work complete until the final Pylance pass is clean, or explicitly
   document any approved exception.

## 4) Validation order

Use this order for changed Python files:

1. Implement code changes.
2. Run targeted lint/tests for touched behavior.
3. Run pre-commit/format hooks used by the repo.
4. Run the Pylance final gate last.
5. Report: tests/lint status + final Pylance status.

## 5) Workspace-path guardrail for diagnostics

- Before running/reading Pylance diagnostics, ensure workspace roots use the
  canonical repo path:
  `/data/home/<you>/nocturnal-hypo-gly-prob-forecast`
- Do not rely on symlink-root sessions (for example `/data/home/<you>/nocturnal`)
  for final diagnostic sign-off, because path/remount drift has previously caused
  misleading results.
