---
name: code-review
description: Repository-specific PR review guidance for high-signal, context-aware feedback.
---

# Code Review Skill

Use this skill when reviewing pull requests in this repository.

## Review objective

- Prioritize high-confidence defects, regressions, and risky design decisions.
- De-prioritize style-only feedback unless it affects correctness or maintainability.

## Repository context to apply

- This is a Python forecasting codebase with model-family dependency isolation.
- Some model work intentionally uses family-specific environments under `.venvs/<family>`.
- Repository housekeeping changes (workflows, agent skills, Copilot config) should be
  reviewed for trigger coverage and duplicate CI execution.

## Required checks by change type

### Python/model code changes

When changes touch model-family code or model wiring (for example `src/models/*`,
`src/workflows/forecasting/modeling.py`, or `src/config/schemas/model_configs.py`):

1. Verify the PR follows `docs/architecture/model-implementation-quality-checklist.md`.
2. Check for schema wiring consistency and behavior-safe defaults.
3. Ensure validation evidence is targeted (not broad, noisy sweeps by default).

### CI/workflow or Copilot configuration changes

When changes touch `.github/workflows/**`, `.github/agents/**`,
`.github/skills/**`, or `.github/copilot-instructions.md`:

1. Verify workflow trigger paths include the files they are intended to validate.
2. Check for duplicate job execution from overlapping `push` + `pull_request` triggers.
3. Confirm job names and scope are clear and aligned to purpose.

## Feedback quality bar

- Provide concrete, actionable comments tied to specific files/lines.
- Explain the impact if the issue is not fixed.
- If uncertain, ask a clarifying question instead of asserting a defect.
