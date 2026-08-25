# Copilot instructions for nocturnal-hypo-gly-prob-forecast

These instructions apply to Copilot work in this repository.

## Project workflow expectations

- Keep changes phase-scoped and avoid unrelated refactors.
- Prefer targeted commands and tests over full-repo sweeps.
- Provide concise progress updates for major transitions.
- Preserve external behavior contracts by default; within approved cleanup scope, remove legacy internal behavior and compatibility shims. If a contract change is intended, state it explicitly and update tests/docs accordingly.

## Branch and PR hygiene

- Before substantial new work, confirm branch state and sync with `origin/main`.
- If asked for housekeeping, complete git cleanup before implementation.
- When opening PRs, include:
  - what changed,
  - validation commands and results,
  - any intentionally deferred follow-ups.

## Python environment policy

- Use Python 3.12 for shared checks and CI-aligned validation.
- Default/shared work should use `.noctprob-venv`.
- Model-family work may require `.venvs/<family>` due to dependency conflicts.
- If imports fail in one environment, switch to the appropriate model-family interpreter before concluding code is broken.

## Validation policy

Run the smallest checks that validate the requested change:

1. Targeted tests for touched behavior.
2. Targeted lint/type checks for touched files.
3. Broader checks only when targeted checks fail due to cross-cutting issues.

Common commands:

- `pytest -v --color=yes --ignore=tests/models`
- `pre-commit run ruff --from-ref origin/<base> --to-ref HEAD`
- `pre-commit run ruff-format --from-ref origin/<base> --to-ref HEAD`
- `pre-commit run pyright --from-ref origin/<base> --to-ref HEAD`

For model integration coverage, use the model-specific make targets in `Makefile` (for example `make test-ttm`, `make test-sundial`, `make test-timesfm`, `make test-autogluon`) rather than running all model-family tests by default.

## Model/schema change gate

When touching model-family code, schema adapters, or workflow model wiring, follow:

- `.github/skills/noctprob-python-quality-gates/SKILL.md`
- `docs/architecture/model-implementation-quality-checklist.md`

## Documentation placement

- Public docs in `docs/` should describe stable contracts and usage.
- Phase/task tracking detail belongs in planning/tracking artifacts (for example `agent_notes/` and tracking CSV), not in canonical public docs unless explicitly requested.
