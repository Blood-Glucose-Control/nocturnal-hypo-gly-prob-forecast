# Copilot instructions for nocturnal-hypo-gly-prob-forecast

These instructions apply to Copilot work in this repository.

## Project workflow expectations

- Keep changes phase-scoped and avoid unrelated refactors.
- Prefer targeted commands and tests over full-repo sweeps.
- Provide concise progress updates for major transitions.
- Preserve external behavior contracts by default; within approved cleanup scope, remove legacy internal behavior and compatibility shims. If a contract change is intended, state it explicitly and update tests/docs accordingly.
- When working with a developer always confirm the branch state and sync with `origin/main` before starting new work. If asked for housekeeping, complete git cleanup before implementation.
- At the end of each task, always follow up with the developer by prompting them with questions on how to proceed with a few options. This prevents unnecessary premium request usage for very minor clarification questions. The question should be short, keep the summary of what you did in the chat output, not the question.

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
- `SKIP=pyright pre-commit run --all-files`
- `pre-commit run pyright --from-ref origin/<base> --to-ref HEAD`

Current notebook lint policy: Ruff hooks are scoped to
`types_or: [python, pyi]` (notebooks excluded).

For model integration coverage, use the model-specific make targets in `Makefile` (for example `make test-ttm`, `make test-sundial`, `make test-timesfm`, `make test-autogluon`) rather than running all model-family tests by default.

## Model/schema change gate

When touching model-family code, schema adapters, or workflow model wiring, follow:

- `.github/skills/noctprob-python-quality-gates/SKILL.md`
- `docs/architecture/model-implementation-quality-checklist.md`

## Documentation placement

- Public docs in `docs/` should describe stable contracts and usage. No internal project management metadata (phase IDs, wave names, task IDs, tracking-file row references, private branch names, or temporary implementation narration) should be included.
- Phase/task tracking detail belongs in planning/tracking artifacts (for example `agent_notes/` and tracking CSV), not in canonical public docs unless explicitly requested.
- Before editing public docs, verify statements against source-of-truth config
  files (`pyproject.toml`, `.pre-commit-config.yaml`, `.github/workflows/*`).

## Repository governance encoding (hard rules)

- Treat `docs/` as public-facing and durable. Do not include internal project
  management metadata there (phase IDs, wave names, task IDs, tracking-file
  row references, private branch names, or temporary implementation narration).
- If a temporary behavior must be documented in `docs/`, describe only the
  current observable behavior and user impact. Keep rationale concise and avoid
  references to internal planning artifacts.
- Keep policy text location-specific:
  - stable usage/contract guidance -> `docs/`
  - execution sequencing/status/temporary rollout notes -> `agent_notes/` and
    `project_tracking.csv`
- When touching `docs/contributing.md`, also fix any adjacent obvious
  inaccuracies discovered in the same section (for example version minimums or
  commands that no longer match CI).
