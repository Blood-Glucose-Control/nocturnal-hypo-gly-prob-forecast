# P1-15 Pydantic Config Schemas — Execution Plan

**Date:** 2026-08-20
**Task ID:** `pydantic-config-schemas`
**Status:** Planning refreshed; implementation next

## What this task means (plain English)

Today, many config paths accept loosely typed dict/dataclass inputs.
This task introduces **Pydantic v2 schemas** as the validation layer so config files are:

- validated at load time with clear errors,
- strongly typed in code,
- versionable/documentable (JSON schema output),
- safer to evolve without silent misconfiguration.

This is **not** a behavior rewrite of model training/eval logic; it is a config contract hardening effort.

## Goals

1. Validate model/data/evaluation config inputs before runtime execution.
2. Keep migration incremental (no flag-day break).
3. Preserve existing entrypoint behavior while adding strict validation boundaries.
4. Generate machine-readable schema docs from source-of-truth models.

## Non-goals

- Rewriting all workflow logic in one pass.
- Breaking existing YAML surfaces abruptly.
- Introducing parallel config systems long-term.

## Proposed rollout

### Phase 1 — Schema foundation

- Create schema package: `src/config/schemas/`.
- Add shared base model policy (strict mode, extra field handling, aliases).
- Add loader helpers:
  - parse YAML → schema validate → normalized typed object
  - emit actionable validation errors with file path and field path.

### Phase 2 — Pilot family migration

- Pick one active model family as pilot (recommend Darts/TSMixer path due recent runtime hardening).
- Add schema + adapter layer from schema object to runtime config class.
- Cover with focused tests for valid + invalid configs.

### Phase 3 — Broaden by domain

- Expand migration slices:
  1. model configs
  2. data configs
  3. evaluation/workflow configs
- Add compatibility shims where needed during transition (explicitly time-bounded).

### Phase 4 — Consolidation

- Remove obsolete duplicate validation paths.
- Generate JSON schema artifacts for docs.
- Finalize contributor documentation for adding/modifying config schema fields.

## Validation plan

- Unit tests for schema parsing and error reporting.
- Regression tests for existing canonical entrypoints using migrated configs.
- `ruff` + targeted pytest on touched modules.
- Smoke-check that previously valid configs still run unchanged unless intentionally tightened.

## Risks and controls

- **Risk:** partial migration causes dual-path confusion.
  **Control:** single loader helper and explicit migration table per domain.

- **Risk:** over-strict schema breaks common configs.
  **Control:** pilot first, capture real-world config examples, tighten iteratively.

- **Risk:** undocumented behavior changes.
  **Control:** changelog notes + updated contributor docs in same PR slices.

## Exit criteria

- Canonical model/data/eval config paths validate through Pydantic schemas.
- Existing runtime entrypoints pass targeted regression checks.
- Schema docs are generated from code and linked in docs.
- Deprecated pre-schema validation paths are removed or explicitly sunset-tracked.

## Immediate next slice (what to implement next)

1. Create `src/config/schemas/` with shared base schema class and loader helper.
2. Migrate one pilot config path end-to-end (YAML load → schema → runtime object).
3. Add focused tests proving:
   - valid config passes,
   - invalid config fails early with actionable diagnostics.
