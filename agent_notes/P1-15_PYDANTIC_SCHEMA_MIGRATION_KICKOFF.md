# P1-15 Pydantic Schema Migration Kickoff

**Date:** 2026-08-13
**Task ID:** `pydantic-config-schemas`
**Status:** Kickoff started (implementation pending)

---

## Current state snapshot

- Most model/data configs are dataclass-based (e.g. under [src/models/](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models)).
- Pydantic is already present in parts of the data layer (e.g. [models.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/data/models.py)).
- Experiment config placeholders currently exist in [configs/experiments/](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/configs/experiments) but are empty.

---

## First implementation slice (next coding step)

1. Introduce a schema package (e.g. `src/config/schemas/`) with:
   - shared base settings (strict mode, extra field policy),
   - typed schema for one pilot model family.
2. Add loader path that validates YAML through schema before building model config.
3. Keep current dataclass configs during transition; no flag day migration.

---

## Guardrails

- Do not break existing training/eval entrypoints while schemas are introduced.
- Keep schema migration incremental by model family.
- Add focused validation tests per migrated family.
