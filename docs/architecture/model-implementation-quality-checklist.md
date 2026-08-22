# Model Implementation Quality Checklist

Status: active contributor standard for all model-family migrations and refactors.

This checklist defines the minimum quality bar for model modules under
[`src/models/`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models)
and related workflow integration points.

## Purpose

Ensure model implementations are:

1. Consistent across families.
2. Easy to reason about and maintain.
3. Safe to evolve without interface drift.
4. Validated by a shared, repeatable test contract.

## Scope

Apply this checklist whenever a PR changes:

- model classes/configs in `src/models/*`,
- model factory/adapter wiring in
  [`modeling.py`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/workflows/forecasting/modeling.py),
- model config schemas in
  [`model_configs.py`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/config/schemas/model_configs.py).

## 1) Interface consistency contract

### Required public contract

Each model implementation should conform to the same public surface inherited
from the base model framework unless there is a documented, necessary exception.

- Constructor accepts a typed config object (model-specific config dataclass).
- Consistent implementation of required properties (for example:
  `training_backend`, `supports_zero_shot`, probabilistic support flag).
- Predict/fit/load/save behavior follows base-class contract semantics.

### Allowed deviations

Deviations are allowed only when model backend constraints require them
(for example AutoGluon-managed training internals). In that case:

1. Document the reason in module docstring and PR notes.
2. Keep external/public behavior consistent with the shared contract.
3. Add targeted tests proving parity at the public interface level.

## 2) Module structure and ordering standard

To keep modules readable and uniform, use this section order:

1. Module docstring (purpose + backend notes)
2. Imports
3. Constants/helpers (if any)
4. Config class
5. Model class
6. Public methods/properties in stable order:
   - lifecycle/init
   - required properties
   - training path
   - inference path
   - serialization/load/save helpers
   - private helpers

When reordering in existing files, preserve behavior and keep the diff focused.

## 3) Dead code and drift control

Before merging model changes:

- Remove unused helpers/branches that no longer serve active runtime paths.
- Eliminate duplicate logic already centralized in shared bases/adapters.
- Avoid parallel legacy pathways unless explicitly transition-scoped and tracked.

## 4) OOP quality requirements

### Encapsulation

- Keep backend-specific details private/internal where possible.
- Expose only stable, intentional public methods.

### Abstraction

- Reuse base abstractions (`BaseTimeSeriesFoundationModel`, adapters, schema
  loaders) instead of duplicating workflow logic per model.

### Inheritance

- Inherit from shared base classes when behavior is shared.
- Override narrowly and intentionally; avoid copy-paste subclassing patterns.

### Polymorphism

- Model classes should be interchangeable at the workflow layer via shared
  factory and base interfaces.
- Workflow code should depend on shared contracts, not model-specific internals.

## 5) Documentation and docstring standard

- Module docstring explains:
  - backend/runtime role,
  - key behavior differences,
  - invariants/limitations that affect usage.
- Class docstring explains intended use and contract obligations.
- Method docstrings explain inputs/outputs and side effects when non-trivial.
- Keep tone concise, factual, and implementation-aligned.

## 6) Linting and static quality gates

No model PR should merge with lint warnings/errors in touched model modules.

Minimum checks for touched Python files:

```bash
ruff check <touched model + workflow/schema files>
```

If formatting hooks apply, rerun checks after formatter changes.

## 7) Required test contract for each model family

Each model family should have tests covering:

1. **Schema/config validation**
   - valid config fixture passes;
   - invalid field/type fails with actionable error;
   - alias normalization behavior (when applicable);
   - default-parity checks for shared fields where model-config defaults must
     remain behavior-compatible (for example covariate defaults).
2. **Factory wiring**
   - factory path routes through schema/runtime adapter;
   - unknown/unsupported fields fail fast (no silent drops);
   - every schema-routed family has at least one dedicated factory-path test.
3. **Core public behavior**
   - required properties/method expectations are satisfied;
   - load/save/predict path assumptions tested where practical.
4. **Regression coverage**
   - at least one fixture-backed real config from `configs/models/<family>/`.
5. **Field semantics consistency**
   - parameters with similar names (for example `learning_rate` vs
     `fine_tune_lr`) are either intentionally distinct and documented, or
     normalized to one canonical field with tests.

## 8) PR acceptance checklist (must pass)

- [ ] Public model interface remains consistent with shared contract.
- [ ] Any required deviations are explicitly documented and justified.
- [ ] Dead/duplicate code removed from touched surfaces.
- [ ] Module/class/method ordering follows the repository standard.
- [ ] Docstrings updated to reflect actual behavior.
- [ ] Touched files pass lint with zero warnings/errors.
- [ ] Required family test contract is present and passing.
- [ ] Config schema artifacts regenerated when schema contracts changed.
- [ ] Model PR description links this checklist and states which gates were
      validated.

## 9) Enforcement strategy

Use this checklist as a gating review rubric for every model-related PR wave.
When practical, convert checklist items into automated tests (contract tests and
schema/fixture validation) so quality does not rely only on manual review.

Current automated contract entrypoint:

```bash
pytest -q tests/models/test_model_family_contract_suite.py
```
