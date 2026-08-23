# P1-14 Model Stack Deep Dive (Base/Factory/AutoGluon)

**Date:** 2026-08-13
**Update Date:** 2026-08-23
**Scope:** `src/models/base/`, `src/models/factory.py`, `src/models/autogluon_base.py`
**Why this exists:** confirm architecture quality before expanding adapter/protocol work and set a runtime-first pruning policy.
**Status:** Historical deep-dive completed; superseded by refreshed runtime evidence in `P1-21`.

---

## 1) Current architecture flow (what actually happens today)

1. Experiment/training scripts call [`create_model_and_config`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/factory.py) to instantiate a model.
2. [`create_model_and_config`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/factory.py) uses a large `if/elif` tree for per-model checkpoint/config logic.
3. Concrete model classes mostly inherit [`BaseTimeSeriesFoundationModel`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/base/base_model.py) directly; AutoGluon families inherit [`AutoGluonBaseModel`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/autogluon_base.py).
4. Base class then drives unified `fit/predict/predict_batch/save/load/get_model_info` lifecycle with guardrails (fitted-state checks, quantile gating, metadata save/load).

---

## 2) What is already good and should be kept

- Base lifecycle contract in [`base_model.py`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/base/base_model.py) is real and valuable; most model classes already conform structurally.
- AutoGluon deduplication in [`autogluon_base.py`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/autogluon_base.py) is the right direction (shared data prep, predictor fit/load, batch prediction).
- [`ModelRegistry`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/base/registry.py) exists and is test-backed; good candidate for cleaner extension path.

---

## 3) High-confidence issues found

### A) Two competing model-construction paths

- Runtime scripts mostly use [`create_model_and_config`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/factory.py).
- [`ModelRegistry`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/base/registry.py) is used mostly in tests, not as the primary runtime constructor.
- Result: duplication and onboarding confusion ("do I add to registry, factory, or both?").

### B) Large factory with repeated logic

- [`factory.py`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/factory.py) repeats similar checkpoint-load + override validation blocks many times.
- This increases regression risk when adding new models or changing override rules.

### C) Base package exports utilities that are effectively unused in runtime

Repository search shows these are currently not part of core model execution flows:

- [`src/models/base/lora_utils.py`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/base/lora_utils.py) (self-contained LoRA wrapper stack; not used by runtime model flows, which use PEFT path in base model).
- [`DistributedManager`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/base/distributed.py) + helper config functions are mostly example-surface, while actual training models rely on methods embedded in [`BaseTimeSeriesFoundationModel`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/base/base_model.py).
- [`create_model_from_config`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/base/base_model.py) is legacy-narrow (limited model coverage; currently used mainly by a targeted test).
- Keeping these in the default runtime export surface creates contributor and coding-agent confusion because they look canonical but are not operationally exercised.

### D) Adapter protocol work is currently additive, not integrated

- [`src/models/adapter.py`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/adapter.py) and [`src/models/naive_baseline/adapter.py`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/naive_baseline/adapter.py) validate a shape, but are not yet the canonical entrypoint used by experiment scripts.
- So your intuition is right: this was a first contract slice, not a full architecture cleanup.

---

## 4) Decision: what to do now vs later

### Do now (P1-14 closeout scope)

1. Establish one contributor-facing model extension path:
   - runtime constructor source of truth,
   - required config/load/override behavior,
   - required tests for new models.
2. Reduce duplicated constructor logic (shared helpers for checkpoint loading + validated overrides).
3. Apply runtime-first pruning:
   - default action for non-runtime surfaces is **remove/un-export**,
   - keep only if there is an explicit near-term wiring plan, owner, and tests.
4. Reclassify retained "not-yet-runtime" utilities into explicit internal/example locations so they are not mistaken for stable runtime API.

### Defer slightly (after above foundations, before large ecosystem churn)

5. Broader module/package reshaping of `src/models/base/` internals can be phased after constructor unification and first Pydantic slice to avoid unstable simultaneous rewrites.

---

## 5) Proposed completion criteria for `model-adapter-protocol`

Task is only `done` when all are true:

1. `create_model_and_config` path is unified/cleaned so contributor steps are unambiguous.
2. Registry/factory responsibilities are explicit and non-duplicative.
3. Legacy/unused base exports are removed from the default public API unless explicitly wired and tested.
4. Adapter contract is exercised by real runtime path(s), not just type/tests.
5. Contributor documentation includes "how to add a new model" with one canonical checklist.

---

## 6) Recommendation

Do **not** do a giant rewrite in one PR.
Do a staged cleanup that keeps behavior stable while reducing architectural confusion and dead surface area.

---

## 7) Tracking tasks added from this decision

The following P1 tasks were added in [`project_tracking.csv`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/project_tracking.csv) to execute this policy:

- `model-runtime-flow-audit` ✅ (artifact: [`P1-21_MODEL_RUNTIME_FLOW_AUDIT.md`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/agent_notes/P1-21_MODEL_RUNTIME_FLOW_AUDIT.md))
- `model-runtime-surface-prune`
- `model-extension-contract-doc`

---

## 8) Wrap-up complete summary

This deep-dive established the runtime-first pruning policy and directly spawned
the P1-21 audit/prune workstream. As of 2026-08-23, the
`model-extension-contract-doc` handoff has been completed (published at
[`docs/base_model_framework_README.md`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/docs/base_model_framework_README.md)).
Remaining follow-on work from this lineage is constructor-logic
dedup/unification under `model-runtime-consolidation-wave`.
