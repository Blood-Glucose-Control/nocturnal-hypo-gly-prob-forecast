# Base Model Framework and Extension Contract

**Date:** 2026-08-13
**Update Date:** 2026-08-23
**Task ID:** `model-extension-contract-doc`
**Status:** Active contributor contract (published)

This document is the canonical contract for adding or modifying model families.

---

## 1) Runtime constructor graph (authoritative)

### Primary evaluation runtime (most important)

[`src/workflows/evaluation/nocturnal_hypo_eval.py`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/workflows/evaluation/nocturnal_hypo_eval.py)
[`src/workflows/evaluation/sliding_window_eval.py`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/workflows/evaluation/sliding_window_eval.py)

Current flow:
1. YAML overrides load through [`load_model_config_from_yaml`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/workflows/forecasting/modeling.py:50).
2. Model creation goes through [`create_model_and_config`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/factory.py:35).
3. Concrete model classes run lifecycle via [`BaseTimeSeriesFoundationModel`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/base/base_model.py).

### Forecasting pipeline runtime

[`src/workflows/forecasting/pipeline.py`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/workflows/forecasting/pipeline.py)

Flow:
1. Build workflow config wrapper (`GenericModelConfig`).
2. Construct/load via [`ModelFactory.create_model`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/workflows/forecasting/modeling.py:124) and [`ModelFactory.load_model`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/workflows/forecasting/modeling.py:601).
3. Route all model runtime payloads through schema adapters in [`MODEL_CONFIG_ROUTES`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/config/schemas/model_configs.py:830).

### Registry role

[`ModelRegistry`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/base/registry.py) is for registration/discovery and tests.
It is not a separate user-facing constructor path for runtime entrypoints.

---

## 2) Required implementation checklist for a new model family

1. Add family module(s) under [`src/models/`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models).
2. Ensure forecaster class sets `config_class = <FamilyConfig>`.
3. Register forecaster class with [`ModelRegistry.register`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/base/registry.py:32).
4. Add schema + runtime adapter route in [`src/config/schemas/model_configs.py`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/config/schemas/model_configs.py).
5. Wire constructor handling:
   - current required path: [`create_model_and_config`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/factory.py:35),
   - schema-routed workflow path: [`ModelFactory`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/workflows/forecasting/modeling.py:99).
6. Add/extend tests:
   - [`tests/workflows/forecasting/test_model_config_schema_loader.py`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/tests/workflows/forecasting/test_model_config_schema_loader.py)
   - [`tests/models/test_model_family_contract_suite.py`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/tests/models/test_model_family_contract_suite.py)
   - family-specific tests in [`tests/models/`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/tests/models).

---

## 3) Keep / wire policy

- Keep runtime-core surfaces that are exercised by primary entrypoints.
- Do not reintroduce removed base-level LoRA/Distributed scaffolding unless there is a concrete runtime owner and regression coverage.
- Treat specialty/experimental scripts as non-authoritative unless promoted explicitly into runtime-core workflows.

---

## 4) Validation gate for model/runtime changes

Run focused checks for touched families/surfaces:

```bash
ruff check src/config/schemas src/workflows/forecasting src/workflows/evaluation src/models/<family>
pytest -q tests/workflows/forecasting/test_model_config_schema_loader.py
pytest -q tests/models/test_model_family_contract_suite.py
```

Then run final Pylance diagnostics on touched files from canonical workspace root:
`/data/home/<you>/nocturnal-hypo-gly-prob-forecast`.

---

## 5) Wrap-up complete summary

This contract resolves the open `model-extension-contract-doc` handoff by
documenting one contributor-facing model extension path, explicit constructor-role
boundaries (factory vs registry), and mandatory validation surfaces.

Follow-on work passed to other tasks:
- constructor dedup/unification: `model-runtime-consolidation-wave`,
- broader runtime boundary governance: `runtime-core-boundary-lock`.
