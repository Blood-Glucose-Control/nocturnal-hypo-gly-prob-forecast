# P1-21 Model Runtime Flow Audit (Keep / Wire / Remove Matrix)

**Date:** 2026-08-13
**Update Date:** 2026-08-23
**Task:** `model-runtime-flow-audit`
**Status:** Completed and reassessed against post-P1-15 runtime state
**Scope:** [`src/models/base/`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/base), [`src/models/factory.py`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/factory.py), [`src/workflows/forecasting/modeling.py`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/workflows/forecasting/modeling.py), [`src/workflows/evaluation/`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/workflows/evaluation)

---

## 1) Runtime tracing method (reassessed)

Caller tracing was refreshed from current runtime entrypoints:

- Primary evaluation runtime:
  - [`nocturnal_hypo_eval.py`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/workflows/evaluation/nocturnal_hypo_eval.py)
  - [`sliding_window_eval.py`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/workflows/evaluation/sliding_window_eval.py)
- Secondary evaluation/runtime check:
  - [`validate_predict_batch.py`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/workflows/evaluation/validate_predict_batch.py)
- Personalization runtime:
  - [`per_patient_finetune.py`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/workflows/personalization/per_patient_finetune.py)
- Forecasting pipeline lane:
  - [`pipeline.py`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/workflows/forecasting/pipeline.py)

Observed constructor reality:

- Primary/secondary eval + personalization flows still instantiate models through [`create_model_and_config`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/factory.py:35).
- YAML config validation in those flows is schema-routed via [`load_model_config_from_yaml`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/workflows/forecasting/modeling.py:50).
- Forecasting pipeline uses [`ModelFactory.create_model`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/workflows/forecasting/modeling.py:124) / [`ModelFactory.load_model`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/workflows/forecasting/modeling.py:601).
- [`ModelRegistry`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/base/registry.py) remains registration/discovery + tests, not the direct primary runtime constructor.

---

## 2) Current constructor graph (what should be treated as main runtime)

### Primary runtime constructor graph (today)

`nocturnal_hypo_eval` / `sliding_window_eval`
→ schema-validated YAML load via [`load_model_config_from_yaml`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/workflows/forecasting/modeling.py:50)
→ model construction via [`create_model_and_config`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/factory.py:35)
→ concrete `*Forecaster` lifecycle through [`BaseTimeSeriesFoundationModel`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/base/base_model.py).

### Parallel runtime lane (already schema-routed)

`forecasting/pipeline.py`
→ [`ModelFactory.create_model`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/workflows/forecasting/modeling.py:124) / [`ModelFactory.load_model`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/workflows/forecasting/modeling.py:601)
→ model-family schema adapters in [`MODEL_CONFIG_ROUTES`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/config/schemas/model_configs.py:830).

### Registry role (explicit)

[`ModelRegistry`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/base/registry.py) should be treated as:
- class registration/discovery and contract-test support,
- **not** an independent third constructor path for entrypoint scripts.

---

## 3) Keep / Wire / Remove matrix (2026-08-23 reassessment)

Legend:
- **KEEP** = active runtime-critical surface now.
- **WIRE** = keep, but converge/clarify behavior in the next phase.
- **REMOVE** = not runtime-critical; remove or keep demoted/internal.

| Surface | Runtime evidence | Class | Action |
|---|---|---|---|
| [`create_model_and_config`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/factory.py:35) | Directly used by primary/secondary evaluation and personalization entrypoints | Active constructor | **KEEP (current)**; **WIRE** to become a thin wrapper over schema-routed constructor logic |
| [`ModelFactory.create_model/load_model`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/workflows/forecasting/modeling.py:124) | Active in forecasting pipeline with full schema-route coverage | Canonical target lane | **KEEP** and treat as unification target |
| [`load_model_config_from_yaml`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/workflows/forecasting/modeling.py:50) | Used in primary/secondary eval flows | Config validation boundary | **KEEP** |
| [`ModelRegistry`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/base/registry.py:26) | Decorator registration + tests; no direct entrypoint constructor calls | Registry/discovery only | **KEEP** with explicit role boundary |
| [`create_model_from_config`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/base/base_model.py:773) | Narrow helper; not core entrypoint path | Legacy helper | **REMOVE/DEPRECATE** from contributor-facing guidance |
| Base-level `LoRAConfig` / `DistributedConfig` surfaces | Removed from base runtime surface under `model-runtime-surface-prune` | Previously wired scaffolding | **REMOVE (completed)** |
| Chronos-2 stage-1 LoRA merge helper ([`merge_stage1_lora`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/workflows/personalization/per_patient_finetune.py:287)) | Narrow personalization-only helper | Specialized path | **WIRE (narrow scope only)**; do not promote as general constructor surface |

---

## 4) Direct answers from this reassessment

1. **Should LoRAConfig/DistributedConfig stay in core runtime?**
   No. Base-level LoRA/Distributed config surfaces were already removed from the core runtime surface and should stay removed unless a fully tested runtime need returns.

2. **What should be the main runtime constructor graph?**
   Today, primary evaluation still runs through [`create_model_and_config`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/factory.py:35).
   Next-phase target should converge constructor behavior onto schema-routed [`ModelFactory`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/workflows/forecasting/modeling.py:99) semantics, with `create_model_and_config` reduced to a compatibility wrapper.

3. **Should minor experimental entrypoints drive architecture decisions?**
   No. Priority should remain:
   - primary: [`nocturnal_hypo_eval.py`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/workflows/evaluation/nocturnal_hypo_eval.py),
   - secondary: [`sliding_window_eval.py`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/workflows/evaluation/sliding_window_eval.py),
   - then other specialty scripts.

---

## 5) Handoff to next items

- `model-extension-contract-doc`: completed in this update pass via [`docs/base_model_framework_README.md`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/docs/base_model_framework_README.md).
- `model-runtime-consolidation-wave`: remove duplicated per-family constructor logic while preserving runtime behavior.
- Keep primary runtime compatibility stable during constructor unification.

---

## 6) Wrap-up complete summary

This update pass refreshed P1-21 against the current codebase and resolved stale assumptions from pre-reorg docs:
- updated runtime evidence to current [`src/workflows/`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/workflows) entrypoints,
- confirmed base LoRA/Distributed removal remains correct,
- clarified the current-vs-target constructor graph (`create_model_and_config` now, schema-routed `ModelFactory` target),
- closed `model-extension-contract-doc` and handed remaining follow-on scope to `model-runtime-consolidation-wave`.
