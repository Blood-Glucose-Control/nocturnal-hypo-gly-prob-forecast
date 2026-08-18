# P1-21 Model Runtime Flow Audit (Keep / Wire / Remove Matrix)

**Date:** 2026-08-13
**Task:** `model-runtime-flow-audit`
**Scope:** [`src/models/base/`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/base), [`src/models/factory.py`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/factory.py), [`src/models/autogluon_base.py`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/autogluon_base.py)

---

## 1) Runtime tracing method

Runtime caller tracing was done from training/eval entrypoints under [`scripts/experiments/`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/scripts/experiments), plus related orchestration/eval scripts that instantiate models:

- [`nocturnal_hypo_eval.py`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/scripts/experiments/nocturnal_hypo_eval.py)
- [`nocturnal_hypo_eval_ctx_ablation.py`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/scripts/experiments/nocturnal_hypo_eval_ctx_ablation.py)
- [`sliding_window_eval.py`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/scripts/experiments/sliding_window_eval.py)
- [`per_patient_finetune.py`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/scripts/experiments/per_patient_finetune.py)
- [`validate_predict_batch.py`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/scripts/experiments/validate_predict_batch.py)
- [`export_single_episode_eval_data.py`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/scripts/data_processing_scripts/export_single_episode_eval_data.py)
- [`compare_forecasts.py`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/scripts/visualization/compare_forecasts.py)

Observed constructor reality:

- Runtime scripts call [`create_model_and_config`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/factory.py:17).
- No runtime scripts call [`create_model_from_config`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/base/base_model.py:1127).
- [`ModelRegistry`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/base/registry.py) is used for model class registration and tests, but is not the main runtime constructor path.

---

## 2) Keep / Wire / Remove matrix

Legend:
- **KEEP** = core runtime surface currently in real execution flow.
- **WIRE** = not primary runtime today, but keep only with explicit near-term wiring plan/owner/tests.
- **REMOVE** = not in core runtime flow; remove or un-export by default.

### A) `src/models/base` surface

| Surface | Runtime evidence | Observed runtime call path | Class | Action |
|---|---|---|---|---|
| [`BaseTimeSeriesFoundationModel`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/base/base_model.py:235) | Base class of all active forecasters | entrypoint script → [`create_model_and_config`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/factory.py:17) → concrete `*Forecaster(config)` → [`BaseTimeSeriesFoundationModel.__init__`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/base/base_model.py:258) → [`fit/predict/predict_batch`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/base/base_model.py:605) | KEEP | Retain as canonical lifecycle base |
| [`ModelConfig`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/base/base_model.py:38) | Used by all model config classes and factory typing | entrypoint script → [`create_model_and_config`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/factory.py:17) branch builds config dataclass → model constructor stores config in base [`__init__`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/base/base_model.py:272) | KEEP | Retain |
| [`LoRAConfig`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/base/base_model.py:163) | Conditional path exists, but current core `per_patient_finetune` model choices do not activate it | [`per_patient_finetune.py`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/scripts/experiments/per_patient_finetune.py:521) sets `LoRAConfig` only if `model.supports_lora`; current choices (`ttm`, `sundial`, `chronos2`, `timesfm`) all return `False` | WIRE | Do not treat as active core runtime; either verify model-level LoRA support end-to-end or remove from core script path |
| [`DistributedConfig`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/base/base_model.py:195) | Present in runtime objects; no experiment entrypoint configures it | model construction → base [`__init__`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/base/base_model.py:274) creates default `DistributedConfig()` → [`fit`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/base/base_model.py:605) → [`_setup_distributed`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/base/base_model.py:822) returns early when `enabled=False` | WIRE | Keep only if canonical scripts will expose/configure it; otherwise demote from core API surface |
| [`TrainingBackend`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/base/base_model.py:29) | Used at runtime for backend reporting/metadata shape (not central dispatcher) | model-specific property (e.g., [`TTMForecaster.training_backend`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/ttm/model.py:118), [`MomentForecaster.training_backend`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/moment/model.py:116)) → base [`fit`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/base/base_model.py:640) logs backend value | KEEP | Retain; clarify that routing is model-local, not base-dispatched |
| [`ModelRegistry`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/base/registry.py:26) | Registration works; not primary constructor for runtime scripts | model module import (e.g., [`chronos2/model.py`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/chronos2/model.py:38)) executes decorator registration; runtime scripts still construct via factory | WIRE | Keep temporarily; clarify role vs factory in contract doc |
| [`create_model_from_config`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/base/base_model.py:1127) | No runtime entrypoint callers | no call path from `scripts/experiments/*` | REMOVE | Remove or move to internal legacy helper; update narrow test |
| [`DistributedManager`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/base/distributed.py:17) | No runtime entrypoint usage; mostly example-facing | no call path from `scripts/experiments/*` (example-only references) | REMOVE | Un-export from runtime API; keep only if actively wired later |
| [`setup_deepspeed_config`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/base/distributed.py:125) | No runtime entrypoint usage | no call path from `scripts/experiments/*` | REMOVE | Un-export/remove from runtime surface |
| [`setup_fsdp_config`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/base/distributed.py:204) | No runtime entrypoint usage | no call path from `scripts/experiments/*` | REMOVE | Un-export/remove from runtime surface |
| [`distributed_manager`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/base/distributed.py:320) | No runtime entrypoint usage | no call path from `scripts/experiments/*` | REMOVE | Un-export/remove from runtime surface |
| [`GPUManager`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/base/distributed.py:235) | Used in examples only, not core training/eval runtime | example scripts (e.g., [`show_hardware_info.py`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/scripts/examples/show_hardware_info.py:43)); no path from `scripts/experiments/*` | REMOVE | Move to examples/internal utility surface |
| [`lora_utils` exported symbols in `__init__`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/base/__init__.py:27) | No runtime entrypoint usage (self-contained utility stack) | no call path from `scripts/experiments/*`; runtime LoRA path goes through base [`_enable_lora`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/base/base_model.py:934) | REMOVE | Un-export now; keep file only if explicit owner/wiring appears |

### B) `src/models/factory.py` surface

| Surface | Runtime evidence | Observed runtime call path | Class | Action |
|---|---|---|---|---|
| [`create_model_and_config`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/factory.py:17) | Primary runtime constructor across experiment/eval scripts | [`nocturnal_hypo_eval.py`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/scripts/experiments/nocturnal_hypo_eval.py:286), [`sliding_window_eval.py`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/scripts/experiments/sliding_window_eval.py:550), [`per_patient_finetune.py`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/scripts/experiments/per_patient_finetune.py:455), etc. → `create_model_and_config` | KEEP | Retain as current canonical path |
| Per-model `if/elif` constructor branches | Actively used but duplicated override/checkpoint logic | entrypoint script → `create_model_and_config` → branch-specific checkpoint/config normalization → concrete model constructor | WIRE | Refactor into shared helpers to reduce duplication/risk |
| Mixed constructor responsibilities with registry | Causes extension confusion | runtime scripts call factory while registry is populated via decorators at import time | WIRE | Resolve in `model-extension-contract-doc` |

### C) `src/models/autogluon_base.py` surface

| Surface | Runtime evidence | Observed runtime call path | Class | Action |
|---|---|---|---|---|
| [`AutoGluonBaseModel`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/autogluon_base.py:44) | Active base for [`NaiveBaselineForecaster`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/naive_baseline/model.py:32), [`StatisticalForecaster`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/statistical/model.py:37), [`DeepARForecaster`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/deepar/model.py:30), [`PatchTSTForecaster`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/patchtst/model.py:27), [`TFTForecaster`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/tft/model.py:32) | entrypoint script → [`create_model_and_config`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/factory.py:17) → AG-backed forecaster instance → base [`fit`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/base/base_model.py:605) dispatches to AG base [`_train_model`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/autogluon_base.py:160) | KEEP | Retain and treat as core runtime |
| Shared AG data prep / fit / predict / checkpoint methods | Called through all AG-backed models in runtime paths | `fit` path: base [`fit`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/base/base_model.py:605) → AG base [`_prepare_training_data`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/autogluon_base.py:110) + [`_train_model`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/autogluon_base.py:160); inference path: AG base [`_predict`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/autogluon_base.py:299) / [`_predict_batch`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/autogluon_base.py:323) | KEEP | Retain and regression-test when pruning nearby surfaces |

---

## 3) Outcome

`model-runtime-flow-audit` conclusion:

1. Runtime constructor source of truth is currently [`create_model_and_config`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/factory.py).
2. Non-runtime base exports should be pruned by default (un-export/remove).
3. `ModelRegistry` remains only with explicit documented role while constructor unification is finalized.
4. [`AutoGluonBaseModel`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/autogluon_base.py) is confirmed core runtime and should not be destabilized during prune.

---

## 4) Hand-off to next P1 tasks

- Next: `model-extension-contract-doc` formalizes one contributor path and resolves factory/registry ambiguity.

---

## 5) Clarification for LoRA/Distributed/TrainingBackend in core experiments

- **LoRAConfig (base-class PEFT path):** currently **not active** in core experiment execution.
  In [`per_patient_finetune.py`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/scripts/experiments/per_patient_finetune.py), LoRA config is gated behind `model.supports_lora`, but allowed models are `ttm`, `sundial`, `chronos2`, `timesfm`, which currently return `False` in their model classes.
  The Chronos-2 "LoRA merge" in the same script ([`merge_stage1_lora`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/scripts/experiments/per_patient_finetune.py:291)) is a separate AutoGluon/PEFT artifact merge path, not the base `LoRAConfig` mechanism.

- **DistributedConfig:** currently **no-op** in core experiment entrypoints.
  No `scripts/experiments/*` script constructs/passes `DistributedConfig`; base fit still calls [`_setup_distributed`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/base/base_model.py:822), which exits immediately under default `enabled=False`.

- **TrainingBackend:** currently **informational** in core runtime.
  It is used for reporting/metadata (e.g., base fit log at [`base_model.py:640`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/base/base_model.py:640)) rather than as a central execution switch in base-class control flow.

---

## 6) Execution update (2026-08-13): prune completed

`model-runtime-surface-prune` has now been executed:

- Removed runtime LoRA/Distributed surfaces from [`src/models/base/base_model.py`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/base/base_model.py) (config classes, setup hooks, metadata fields).
- Removed LoRA/Distributed constructor plumbing and `supports_lora` properties across active model classes in [`src/models/`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models).
- Removed stage-2 LoRA branch from [`per_patient_finetune.py`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/scripts/experiments/per_patient_finetune.py).
- Updated [`example_holdout_generic_workflow.py`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/scripts/examples/example_holdout_generic_workflow.py) to stop passing removed distributed constructor arguments.

Validation completed:

- Holdout workflow smoke runs succeeded for `chronos2`, `patchtst`, `timesfm` on `aleppo_2017`.
- Nocturnal evaluation artifacts were produced under:
  - [`experiments/nocturnal_forecasting/512ctx_96fh/chronos2/prune-smoke`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/experiments/nocturnal_forecasting/512ctx_96fh/chronos2/prune-smoke)
  - [`experiments/nocturnal_forecasting/512ctx_96fh/patchtst/prune-smoke`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/experiments/nocturnal_forecasting/512ctx_96fh/patchtst/prune-smoke)
  - [`experiments/nocturnal_forecasting/512ctx_96fh/timesfm/prune-smoke`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/experiments/nocturnal_forecasting/512ctx_96fh/timesfm/prune-smoke)
