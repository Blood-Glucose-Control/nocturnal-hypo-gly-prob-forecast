# P1-38A Method Contract Consolidation Task

## Objective

Complete cross-model method-contract consolidation derived from the WS1
presence matrix in
[P1-38_MODEL_RUNTIME_CONSOLIDATION_PLAN.md](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/agent_notes/P1-38_MODEL_RUNTIME_CONSOLIDATION_PLAN.md).
Current matrix baseline covers 146 distinct methods/functions.

This task is now a required closeout gate for P1-38.

## Scope

### In scope

- Repeated helper/method surfaces across:
  - [src/models/ttm/model.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/ttm/model.py)
  - [src/models/timesfm/model.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/timesfm/model.py)
  - [src/models/moirai/model.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/moirai/model.py)
  - [src/models/moment/model.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/moment/model.py)
  - [src/models/chronos2/model.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/chronos2/model.py)
  - [src/models/toto/model.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/toto/model.py)
  - [src/models/tide/model.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/tide/model.py)
- Promotion candidates to shared surfaces:
  - [src/models/base/base_model.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/base/base_model.py)
  - [src/models/autogluon_data_utils.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/autogluon_data_utils.py)
  - [src/models/autogluon_base.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/autogluon_base.py)

### Out of scope

- Architecture redesign per model family.
- Metric definition changes.
- Dataset semantics changes.

## Work packages

1. **MC1: lifecycle + checkpoint consolidation**
   - Prioritize `_initialize_model`, `_save_checkpoint`, `_load_checkpoint`,
     preprocessor/checkpoint-path helpers.
2. **MC2: training data/trainer helper consolidation**
   - Prioritize repeated training-input normalization, trainer-argument
     assembly, callback/trainer factories.
3. **MC3: batch/quantile inference consolidation**
   - Prioritize repeated batch collectors, quantile extraction/validation, and
     zero-shot batch helper variants.
4. **MC4: AutoGluon context/covariate consolidation**
   - Prioritize known-covariate/context shaping and predictor extraction
     helpers across Chronos2/TiDE.

## Acceptance criteria

1. Matrix candidate rows are converted into concrete combine/promote actions.
2. Duplicate method logic is reduced through shared helpers or base-method
   promotion while preserving behavior.
3. Contract stability is maintained (`predict`, `predict_batch`, checkpoint IO,
   schema-routed config construction).
4. P1-38 remains open until MC1-MC4 are complete and validated.

## Validation gates

- `pytest -q tests/models/test_model_family_contract_suite.py`
- `pytest -q tests/workflows/forecasting/test_model_config_schema_loader.py -k "ttm or timesfm or moirai or moment or chronos2 or toto or tide"`
- Family-targeted regression suites for touched slices.
- `SKIP=pyright pre-commit run --files <touched_python_files>`
- Final Pylance diagnostics clean for touched Python files.
