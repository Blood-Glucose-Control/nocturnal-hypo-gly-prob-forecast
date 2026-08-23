# Base Time-Series Model Framework

This repository uses a shared model framework so all forecasting families follow
the same lifecycle contracts (config → initialize → fit → predict → save/load),
while still owning family-specific implementation details.

## Core framework components

- [`BaseTimeSeriesFoundationModel`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/base/base_model.py)
  defines the common lifecycle APIs and persistence behavior.
- [`ModelConfig`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/base/base_model.py)
  defines shared training/runtime fields used by model families.
- [`TrainingBackend`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/base/base_model.py)
  captures backend style (`transformers`, `pytorch`, `custom`).
- [`ModelRegistry`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/base/registry.py)
  provides family registration/discovery.
- [`ModelFactory`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/workflows/forecasting/modeling.py)
  wires workflow config loading to model-family constructors.

## Configuration and schema validation

Workflow model YAMLs are validated by strict Pydantic schemas before runtime:

- schema definitions: [`src/config/schemas/model_configs.py`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/config/schemas/model_configs.py)
- workflow loader path: [`load_model_config_from_yaml`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/workflows/forecasting/modeling.py:50)
- generated schema artifacts: [`docs/architecture/generated-config-schemas/`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/docs/architecture/generated-config-schemas)

This keeps config contracts explicit and prevents drift between docs and runtime.

## Adding a new model family

1. Implement family config/model classes in `src/models/<family>/`.
2. Ensure the model class binds `config_class = <FamilyConfig>` so base load paths deserialize correctly.
3. Register the model with [`ModelRegistry`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/models/base/registry.py).
4. Add schema + runtime adapter entry in [`MODEL_CONFIG_ROUTES`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/config/schemas/model_configs.py).
5. Add focused schema/factory tests in
   [`tests/workflows/forecasting/test_model_config_schema_loader.py`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/tests/workflows/forecasting/test_model_config_schema_loader.py).

## Validation commands

Use targeted checks for touched files/families:

```bash
ruff check src/config/schemas src/workflows/forecasting src/models/<family>
pytest -q tests/workflows/forecasting/test_model_config_schema_loader.py
pytest -q tests/models/test_model_family_contract_suite.py
```
