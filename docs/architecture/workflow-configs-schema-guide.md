# Workflow Config Schemas (`src/config/schemas/workflow_configs.py`)

This module defines the schema boundary for forecasting workflow request inputs and
for model feature-selection overrides consumed by training/evaluation.

## Why this module exists

Before this schema lane, workflow inputs were accepted as loose dict/namespace
values and feature overrides were read ad-hoc from model config dictionaries.

`workflow_configs.py` centralizes those contracts so:

- request payload validation is explicit and early,
- invalid values fail with actionable field-level messages,
- feature selection keys are validated consistently across pipeline and evaluation.

## Core classes

### `ForecastingWorkflowRequestSchema`

Purpose:

- validates the request surface used by forecasting workflow entrypoints.

Key fields:

- `model_type`: non-empty model identifier.
- `datasets`: non-empty list of dataset names.
- `config_dir`, `output_dir`: workflow paths.
- `skip_training`, `skip_steps`, `epochs`, `batch_size`.
- `model_config_path`: path to the model YAML override file.

How this connects to CLI:

- The CLI flag is still `--model-config` in the workflow parser.
- `pipeline.py` maps that CLI value into the schema field `model_config_path`
  before validation.
- We keep one schema field name (`model_config_path`) as the canonical internal
  contract.

Validation behavior:

- `skip_steps` must contain only step numbers `1..7`.
- numeric fields (`epochs`, `batch_size`) must be positive when provided.

Usage in repo:

- `src/workflows/forecasting/pipeline.py` (`run_with_args`) validates incoming
  args via `validate_forecasting_workflow_request(...)` before workflow execution.

### `EvaluationFeatureOverrideEnvelope`

Purpose:

- validates only feature-selection keys (`input_features`, `target_features`)
  while permitting other model-config keys to pass through untouched.

Why `extra="allow"`:

- model configs contain many non-feature keys (architecture/training/runtime).
- this schema is intentionally narrow: it asserts feature list types while not
  constraining the entire model-config payload.

Usage in repo:

- consumed by `get_model_feature_override_columns(...)`.
- used by:
  - `src/workflows/forecasting/pipeline.py` (step 5/7 training-column filtering),
  - `src/workflows/forecasting/evaluation.py` (`_generate_forecasts` filtering).

## Helper functions

### `validate_forecasting_workflow_request(payload)`

- validates request payload against `ForecastingWorkflowRequestSchema`.
- raises `ValueError` with field-path diagnostics on invalid input.

### `get_model_feature_override_columns(model_config_overrides)`

- validates feature override keys through `EvaluationFeatureOverrideEnvelope`.
- returns:
  - `None` when no feature override keys are provided,
  - combined feature list `[...input_features, ...target_features]` otherwise.
- raises `ValueError` on invalid types (e.g., string instead of list for
  `input_features`).

## Design pattern in the repository

`workflow_configs.py` follows the same split used by other schema modules:

1. **Schema layer** (`src/config/schemas/*`) validates external/runtime payloads.
2. **Workflow/model runtime layer** (`src/workflows/*`, `src/models/*`) performs
   execution with normalized validated values.

This keeps strict validation at boundaries without relocating model-owned runtime
config classes from `src/models/*/config.py`.

## JSON schema artifact tie-in

The schema classes documented here are also exported as JSON schema artifacts in
[`docs/architecture/generated-config-schemas/`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/docs/architecture/generated-config-schemas)
via:

```bash
python -m src.config.schemas.json_schema_artifacts
```

This keeps human-readable architecture docs and machine-readable schema contracts
in sync during review.
