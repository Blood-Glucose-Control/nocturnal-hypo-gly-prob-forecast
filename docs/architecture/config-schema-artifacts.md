# Config JSON Schema Artifacts

This page explains why we generate JSON schema files from our Pydantic config
schemas, and how contributors should use them.

## Why this exists

Our runtime reads YAML config files for model training/evaluation workflows.
Those YAML files are validated by Pydantic schema classes in
`src/config/schemas/`.

The generated JSON schema artifacts are a **documentation and contract output**
for those same schema classes:

- they show exactly which fields are allowed,
- they show required vs optional fields,
- they show types and bounds (e.g., `gt=0`, enums, list types),
- they make config contract changes visible in PR diffs.

In short: these files let someone understand the config contract without reading
all Python implementation details.

## Why this is useful for an inexperienced developer

If you're new to the repo, this is the fastest way to answer:

- "What keys can I put in this YAML?"
- "Which fields are required?"
- "What value types/ranges are valid?"
- "Did this PR change the config contract?"

Instead of inferring behavior from scattered loader code, you can inspect one
schema artifact and get an explicit answer.

## Why this is a good practice

Generating artifacts from source-of-truth schemas gives us:

1. **Single source of truth**: validation logic and docs stay aligned.
2. **Reviewability**: config contract changes are explicit in code review.
3. **Tool interoperability**: JSON Schema can be consumed by editors, validators,
   and external tooling.
4. **Safer evolution**: schema changes are intentional and easy to audit.

## What this does *not* do

- It does not replace runtime validation.
- It does not validate files by itself.
- It does not imply every model family is fully migrated yet.

It is a generated view of the schema lanes that are currently active.

## Regeneration command

Run from repository root:

```bash
python -m src.config.schemas.json_schema_artifacts
```

This command writes JSON schema files into
[`docs/architecture/generated-config-schemas/`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/docs/architecture/generated-config-schemas):

- `model_configs.chronos2.schema.json`
- `model_configs.deepar.schema.json`
- `model_configs.moirai.schema.json`
- `model_configs.naive_baseline.schema.json`
- `model_configs.patchtst.schema.json`
- `model_configs.statistical.schema.json`
- `model_configs.sundial.schema.json`
- `model_configs.tft.schema.json`
- `model_configs.tide.schema.json`
- `model_configs.ttm.schema.json`
- `model_configs.tsmixer.schema.json`
- `data_configs.holdout.schema.json`
- `workflow_configs.forecasting_request.schema.json`
- `workflow_configs.feature_override_envelope.schema.json`
- `manifest.json`

## When to regenerate

Regenerate and commit these artifacts whenever you change any schema in:

- `src/config/schemas/model_configs.py`
- `src/config/schemas/data_configs.py`
- `src/config/schemas/workflow_configs.py`

If the schema changes but artifacts do not, reviewers lose visibility into the
contract delta.

## Artifact index

- [manifest.json](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/docs/architecture/generated-config-schemas/manifest.json)
- [model_configs.chronos2.schema.json](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/docs/architecture/generated-config-schemas/model_configs.chronos2.schema.json)
- [model_configs.deepar.schema.json](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/docs/architecture/generated-config-schemas/model_configs.deepar.schema.json)
- [model_configs.moirai.schema.json](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/docs/architecture/generated-config-schemas/model_configs.moirai.schema.json)
- [model_configs.naive_baseline.schema.json](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/docs/architecture/generated-config-schemas/model_configs.naive_baseline.schema.json)
- [model_configs.patchtst.schema.json](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/docs/architecture/generated-config-schemas/model_configs.patchtst.schema.json)
- [model_configs.statistical.schema.json](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/docs/architecture/generated-config-schemas/model_configs.statistical.schema.json)
- [model_configs.sundial.schema.json](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/docs/architecture/generated-config-schemas/model_configs.sundial.schema.json)
- [model_configs.tft.schema.json](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/docs/architecture/generated-config-schemas/model_configs.tft.schema.json)
- [model_configs.tide.schema.json](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/docs/architecture/generated-config-schemas/model_configs.tide.schema.json)
- [model_configs.ttm.schema.json](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/docs/architecture/generated-config-schemas/model_configs.ttm.schema.json)
- [model_configs.tsmixer.schema.json](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/docs/architecture/generated-config-schemas/model_configs.tsmixer.schema.json)
- [data_configs.holdout.schema.json](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/docs/architecture/generated-config-schemas/data_configs.holdout.schema.json)
- [workflow_configs.forecasting_request.schema.json](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/docs/architecture/generated-config-schemas/workflow_configs.forecasting_request.schema.json)
- [workflow_configs.feature_override_envelope.schema.json](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/docs/architecture/generated-config-schemas/workflow_configs.feature_override_envelope.schema.json)
