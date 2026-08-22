# Config JSON Schema Artifacts

Pydantic schema artifacts for active config lanes are generated into
[`docs/architecture/generated-config-schemas/`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/docs/architecture/generated-config-schemas).

## Regeneration command

Run from repository root:

```bash
python -m src.config.schemas.json_schema_artifacts
```

This writes:

- `model_configs.tsmixer.schema.json`
- `data_configs.holdout.schema.json`
- `workflow_configs.forecasting_request.schema.json`
- `workflow_configs.feature_override_envelope.schema.json`
- `manifest.json`

## Artifact index

- [manifest.json](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/docs/architecture/generated-config-schemas/manifest.json)
- [model_configs.tsmixer.schema.json](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/docs/architecture/generated-config-schemas/model_configs.tsmixer.schema.json)
- [data_configs.holdout.schema.json](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/docs/architecture/generated-config-schemas/data_configs.holdout.schema.json)
- [workflow_configs.forecasting_request.schema.json](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/docs/architecture/generated-config-schemas/workflow_configs.forecasting_request.schema.json)
- [workflow_configs.feature_override_envelope.schema.json](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/docs/architecture/generated-config-schemas/workflow_configs.feature_override_envelope.schema.json)
