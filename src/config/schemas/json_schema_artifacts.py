"""Generate JSON schema artifacts from active config schema modules."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, NamedTuple

from .base import BaseConfigSchema
from .data_configs import HoldoutConfigSchema
from .model_configs import (
    Chronos2ModelConfigSchema,
    DeepARModelConfigSchema,
    MoiraiModelConfigSchema,
    NaiveBaselineModelConfigSchema,
    PatchTSTModelConfigSchema,
    StatisticalModelConfigSchema,
    SundialModelConfigSchema,
    TFTModelConfigSchema,
    TiDEModelConfigSchema,
    TSMixerModelConfigSchema,
    TTMModelConfigSchema,
)
from .workflow_configs import (
    EvaluationFeatureOverrideEnvelope,
    ForecastingWorkflowRequestSchema,
)


class SchemaArtifactSpec(NamedTuple):
    """One schema artifact export target."""

    filename: str
    schema_type: type[BaseConfigSchema]
    source_symbol: str


SCHEMA_ARTIFACT_SPECS: tuple[SchemaArtifactSpec, ...] = (
    SchemaArtifactSpec(
        filename="model_configs.chronos2.schema.json",
        schema_type=Chronos2ModelConfigSchema,
        source_symbol="src.config.schemas.model_configs.Chronos2ModelConfigSchema",
    ),
    SchemaArtifactSpec(
        filename="model_configs.deepar.schema.json",
        schema_type=DeepARModelConfigSchema,
        source_symbol="src.config.schemas.model_configs.DeepARModelConfigSchema",
    ),
    SchemaArtifactSpec(
        filename="model_configs.moirai.schema.json",
        schema_type=MoiraiModelConfigSchema,
        source_symbol="src.config.schemas.model_configs.MoiraiModelConfigSchema",
    ),
    SchemaArtifactSpec(
        filename="model_configs.naive_baseline.schema.json",
        schema_type=NaiveBaselineModelConfigSchema,
        source_symbol="src.config.schemas.model_configs.NaiveBaselineModelConfigSchema",
    ),
    SchemaArtifactSpec(
        filename="model_configs.patchtst.schema.json",
        schema_type=PatchTSTModelConfigSchema,
        source_symbol="src.config.schemas.model_configs.PatchTSTModelConfigSchema",
    ),
    SchemaArtifactSpec(
        filename="model_configs.statistical.schema.json",
        schema_type=StatisticalModelConfigSchema,
        source_symbol="src.config.schemas.model_configs.StatisticalModelConfigSchema",
    ),
    SchemaArtifactSpec(
        filename="model_configs.sundial.schema.json",
        schema_type=SundialModelConfigSchema,
        source_symbol="src.config.schemas.model_configs.SundialModelConfigSchema",
    ),
    SchemaArtifactSpec(
        filename="model_configs.tft.schema.json",
        schema_type=TFTModelConfigSchema,
        source_symbol="src.config.schemas.model_configs.TFTModelConfigSchema",
    ),
    SchemaArtifactSpec(
        filename="model_configs.tide.schema.json",
        schema_type=TiDEModelConfigSchema,
        source_symbol="src.config.schemas.model_configs.TiDEModelConfigSchema",
    ),
    SchemaArtifactSpec(
        filename="model_configs.ttm.schema.json",
        schema_type=TTMModelConfigSchema,
        source_symbol="src.config.schemas.model_configs.TTMModelConfigSchema",
    ),
    SchemaArtifactSpec(
        filename="model_configs.tsmixer.schema.json",
        schema_type=TSMixerModelConfigSchema,
        source_symbol="src.config.schemas.model_configs.TSMixerModelConfigSchema",
    ),
    SchemaArtifactSpec(
        filename="data_configs.holdout.schema.json",
        schema_type=HoldoutConfigSchema,
        source_symbol="src.config.schemas.data_configs.HoldoutConfigSchema",
    ),
    SchemaArtifactSpec(
        filename="workflow_configs.forecasting_request.schema.json",
        schema_type=ForecastingWorkflowRequestSchema,
        source_symbol="src.config.schemas.workflow_configs.ForecastingWorkflowRequestSchema",
    ),
    SchemaArtifactSpec(
        filename="workflow_configs.feature_override_envelope.schema.json",
        schema_type=EvaluationFeatureOverrideEnvelope,
        source_symbol="src.config.schemas.workflow_configs.EvaluationFeatureOverrideEnvelope",
    ),
)


def get_default_schema_artifact_dir() -> Path:
    """Return canonical docs-visible output directory for generated schemas."""
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "docs" / "architecture" / "generated-config-schemas"


def _serialize_schema(schema_payload: dict[str, Any]) -> str:
    return json.dumps(schema_payload, indent=2, sort_keys=True) + "\n"


def generate_json_schema_artifacts(output_dir: Path | str) -> list[Path]:
    """Write configured JSON schema artifacts and return written paths."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written_paths: list[Path] = []
    manifest_items: list[dict[str, str]] = []

    for spec in SCHEMA_ARTIFACT_SPECS:
        payload = spec.schema_type.model_json_schema()
        out_path = out_dir / spec.filename
        out_path.write_text(_serialize_schema(payload))
        written_paths.append(out_path)
        manifest_items.append(
            {
                "file": spec.filename,
                "source_symbol": spec.source_symbol,
            }
        )

    manifest_payload = {"schemas": manifest_items}
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(_serialize_schema(manifest_payload))
    written_paths.append(manifest_path)
    return written_paths


def main() -> int:
    """CLI entrypoint for schema artifact generation."""
    out_dir = get_default_schema_artifact_dir()
    generated = generate_json_schema_artifacts(out_dir)
    print(f"Wrote {len(generated)} schema artifact files to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
