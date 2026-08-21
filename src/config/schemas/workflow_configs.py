"""Workflow request schemas and validation helpers."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import ConfigDict, Field, ValidationError, field_validator

from .base import BaseConfigSchema


class ForecastingWorkflowRequestSchema(BaseConfigSchema):
    """Validated request contract for forecasting workflow entrypoints."""

    model_type: str = Field(min_length=1)
    datasets: list[str] = Field(min_length=1)
    config_dir: str = Field(default="configs/data/holdout_10pct", min_length=1)
    output_dir: Optional[str] = Field(default=None)
    skip_training: bool = Field(default=False)
    skip_steps: list[int] = Field(default_factory=list)
    epochs: Optional[int] = Field(default=None, gt=0)
    batch_size: Optional[int] = Field(default=None, gt=0)
    model_config_path: Optional[str] = Field(default=None)

    @field_validator("skip_steps")
    @classmethod
    def _validate_skip_steps(cls, steps: list[int]) -> list[int]:
        for step in steps:
            if step < 1 or step > 7:
                raise ValueError(
                    f"skip_steps entries must be between 1 and 7 (got {step})"
                )
        return steps


class EvaluationFeatureOverrideEnvelope(BaseConfigSchema):
    """Envelope schema that validates feature override keys while allowing extras."""

    model_config = ConfigDict(
        extra="allow",
        strict=True,
        populate_by_name=True,
        use_enum_values=True,
        str_strip_whitespace=True,
    )
    input_features: Optional[list[str]] = Field(default=None)
    target_features: Optional[list[str]] = Field(default=None)


def _format_validation_details(exc: ValidationError) -> str:
    details = []
    for err in exc.errors():
        loc = ".".join(str(part) for part in err.get("loc", ()))
        msg = err.get("msg", "validation error")
        details.append(f"  - {loc}: {msg}" if loc else f"  - {msg}")
    return "\n".join(details)


def validate_forecasting_workflow_request(
    payload: dict[str, Any],
) -> ForecastingWorkflowRequestSchema:
    """Validate forecasting workflow request payload."""
    try:
        return ForecastingWorkflowRequestSchema.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(
            "Invalid forecasting workflow request payload:\n"
            f"{_format_validation_details(exc)}"
        ) from exc


def get_model_feature_override_columns(
    model_config_overrides: Optional[dict[str, Any]],
) -> Optional[list[str]]:
    """Validate and extract input/target feature columns from model overrides."""
    if not model_config_overrides:
        return None
    try:
        parsed = EvaluationFeatureOverrideEnvelope.model_validate(
            model_config_overrides
        )
    except ValidationError as exc:
        raise ValueError(
            "Invalid feature override keys in model config payload:\n"
            f"{_format_validation_details(exc)}"
        ) from exc

    input_features = parsed.input_features or []
    target_features = parsed.target_features or []
    feature_columns = list(input_features) + list(target_features)
    return feature_columns or None
