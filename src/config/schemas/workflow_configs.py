"""Workflow request schemas and validation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from pydantic import ConfigDict, Field, ValidationError, ValidationInfo, field_validator

from .base import BaseConfigSchema
from .loader import load_yaml_as_schema


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


class ForecastingSweepTrainJobSchema(BaseConfigSchema):
    """Schema contract for one forecasting-train sweep job entry."""

    model_config_path: str = Field(alias="model_config", min_length=1)
    datasets: list[str] = Field(min_length=1)

    @field_validator("datasets")
    @classmethod
    def _validate_datasets(cls, values: list[str]) -> list[str]:
        for idx, value in enumerate(values):
            if not value:
                raise ValueError(f"datasets[{idx}] must be a non-empty string")
        return values


class ForecastingSweepTrainSpecSchema(BaseConfigSchema):
    """Schema contract for forecasting-train sweep specifications."""

    jobs: list[ForecastingSweepTrainJobSchema] = Field(min_length=1)


class ForecastingSweepEvalJobSchema(BaseConfigSchema):
    """Schema contract for one forecasting-eval sweep job entry."""

    model_config_path: str = Field(alias="model_config", min_length=1)
    context_length: int = Field(gt=0)
    finetuned_datasets: list[str] = Field(default_factory=list)
    zeroshot_datasets: list[str] = Field(default_factory=list)
    covariate_cols: list[str] = Field(default_factory=list)
    probabilistic: Optional[bool] = Field(default=None)
    no_dilate: Optional[bool] = Field(default=None)
    forecast_length: Optional[int] = Field(default=None, gt=0)
    output_dir_template: Optional[str] = Field(default=None, min_length=1)

    @field_validator("finetuned_datasets", "zeroshot_datasets", "covariate_cols")
    @classmethod
    def _validate_string_lists(
        cls, values: list[str], info: ValidationInfo
    ) -> list[str]:
        field_name = info.field_name or "values"
        for idx, value in enumerate(values):
            if not value:
                raise ValueError(f"{field_name}[{idx}] must be a non-empty string")
        return values


class ForecastingSweepEvalSpecSchema(BaseConfigSchema):
    """Schema contract for forecasting-eval sweep specifications."""

    jobs: list[ForecastingSweepEvalJobSchema] = Field(min_length=1)
    probabilistic: bool = Field(default=True)
    no_dilate: bool = Field(default=False)
    forecast_length: int = Field(default=96, gt=0)
    output_dir_template: Optional[str] = Field(default=None, min_length=1)


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


def load_forecasting_train_sweep_spec_from_yaml(
    file_path: str | Path,
) -> ForecastingSweepTrainSpecSchema:
    """Load and validate a forecasting-train sweep spec YAML file."""
    return load_yaml_as_schema(file_path, ForecastingSweepTrainSpecSchema)


def load_forecasting_eval_sweep_spec_from_yaml(
    file_path: str | Path,
) -> ForecastingSweepEvalSpecSchema:
    """Load and validate a forecasting-eval sweep spec YAML file."""
    return load_yaml_as_schema(file_path, ForecastingSweepEvalSpecSchema)


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
