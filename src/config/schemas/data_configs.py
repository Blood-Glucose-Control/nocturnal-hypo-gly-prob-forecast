"""Data/holdout config schemas and runtime adapters."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field, ValidationError, model_validator

from .base import BaseConfigSchema
from .loader import load_yaml_as_schema


class TemporalHoldoutConfigSchema(BaseConfigSchema):
    """Schema contract for temporal holdout config section."""

    holdout_percentage: float = Field(gt=0.0, lt=1.0)
    min_train_samples: int = Field(gt=0)
    min_holdout_samples: int = Field(gt=0)


class PatientHoldoutConfigSchema(BaseConfigSchema):
    """Schema contract for patient holdout config section."""

    holdout_patients: list[str] = Field(default_factory=list)
    holdout_percentage: Optional[float] = Field(default=None, gt=0.0, lt=1.0)
    min_train_patients: int = Field(gt=0)
    min_holdout_patients: int = Field(gt=0)
    random_seed: int = Field(default=42)


class HoldoutConfigSchema(BaseConfigSchema):
    """Schema contract for full holdout configuration YAML."""

    dataset_name: str = Field(min_length=1)
    holdout_type: Literal["temporal", "patient_based", "hybrid"]
    temporal_config: Optional[TemporalHoldoutConfigSchema] = Field(default=None)
    patient_config: Optional[PatientHoldoutConfigSchema] = Field(default=None)
    description: str = Field(default="")
    created_date: Optional[str] = Field(default=None)
    version: str = Field(default="1.0")

    @model_validator(mode="after")
    def _validate_holdout_type_requirements(self) -> "HoldoutConfigSchema":
        if self.holdout_type == "temporal" and self.temporal_config is None:
            raise ValueError("temporal_config required for holdout_type=temporal")
        if self.holdout_type == "patient_based" and self.patient_config is None:
            raise ValueError("patient_config required for holdout_type=patient_based")
        if self.holdout_type == "hybrid":
            if self.temporal_config is None or self.patient_config is None:
                raise ValueError(
                    "Both temporal_config and patient_config required for holdout_type=hybrid"
                )
        return self


def _format_validation_details(exc: ValidationError) -> str:
    details = []
    for err in exc.errors():
        loc = ".".join(str(part) for part in err.get("loc", ()))
        msg = err.get("msg", "validation error")
        details.append(f"  - {loc}: {msg}" if loc else f"  - {msg}")
    return "\n".join(details)


def _to_holdout_config(validated_payload: dict):
    holdout_module = import_module("src.data.versioning.holdout_config")
    holdout_cls = getattr(holdout_module, "HoldoutConfig")
    return holdout_cls.from_dict(validated_payload)


def build_holdout_runtime_config(config_data: dict):
    """Validate and adapt mapping payload to HoldoutConfig runtime object."""
    try:
        schema = HoldoutConfigSchema.model_validate(config_data)
    except ValidationError as exc:
        raise ValueError(
            "Invalid holdout config payload for HoldoutConfigSchema:\n"
            f"{_format_validation_details(exc)}"
        ) from exc

    return _to_holdout_config(schema.model_dump(exclude_none=True))


def load_holdout_runtime_config_from_yaml(config_path: str | Path):
    """Load YAML holdout config using schema validation and runtime adapter."""
    validated = load_yaml_as_schema(config_path, HoldoutConfigSchema)
    return _to_holdout_config(validated.model_dump(exclude_none=True))
