"""Model-config schema definitions and schema routing helpers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple, Optional

from pydantic import AliasChoices, Field
from pydantic import ValidationError

from .base import BaseConfigSchema


class TSMixerModelConfigSchema(BaseConfigSchema):
    """Schema contract for TSMixer model YAML configs."""

    model_type: str = Field(default="tsmixer")
    model_path: Optional[str] = Field(default=None)
    training_mode: str = Field(default="from_scratch")
    freeze_backbone: bool = Field(default=False)
    use_cpu: bool = Field(default=False)
    fp16: bool = Field(default=True)
    context_length: int = Field(default=512, gt=0)
    forecast_length: int = Field(default=96, gt=0)

    hidden_size: int = Field(default=64, gt=0)
    ff_size: int = Field(default=64, gt=0)
    num_blocks: int = Field(default=2, gt=0)
    activation: str = Field(default="ReLU")
    dropout: float = Field(default=0.1, ge=0.0, le=1.0)
    norm_type: str = Field(default="LayerNorm")
    normalize_before: bool = Field(default=False)
    use_static_covariates: bool = Field(default=False)
    random_state: int = Field(default=42)

    learning_rate: float = Field(
        default=1e-3,
        gt=0.0,
        validation_alias=AliasChoices("learning_rate", "lr"),
    )
    batch_size: int = Field(default=32, gt=0)
    num_epochs: int = Field(default=10, gt=0)
    quantile_levels: Optional[list[float]] = Field(default=None)

    covariate_cols: list[str] = Field(default_factory=list)
    target_col: str = Field(default="bg_mM")
    patient_col: str = Field(default="p_num")
    time_col: str = Field(default="datetime")
    interval_mins: int = Field(default=5, gt=0)
    imputation_threshold_mins: int = Field(default=45, gt=0)
    min_segment_length: Optional[int] = Field(default=None, gt=0)


RuntimeConfigAdapter = Callable[[dict[str, Any]], dict[str, Any]]


class ModelConfigRoute(NamedTuple):
    """Schema + runtime-adapter route for one model family."""

    schema_type: type[BaseConfigSchema]
    runtime_adapter: RuntimeConfigAdapter


def _format_validation_details(exc: ValidationError) -> str:
    details = []
    for err in exc.errors():
        loc = ".".join(str(part) for part in err.get("loc", ()))
        msg = err.get("msg", "validation error")
        details.append(f"  - {loc}: {msg}" if loc else f"  - {msg}")
    return "\n".join(details)


def build_tsmixer_runtime_config(config_data: dict[str, Any]):
    """Validate and normalize TSMixer runtime config values."""
    try:
        schema = TSMixerModelConfigSchema.model_validate(config_data)
    except ValidationError as exc:
        raise ValueError(
            "Invalid runtime config for model_type=tsmixer via "
            "TSMixerModelConfigSchema:\n"
            f"{_format_validation_details(exc)}"
        ) from exc

    return schema.model_dump(exclude_none=True)


MODEL_CONFIG_ROUTES: dict[str, ModelConfigRoute] = {
    "tsmixer": ModelConfigRoute(
        schema_type=TSMixerModelConfigSchema,
        runtime_adapter=build_tsmixer_runtime_config,
    )
}


def _get_route(model_type: str) -> Optional[ModelConfigRoute]:
    return MODEL_CONFIG_ROUTES.get(model_type.lower())


def get_model_config_schema(model_type: str | None) -> Optional[type[BaseConfigSchema]]:
    """Return schema class for a model type, or None when not yet migrated."""
    if not model_type:
        return None
    route = _get_route(model_type)
    if route is None:
        return None
    return route.schema_type


def get_registered_model_config_types() -> tuple[str, ...]:
    """List model types currently routed through schema + runtime adapters."""
    return tuple(sorted(MODEL_CONFIG_ROUTES.keys()))


def build_model_runtime_config(model_type: str, config_data: dict[str, Any]):
    """Build a model runtime config object via the model schema adapter."""
    route = _get_route(model_type)
    if route is None:
        registered_types = ", ".join(get_registered_model_config_types()) or "(none)"
        raise ValueError(
            f"No runtime config adapter registered for model_type={model_type}. "
            f"Registered adapter types: {registered_types}"
        )
    return route.runtime_adapter(config_data)
