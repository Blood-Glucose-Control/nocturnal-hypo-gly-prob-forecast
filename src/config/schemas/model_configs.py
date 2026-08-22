"""Model-config schema definitions and schema routing helpers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, NamedTuple, Optional

from pydantic import (
    AliasChoices,
    Field,
    ValidationInfo,
    ValidationError,
    field_validator,
    model_validator,
)

from .base import BaseConfigSchema


def _coerce_numeric_string(value: Any, field_name: str) -> Any:
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError as exc:
            raise ValueError(f"{field_name} must be a numeric value") from exc
    return value


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


class AutoGluonModelConfigSchema(BaseConfigSchema):
    """Shared schema surface for AutoGluon-backed model families."""

    context_length: int = Field(default=512, gt=0)
    forecast_length: int = Field(default=96, gt=0)
    training_mode: str = Field(default="from_scratch")

    min_segment_length: Optional[int] = Field(default=None, gt=0)
    imputation_threshold_mins: int = Field(default=45, gt=0)

    covariate_cols: list[str] = Field(default_factory=list)
    target_col: str = Field(default="bg_mM", min_length=1)
    patient_col: str = Field(default="p_num", min_length=1)
    time_col: str = Field(default="datetime", min_length=1)
    interval_mins: int = Field(default=5, gt=0)

    eval_metric: str = Field(default="WQL", min_length=1)
    enable_ensemble: bool = Field(default=False)
    time_limit: Optional[int] = Field(default=None, gt=0)
    quantile_levels: Optional[list[float]] = Field(default=None)

    @field_validator("quantile_levels")
    @classmethod
    def _validate_quantile_levels(
        cls, levels: Optional[list[float]]
    ) -> Optional[list[float]]:
        if levels is None:
            return None
        if not levels:
            raise ValueError("quantile_levels must not be empty when provided")
        if levels != sorted(levels):
            raise ValueError("quantile_levels must be sorted in ascending order")
        if len(set(levels)) != len(levels):
            raise ValueError("quantile_levels must not contain duplicates")
        for level in levels:
            if level <= 0.0 or level >= 1.0:
                raise ValueError(
                    f"quantile_levels entries must be in (0, 1), got {level}"
                )
        return levels


class Chronos2ModelConfigSchema(AutoGluonModelConfigSchema):
    """Schema contract for Chronos-2 (AutoGluon backend) YAML configs."""

    model_type: Literal["chronos2"] = Field(default="chronos2")
    model_path: Optional[str] = Field(default="autogluon/chronos-2")
    training_mode: str = Field(default="fine_tune")
    freeze_backbone: bool = Field(default=False)
    use_cpu: bool = Field(default=False)
    fp16: bool = Field(default=True)
    learning_rate: float = Field(default=1e-4, gt=0.0)

    fine_tune_steps: int = Field(default=15000, gt=0)
    fine_tune_lr: float = Field(default=1e-5, gt=0.0)
    fine_tune_batch_size: Optional[int] = Field(default=None, gt=0)
    batch_size: Optional[int] = Field(default=None, gt=0)
    fine_tune_logging_steps: Optional[int] = Field(default=None, gt=0)
    eval_during_fine_tune: bool = Field(default=True)
    min_past: int = Field(default=1, gt=0)
    checkpoint_save_steps: Optional[int] = Field(default=None, gt=0)
    covariate_cols: list[str] = Field(default_factory=lambda: ["iob"])
    known_covariate_cols: list[str] = Field(default_factory=list)
    joint_target_cols: list[str] = Field(default_factory=list)

    @field_validator("learning_rate", "fine_tune_lr", mode="before")
    @classmethod
    def _normalize_learning_rate_fields(cls, value: Any, info: ValidationInfo) -> Any:
        field_name = info.field_name or "learning_rate"
        return _coerce_numeric_string(value, field_name)

    @model_validator(mode="after")
    def _validate_joint_target_columns(self) -> "Chronos2ModelConfigSchema":
        if len(self.joint_target_cols) == 1:
            raise ValueError(
                "joint_target_cols must have 0 or >=2 entries; a single entry is invalid"
            )
        if self.joint_target_cols and self.target_col not in self.joint_target_cols:
            raise ValueError(
                f"target_col '{self.target_col}' must be present in joint_target_cols"
            )
        return self


class NaiveBaselineModelConfigSchema(AutoGluonModelConfigSchema):
    """Schema contract for Naive/Average AutoGluon baseline configs."""

    model_type: Literal["naive_baseline"] = Field(default="naive_baseline")
    model_name: Literal["Naive", "Average"] = Field(default="Naive")


class StatisticalModelConfigSchema(AutoGluonModelConfigSchema):
    """Schema contract for AutoGluon statistical model configs."""

    model_type: Literal["statistical"] = Field(default="statistical")
    model_name: Literal["AutoARIMA", "Theta", "NPTS"] = Field(default="AutoARIMA")
    autoarima_max_p: int = Field(default=3, gt=0)
    autoarima_max_q: int = Field(default=3, gt=0)
    autoarima_d: Optional[int] = Field(default=None, ge=0)
    autoarima_D: Optional[int] = Field(default=None, ge=0)
    theta_decomposition_type: Literal["additive", "multiplicative"] = Field(
        default="additive"
    )


class DeepARModelConfigSchema(AutoGluonModelConfigSchema):
    """Schema contract for DeepAR AutoGluon configs."""

    model_type: Literal["deepar"] = Field(default="deepar")
    hidden_size: int = Field(default=64, gt=0)
    num_layers: int = Field(default=2, gt=0)
    dropout_rate: float = Field(default=0.1, ge=0.0, le=1.0)
    lr: float = Field(
        default=1e-3,
        gt=0.0,
        validation_alias=AliasChoices("lr", "learning_rate"),
    )
    num_batches_per_epoch: int = Field(default=50, gt=0)
    batch_size: int = Field(default=128, gt=0)
    max_epochs: int = Field(default=100, gt=0)
    early_stopping_patience: int = Field(default=20, ge=0)
    gradient_clip_val: float = Field(default=10.0, ge=0.0)

    @field_validator("lr", mode="before")
    @classmethod
    def _normalize_lr(cls, value: Any) -> Any:
        return _coerce_numeric_string(value, "lr")


class PatchTSTModelConfigSchema(AutoGluonModelConfigSchema):
    """Schema contract for PatchTST AutoGluon configs."""

    model_type: Literal["patchtst"] = Field(default="patchtst")
    patch_len: int = Field(default=16, gt=0)
    stride: int = Field(default=8, gt=0)
    d_model: int = Field(default=128, gt=0)
    nhead: int = Field(default=16, gt=0)
    num_encoder_layers: int = Field(default=3, gt=0)
    lr: float = Field(
        default=1e-4,
        gt=0.0,
        validation_alias=AliasChoices("lr", "learning_rate"),
    )
    weight_decay: float = Field(default=1e-8, ge=0.0)
    num_batches_per_epoch: int = Field(default=100, gt=0)
    batch_size: int = Field(default=256, gt=0)
    max_epochs: int = Field(default=100, gt=0)
    early_stopping_patience: int = Field(default=20, ge=0)

    @field_validator("lr", mode="before")
    @classmethod
    def _normalize_lr(cls, value: Any) -> Any:
        return _coerce_numeric_string(value, "lr")

    @field_validator("weight_decay", mode="before")
    @classmethod
    def _normalize_weight_decay(cls, value: Any) -> Any:
        return _coerce_numeric_string(value, "weight_decay")

    @model_validator(mode="after")
    def _validate_attention_head_factor(self) -> "PatchTSTModelConfigSchema":
        if self.d_model % self.nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        return self


class TFTModelConfigSchema(AutoGluonModelConfigSchema):
    """Schema contract for Temporal Fusion Transformer AutoGluon configs."""

    model_type: Literal["tft"] = Field(default="tft")
    hidden_dim: int = Field(default=32, gt=0)
    variable_dim: int = Field(default=32, gt=0)
    num_heads: int = Field(default=4, gt=0)
    dropout_rate: float = Field(default=0.1, ge=0.0, le=1.0)
    num_outputs: int = Field(default=1, gt=0)
    lr: float = Field(
        default=1e-3,
        gt=0.0,
        validation_alias=AliasChoices("lr", "learning_rate"),
    )
    num_batches_per_epoch: int = Field(default=50, gt=0)
    batch_size: int = Field(default=256, gt=0)
    max_epochs: int = Field(default=100, gt=0)
    early_stopping_patience: int = Field(default=20, ge=0)
    gradient_clip_val: float = Field(default=1.0, ge=0.0)

    @field_validator("lr", mode="before")
    @classmethod
    def _normalize_lr(cls, value: Any) -> Any:
        return _coerce_numeric_string(value, "lr")


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


def _build_runtime_config(
    model_type: str,
    schema_type: type[BaseConfigSchema],
    config_data: dict[str, Any],
) -> dict[str, Any]:
    try:
        schema = schema_type.model_validate(config_data)
    except ValidationError as exc:
        raise ValueError(
            f"Invalid runtime config for model_type={model_type} via "
            f"{schema_type.__name__}:\n{_format_validation_details(exc)}"
        ) from exc
    return schema.model_dump(exclude_none=True)


def build_tsmixer_runtime_config(config_data: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize TSMixer runtime config values."""
    return _build_runtime_config("tsmixer", TSMixerModelConfigSchema, config_data)


def build_chronos2_runtime_config(config_data: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize Chronos-2 runtime config values."""
    return _build_runtime_config("chronos2", Chronos2ModelConfigSchema, config_data)


def build_naive_baseline_runtime_config(config_data: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize Naive/Average runtime config values."""
    return _build_runtime_config(
        "naive_baseline",
        NaiveBaselineModelConfigSchema,
        config_data,
    )


def build_statistical_runtime_config(config_data: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize statistical runtime config values."""
    return _build_runtime_config(
        "statistical", StatisticalModelConfigSchema, config_data
    )


def build_deepar_runtime_config(config_data: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize DeepAR runtime config values."""
    return _build_runtime_config("deepar", DeepARModelConfigSchema, config_data)


def build_patchtst_runtime_config(config_data: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize PatchTST runtime config values."""
    return _build_runtime_config("patchtst", PatchTSTModelConfigSchema, config_data)


def build_tft_runtime_config(config_data: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize TFT runtime config values."""
    return _build_runtime_config("tft", TFTModelConfigSchema, config_data)


MODEL_CONFIG_ROUTES: dict[str, ModelConfigRoute] = {
    "chronos2": ModelConfigRoute(
        schema_type=Chronos2ModelConfigSchema,
        runtime_adapter=build_chronos2_runtime_config,
    ),
    "deepar": ModelConfigRoute(
        schema_type=DeepARModelConfigSchema,
        runtime_adapter=build_deepar_runtime_config,
    ),
    "naive_baseline": ModelConfigRoute(
        schema_type=NaiveBaselineModelConfigSchema,
        runtime_adapter=build_naive_baseline_runtime_config,
    ),
    "patchtst": ModelConfigRoute(
        schema_type=PatchTSTModelConfigSchema,
        runtime_adapter=build_patchtst_runtime_config,
    ),
    "statistical": ModelConfigRoute(
        schema_type=StatisticalModelConfigSchema,
        runtime_adapter=build_statistical_runtime_config,
    ),
    "tft": ModelConfigRoute(
        schema_type=TFTModelConfigSchema,
        runtime_adapter=build_tft_runtime_config,
    ),
    "tsmixer": ModelConfigRoute(
        schema_type=TSMixerModelConfigSchema,
        runtime_adapter=build_tsmixer_runtime_config,
    ),
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
