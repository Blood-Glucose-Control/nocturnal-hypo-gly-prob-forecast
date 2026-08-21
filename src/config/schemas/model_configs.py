"""Model-config schema definitions and schema routing helpers."""

from __future__ import annotations

from typing import Optional

from pydantic import AliasChoices, Field

from .base import BaseConfigSchema


class TSMixerModelConfigSchema(BaseConfigSchema):
    """Schema contract for TSMixer model YAML configs."""

    model_type: str = Field(default="tsmixer")
    training_mode: str = Field(default="from_scratch")
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


MODEL_CONFIG_SCHEMAS = {
    "tsmixer": TSMixerModelConfigSchema,
}


def get_model_config_schema(model_type: str):
    """Return schema class for a model type, or None when not yet migrated."""
    return MODEL_CONFIG_SCHEMAS.get(model_type.lower())
