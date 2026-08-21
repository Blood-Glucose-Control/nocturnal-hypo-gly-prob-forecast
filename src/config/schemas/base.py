"""Shared Pydantic schema foundations for configuration validation."""

from pydantic import BaseModel, ConfigDict


class BaseConfigSchema(BaseModel):
    """Default strict schema policy for config validation."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        populate_by_name=True,
        use_enum_values=True,
        str_strip_whitespace=True,
    )
