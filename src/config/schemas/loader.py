"""YAML-to-schema validation helpers with actionable error messages."""

from __future__ import annotations

from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from ..loader import load_yaml_config

SchemaT = TypeVar("SchemaT", bound=BaseModel)


def load_yaml_as_schema(file_path: str | Path, schema_type: type[SchemaT]) -> SchemaT:
    """Load YAML and validate against a schema class."""
    config_path = Path(file_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    raw = load_yaml_config(str(config_path))
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(
            f"Config file must contain a YAML mapping/object at top level: {config_path}"
        )

    try:
        return schema_type.model_validate(raw)
    except ValidationError as exc:
        details = []
        for err in exc.errors():
            loc = ".".join(str(part) for part in err.get("loc", ()))
            msg = err.get("msg", "validation error")
            details.append(f"  - {loc}: {msg}" if loc else f"  - {msg}")
        detail_text = "\n".join(details)
        raise ValueError(
            f"Invalid config in {config_path} for schema {schema_type.__name__}:\n"
            f"{detail_text}"
        ) from exc
