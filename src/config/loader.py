# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)
# For commercial licensing, contact: christopher/cjrisi AT gluroo/uwaterloo DOT com/ca

"""Configuration loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml_config(file_path: str | Path) -> Any:
    """Load and parse a YAML configuration file."""
    with open(file_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)
