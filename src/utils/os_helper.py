# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)
# For commercial licensing, contact: christopher/cjrisi AT gluroo/uwaterloo DOT com/ca

from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def get_project_root() -> Path:
    """
    Get the project root directory.

    This function looks for setup.py or pyproject.toml to identify the project root.

    Returns:
        Path: Path to the project root directory

    Raises:
        FileNotFoundError: If project root cannot be determined
    """
    current_path = Path.cwd()

    # Walk up the directory tree looking for project root indicators
    for path in [current_path] + list(current_path.parents):
        if (path / "setup.py").exists() or (path / "pyproject.toml").exists():
            return path

    # Fallback: if we can't find project root, use current directory
    logger.warning("Could not determine project root, using current directory")
    return current_path
