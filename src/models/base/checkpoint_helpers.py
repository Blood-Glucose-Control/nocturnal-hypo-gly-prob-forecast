"""Shared checkpoint reference helpers for model persistence."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CheckpointBundlePaths:
    """Canonical filesystem paths for a model checkpoint bundle."""

    root_dir: str
    config_json: str
    metadata_json: str
    training_metadata_json: str


def checkpoint_bundle_paths(model_dir: str) -> CheckpointBundlePaths:
    """Return canonical bundle paths rooted at ``model_dir``."""
    return CheckpointBundlePaths(
        root_dir=model_dir,
        config_json=os.path.join(model_dir, "config.json"),
        metadata_json=os.path.join(model_dir, "metadata.json"),
        training_metadata_json=os.path.join(model_dir, "training_metadata.json"),
    )


def write_checkpoint_json(path: str, payload: dict[str, Any]) -> None:
    """Write checkpoint JSON payload with UTF-8 encoding."""
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def read_checkpoint_json(path: str) -> dict[str, Any]:
    """Read and return checkpoint JSON payload."""
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def write_checkpoint_reference(
    output_dir: str,
    reference_filename: str,
    target_path: str,
    *,
    path_key: str = "predictor_path",
    relative_to_output: bool = False,
) -> str:
    """Write a JSON file that stores a checkpoint-related filesystem path."""
    os.makedirs(output_dir, exist_ok=True)
    reference_path = os.path.join(output_dir, reference_filename)
    stored_path = (
        os.path.relpath(target_path, output_dir) if relative_to_output else target_path
    )
    with open(reference_path, "w", encoding="utf-8") as handle:
        json.dump({path_key: stored_path}, handle, indent=2)
    return reference_path


def resolve_checkpoint_reference(
    model_dir: str,
    reference_filename: str,
    *,
    path_key: str = "predictor_path",
    required_file: str | None = None,
    logger: logging.Logger | None = None,
) -> str:
    """Resolve a referenced path from JSON with fallback to ``model_dir``."""
    reference_path = os.path.join(model_dir, reference_filename)
    if not os.path.exists(reference_path):
        return model_dir

    with open(reference_path, encoding="utf-8") as handle:
        referenced_path = json.load(handle)[path_key]

    if not os.path.isabs(referenced_path):
        referenced_path = os.path.normpath(
            os.path.join(os.path.dirname(reference_path), referenced_path)
        )

    if required_file is None:
        resolved_exists = os.path.exists(referenced_path)
    else:
        resolved_exists = os.path.exists(os.path.join(referenced_path, required_file))

    if not resolved_exists:
        if logger is not None:
            logger.warning(
                "Checkpoint reference %s not found at %s; falling back to %s",
                reference_filename,
                referenced_path,
                model_dir,
            )
        return model_dir

    if logger is not None:
        logger.info("Loading checkpoint reference from %s", referenced_path)
    return referenced_path
