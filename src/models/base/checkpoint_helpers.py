"""Shared checkpoint reference helpers for model persistence."""

from __future__ import annotations

import json
import logging
import os
import pickle
import shutil
from dataclasses import dataclass
from typing import Any, Callable

CHECKPOINT_PATH_KEY = "predictor_path"
CHECKPOINT_WEIGHTS_FILE_KEY = "weights_file"


@dataclass(frozen=True)
class CheckpointBundlePaths:
    """Canonical filesystem paths for a model checkpoint bundle."""

    root_dir: str
    config_json: str
    metadata_json: str
    training_metadata_json: str


@dataclass(frozen=True)
class LoadedCheckpointBundle:
    """Loaded checkpoint bundle payloads and source paths."""

    paths: CheckpointBundlePaths
    config: dict[str, Any] | None
    metadata: dict[str, Any]
    metadata_path: str | None


@dataclass(frozen=True)
class CheckpointFilenamePolicy:
    """Shared artifact naming policy for model-specific checkpoint bundles."""

    ttm_preprocessor_artifacts: tuple[str, str] = (
        "preprocessor.pkl",
        "model.pt/preprocessor.pkl",
    )
    timesfm_artifacts: tuple[str, str] = ("hf_model", "timesfm_config.json")
    toto_artifacts: tuple[str, str] = ("toto_backbone.pt", "toto_checkpoint.json")


CHECKPOINT_FILENAME_POLICY = CheckpointFilenamePolicy()


def _shared_checkpoint_paths(base_dir: str, *relative_paths: str) -> tuple[str, ...]:
    """Resolve model-specific artifact names to full checkpoint paths."""
    if not relative_paths:
        raise ValueError("At least one relative artifact path is required.")
    return tuple(
        os.path.join(base_dir, relative_path) for relative_path in relative_paths
    )


def checkpoint_bundle_paths(model_dir: str) -> CheckpointBundlePaths:
    """Return canonical bundle paths rooted at ``model_dir``."""
    return CheckpointBundlePaths(
        root_dir=model_dir,
        config_json=os.path.join(model_dir, "config.json"),
        metadata_json=os.path.join(model_dir, "metadata.json"),
        training_metadata_json=os.path.join(model_dir, "training_metadata.json"),
    )


def _shared_save_checkpoint_bundle(
    model_dir: str,
    *,
    config_payload: dict[str, Any] | None = None,
    metadata_payload: dict[str, Any] | None = None,
    training_metadata_payload: dict[str, Any] | None = None,
) -> CheckpointBundlePaths:
    """Write standard checkpoint-bundle JSON payloads under ``model_dir``."""
    os.makedirs(model_dir, exist_ok=True)
    paths = checkpoint_bundle_paths(model_dir)
    if config_payload is not None:
        write_checkpoint_json(paths.config_json, config_payload)
    if metadata_payload is not None:
        write_checkpoint_json(paths.metadata_json, metadata_payload)
    if training_metadata_payload is not None:
        write_checkpoint_json(paths.training_metadata_json, training_metadata_payload)
    return paths


def _shared_load_checkpoint_bundle(model_dir: str) -> LoadedCheckpointBundle:
    """Load standard checkpoint bundle JSON payloads from ``model_dir``."""
    paths = checkpoint_bundle_paths(model_dir)
    config: dict[str, Any] | None = None
    if os.path.exists(paths.config_json):
        config = read_checkpoint_json(paths.config_json)

    metadata: dict[str, Any] = {}
    metadata_path: str | None = None
    if os.path.exists(paths.metadata_json):
        metadata_path = paths.metadata_json
        metadata = read_checkpoint_json(paths.metadata_json)
    elif os.path.exists(paths.training_metadata_json):
        metadata_path = paths.training_metadata_json
        metadata = read_checkpoint_json(paths.training_metadata_json)

    return LoadedCheckpointBundle(
        paths=paths,
        config=config,
        metadata=metadata,
        metadata_path=metadata_path,
    )


def write_checkpoint_json(path: str, payload: dict[str, Any]) -> None:
    """Write checkpoint JSON payload with UTF-8 encoding."""
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def read_checkpoint_json(path: str) -> dict[str, Any]:
    """Read and return checkpoint JSON payload."""
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def write_checkpoint_config_payload(
    output_dir: str, filename: str, payload: dict[str, Any]
) -> str:
    """Write a model-specific checkpoint config payload and return its path."""
    config_path = _shared_checkpoint_paths(output_dir, filename)[0]
    write_checkpoint_json(config_path, payload)
    return config_path


def read_checkpoint_config_payload(
    model_dir: str, filename: str
) -> dict[str, Any] | None:
    """Read a model-specific checkpoint config payload when present."""
    config_path = _shared_checkpoint_paths(model_dir, filename)[0]
    if not os.path.exists(config_path):
        return None
    return read_checkpoint_json(config_path)


def save_pickle_checkpoint_artifact(
    artifact: Any,
    *,
    paths: tuple[str, ...],
    pickle_module: Any,
    write_missing_parent_paths: bool = False,
) -> tuple[str, ...]:
    """Persist a pickle artifact to one or more candidate checkpoint paths."""
    written_paths: list[str] = []
    for idx, path in enumerate(paths):
        parent_dir = os.path.dirname(path) or "."
        if idx == 0:
            os.makedirs(parent_dir, exist_ok=True)
        elif not os.path.exists(parent_dir):
            if write_missing_parent_paths:
                os.makedirs(parent_dir, exist_ok=True)
            else:
                continue
        with open(path, "wb") as handle:
            pickle_module.dump(artifact, handle)
        written_paths.append(path)
    return tuple(written_paths)


def load_pickle_checkpoint_artifact(
    *,
    paths: tuple[str, ...],
    pickle_module: Any,
) -> tuple[Any | None, str | None]:
    """Load the first available pickle artifact from candidate paths."""
    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path, "rb") as handle:
            return pickle_module.load(handle), path
    return None, None


def _relative_symlink(target: str, link: str) -> None:
    os.symlink(os.path.relpath(os.path.abspath(target), os.path.dirname(link)), link)


def list_intermediate_checkpoint_adapters(
    w0_dir: str,
    *,
    adapter_filename: str = "adapter_model.safetensors",
    log_fn: Callable[[str], None] | None = None,
) -> list[tuple[str, int, str]]:
    """Return intermediate checkpoint dirs that contain LoRA adapter weights."""
    checkpoints = sorted(
        [
            d
            for d in os.listdir(w0_dir)
            if d.startswith("checkpoint-") and os.path.isdir(os.path.join(w0_dir, d))
        ],
        key=lambda checkpoint_name: int(checkpoint_name.split("-")[1]),
    )
    materializable: list[tuple[str, int, str]] = []
    for checkpoint_name in checkpoints:
        step_num = int(checkpoint_name.split("-")[1])
        adapter_src = os.path.join(w0_dir, checkpoint_name, adapter_filename)
        if not os.path.exists(adapter_src):
            if log_fn is not None:
                log_fn(f"  {checkpoint_name}: no {adapter_filename}, skipping")
            continue
        materializable.append((checkpoint_name, step_num, adapter_src))
    return materializable


def build_chronos2_shadow_predictor_snapshot(
    *,
    output_dir: str,
    w0_dir: str,
    adapter_src: str,
    shadow_predictor: str,
) -> None:
    """Create a Chronos2 shadow predictor tree with a swapped adapter checkpoint."""
    os.makedirs(shadow_predictor, exist_ok=True)
    for entry in os.listdir(output_dir):
        if entry in ("models", "snapshots"):
            continue
        _relative_symlink(
            os.path.join(output_dir, entry),
            os.path.join(shadow_predictor, entry),
        )

    models_orig = os.path.join(output_dir, "models")
    shadow_models = os.path.join(shadow_predictor, "models")
    os.makedirs(shadow_models, exist_ok=True)
    for entry in os.listdir(models_orig):
        if entry == "Chronos2":
            continue
        _relative_symlink(
            os.path.join(models_orig, entry),
            os.path.join(shadow_models, entry),
        )

    shadow_c2 = os.path.join(shadow_models, "Chronos2")
    os.makedirs(shadow_c2, exist_ok=True)
    c2_orig = os.path.join(models_orig, "Chronos2")
    for entry in os.listdir(c2_orig):
        if entry == "W0":
            continue
        _relative_symlink(
            os.path.join(c2_orig, entry),
            os.path.join(shadow_c2, entry),
        )

    shadow_w0 = os.path.join(shadow_c2, "W0")
    os.makedirs(shadow_w0, exist_ok=True)
    for entry in os.listdir(w0_dir):
        if entry == "fine-tuned-ckpt" or entry.startswith("checkpoint-"):
            continue
        if entry == "model.pkl":
            with open(os.path.join(w0_dir, "model.pkl"), "rb") as model_file:
                w0_model = pickle.load(model_file)
            w0_model.path = os.path.abspath(shadow_w0)
            with open(os.path.join(shadow_w0, "model.pkl"), "wb") as model_file:
                pickle.dump(w0_model, model_file)
            continue
        _relative_symlink(
            os.path.join(w0_dir, entry),
            os.path.join(shadow_w0, entry),
        )

    orig_ft_ckpt = os.path.join(w0_dir, "fine-tuned-ckpt")
    shadow_ft_ckpt = os.path.join(shadow_w0, "fine-tuned-ckpt")
    os.makedirs(shadow_ft_ckpt, exist_ok=True)
    for entry in os.listdir(orig_ft_ckpt):
        if entry == "adapter_model.safetensors":
            continue
        _relative_symlink(
            os.path.join(orig_ft_ckpt, entry),
            os.path.join(shadow_ft_ckpt, entry),
        )
    shutil.copy2(
        adapter_src,
        os.path.join(shadow_ft_ckpt, "adapter_model.safetensors"),
    )


def write_snapshot_model_pt_bundle(
    *,
    snapshot_dir: str,
    main_model_pt: str,
    predictor_reference_filename: str,
    predictor_target_path: str,
    config_payload: dict[str, Any],
) -> str:
    """Write model.pt snapshot payload for an intermediate checkpoint."""
    snapshot_model_pt = os.path.join(snapshot_dir, "model.pt")
    os.makedirs(snapshot_model_pt, exist_ok=True)
    write_checkpoint_reference(
        output_dir=snapshot_model_pt,
        reference_filename=predictor_reference_filename,
        target_path=predictor_target_path,
    )
    write_checkpoint_json(
        os.path.join(snapshot_model_pt, "config.json"), config_payload
    )
    meta_src = os.path.join(main_model_pt, "metadata.json")
    if os.path.exists(meta_src):
        shutil.copy2(meta_src, os.path.join(snapshot_model_pt, "metadata.json"))
    return snapshot_model_pt


def write_checkpoint_reference(
    output_dir: str,
    reference_filename: str,
    target_path: str,
    *,
    path_key: str = CHECKPOINT_PATH_KEY,
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
    path_key: str = CHECKPOINT_PATH_KEY,
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
