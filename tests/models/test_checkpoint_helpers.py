"""Unit tests for shared checkpoint helper utilities."""

from __future__ import annotations

import pickle
from pathlib import Path

from src.models.base.checkpoint_helpers import (
    _shared_checkpoint_paths,
    load_pickle_checkpoint_artifact,
    read_checkpoint_config_payload,
    save_pickle_checkpoint_artifact,
    write_checkpoint_config_payload,
)


def test_shared_checkpoint_paths_resolves_relative_artifacts(tmp_path: Path) -> None:
    first, second = _shared_checkpoint_paths(
        str(tmp_path), "first.bin", "nested/second.bin"
    )
    assert first == str(tmp_path / "first.bin")
    assert second == str(tmp_path / "nested" / "second.bin")


def test_checkpoint_config_payload_round_trip(tmp_path: Path) -> None:
    payload = {"weights_file": "weights.pt", "is_finetuned": True}
    written_path = write_checkpoint_config_payload(
        str(tmp_path), "payload.json", payload
    )
    assert written_path == str(tmp_path / "payload.json")
    assert read_checkpoint_config_payload(str(tmp_path), "payload.json") == payload


def test_save_and_load_pickle_checkpoint_artifact_prefers_first_available_path(
    tmp_path: Path,
) -> None:
    primary, secondary = _shared_checkpoint_paths(
        str(tmp_path), "preprocessor.pkl", "model.pt/preprocessor.pkl"
    )
    artifact = {"key": "value"}
    written = save_pickle_checkpoint_artifact(
        artifact, paths=(primary, secondary), pickle_module=pickle
    )

    assert written == (primary,)
    loaded, loaded_path = load_pickle_checkpoint_artifact(
        paths=(primary, secondary), pickle_module=pickle
    )
    assert loaded == artifact
    assert loaded_path == primary


def test_save_pickle_checkpoint_artifact_writes_secondary_when_parent_exists(
    tmp_path: Path,
) -> None:
    nested_dir = tmp_path / "model.pt"
    nested_dir.mkdir(parents=True, exist_ok=True)
    primary, secondary = _shared_checkpoint_paths(
        str(tmp_path), "preprocessor.pkl", "model.pt/preprocessor.pkl"
    )
    written = save_pickle_checkpoint_artifact(
        {"x": 1}, paths=(primary, secondary), pickle_module=pickle
    )
    assert written == (primary, secondary)
