"""Unit tests for shared checkpoint helper utilities."""

from __future__ import annotations

import pickle
from pathlib import Path

from src.models.base.checkpoint_helpers import (
    CHECKPOINT_FILENAME_POLICY,
    CHECKPOINT_WEIGHTS_FILE_KEY,
    _shared_checkpoint_paths,
    _shared_load_checkpoint_bundle,
    _shared_save_checkpoint_bundle,
    list_intermediate_checkpoint_adapters,
    load_pickle_checkpoint_artifact,
    read_checkpoint_config_payload,
    save_pickle_checkpoint_artifact,
    write_checkpoint_config_payload,
)


def test_checkpoint_filename_policy_constants_are_stable() -> None:
    assert CHECKPOINT_FILENAME_POLICY.ttm_preprocessor_artifacts == (
        "preprocessor.pkl",
        "model.pt/preprocessor.pkl",
    )
    assert CHECKPOINT_FILENAME_POLICY.timesfm_artifacts == (
        "hf_model",
        "timesfm_config.json",
    )
    assert CHECKPOINT_FILENAME_POLICY.toto_artifacts == (
        "toto_backbone.pt",
        "toto_checkpoint.json",
    )


def test_shared_checkpoint_paths_resolves_relative_artifacts(tmp_path: Path) -> None:
    first, second = _shared_checkpoint_paths(
        str(tmp_path), "first.bin", "nested/second.bin"
    )
    assert first == str(tmp_path / "first.bin")
    assert second == str(tmp_path / "nested" / "second.bin")


def test_checkpoint_config_payload_round_trip(tmp_path: Path) -> None:
    payload = {CHECKPOINT_WEIGHTS_FILE_KEY: "weights.pt", "is_finetuned": True}
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


def test_shared_checkpoint_bundle_round_trip_and_metadata_fallback(
    tmp_path: Path,
) -> None:
    _shared_save_checkpoint_bundle(
        str(tmp_path),
        config_payload={"model_type": "stub"},
        training_metadata_payload={"metrics": {"loss": 1.0}},
    )
    bundle = _shared_load_checkpoint_bundle(str(tmp_path))
    assert bundle.config == {"model_type": "stub"}
    assert bundle.metadata == {"metrics": {"loss": 1.0}}
    assert bundle.metadata_path == str(tmp_path / "training_metadata.json")

    _shared_save_checkpoint_bundle(
        str(tmp_path),
        metadata_payload={"is_fitted": True},
    )
    bundle = _shared_load_checkpoint_bundle(str(tmp_path))
    assert bundle.metadata == {"is_fitted": True}
    assert bundle.metadata_path == str(tmp_path / "metadata.json")


def test_list_intermediate_checkpoint_adapters_filters_missing(tmp_path: Path) -> None:
    w0_dir = tmp_path / "W0"
    w0_dir.mkdir()
    good = w0_dir / "checkpoint-1000"
    bad = w0_dir / "checkpoint-2000"
    good.mkdir()
    bad.mkdir()
    (good / "adapter_model.safetensors").write_text("ok")

    logs: list[str] = []
    checkpoints = list_intermediate_checkpoint_adapters(str(w0_dir), log_fn=logs.append)
    assert checkpoints == [
        (
            "checkpoint-1000",
            1000,
            str(good / "adapter_model.safetensors"),
        )
    ]
    assert logs == ["  checkpoint-2000: no adapter_model.safetensors, skipping"]
