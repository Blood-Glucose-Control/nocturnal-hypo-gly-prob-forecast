"""Tests for forecasting workflow output directory resolution."""

from pathlib import Path

from src.workflows.forecasting.pipeline import _is_existing_run_directory


def test_is_existing_run_directory_true_for_rid_run_name() -> None:
    run_dir = Path(
        "trained_models/artifacts/tsmixer/2026-08-18_22:39_RID123_forecasting_workflow"
    )
    assert _is_existing_run_directory(run_dir)


def test_is_existing_run_directory_false_for_artifact_root() -> None:
    artifact_root = Path("trained_models/artifacts/tsmixer")
    assert not _is_existing_run_directory(artifact_root)
