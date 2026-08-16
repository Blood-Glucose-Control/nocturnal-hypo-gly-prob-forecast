"""Canonical run-manifest helpers for workflow entrypoints."""

from __future__ import annotations

import getpass
import json
import os
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.utils import get_git_branch, get_git_commit_hash, is_git_dirty


def utc_now() -> datetime:
    """Return current UTC timestamp with timezone information."""
    return datetime.now(timezone.utc)


def utc_iso(dt: datetime) -> str:
    """Serialize a timestamp as ISO8601 UTC string."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")


def _run_git_command(args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    value = result.stdout.strip()
    return value if value else None


def detect_repository() -> str:
    """Return repository identifier/URL for provenance metadata."""
    repo = os.environ.get("GITHUB_REPOSITORY")
    if repo:
        return repo
    remote = _run_git_command(["config", "--get", "remote.origin.url"])
    return remote or "unknown"


def detect_launcher_type() -> str:
    """Infer launcher type from runtime environment."""
    if os.environ.get("SLURM_JOB_ID"):
        return "slurm"
    if os.environ.get("TMUX"):
        return "tmux"
    return "local"


def collect_code_provenance() -> dict[str, Any]:
    """Collect source-code provenance metadata."""
    return {
        "git_commit": get_git_commit_hash(),
        "git_branch": get_git_branch(),
        "git_dirty": is_git_dirty(),
        "repository": detect_repository(),
    }


def collect_execution_context() -> dict[str, Any]:
    """Collect runtime execution-context metadata."""
    return {
        "launcher_type": detect_launcher_type(),
        "host": socket.gethostname(),
        "user": getpass.getuser(),
        "python_version": sys.version.split()[0],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }


def build_run_manifest(
    *,
    run_id: str,
    workflow_name: str,
    workflow_version: str,
    created_at_utc: datetime,
    started_at_utc: datetime,
    ended_at_utc: datetime,
    parent_run_id: str | None = None,
    data_config_paths: list[str] | None = None,
    data_snapshot_ids: list[str] | None = None,
    model_config_path: str | None = None,
    experiment_config_path: str | None = None,
    seed: int | None = None,
    resolved_runtime_config: Mapping[str, Any] | None = None,
    artifact_root: str | None = None,
    checkpoint_paths: list[str] | None = None,
    prediction_paths: list[str] | None = None,
    plot_paths: list[str] | None = None,
    key_metrics: Mapping[str, Any] | None = None,
    status: str = "success",
    failure_message: str | None = None,
) -> dict[str, Any]:
    """Build canonical pre-MLflow run-manifest payload."""
    if status not in {"success", "failed", "interrupted"}:
        raise ValueError(f"Unsupported run-manifest status: {status}")

    provenance = collect_code_provenance()
    context = collect_execution_context()
    duration_seconds = max(
        0.0,
        (ended_at_utc - started_at_utc).total_seconds(),
    )

    return {
        "run_id": run_id,
        "parent_run_id": parent_run_id,
        "workflow_name": workflow_name,
        "workflow_version": workflow_version,
        "created_at_utc": utc_iso(created_at_utc),
        "started_at_utc": utc_iso(started_at_utc),
        "ended_at_utc": utc_iso(ended_at_utc),
        "duration_seconds": duration_seconds,
        **provenance,
        **context,
        "data_config_paths": data_config_paths or [],
        "data_snapshot_ids": data_snapshot_ids or [],
        "model_config_path": model_config_path,
        "experiment_config_path": experiment_config_path,
        "seed": seed,
        "resolved_runtime_config": dict(resolved_runtime_config or {}),
        "artifact_root": artifact_root,
        "checkpoint_paths": checkpoint_paths or [],
        "prediction_paths": prediction_paths or [],
        "plot_paths": plot_paths or [],
        "key_metrics": dict(key_metrics or {}),
        "status": status,
        "failure_message": failure_message,
    }


def write_run_manifest(
    *,
    output_dir: str | Path,
    manifest: Mapping[str, Any],
    filename: str = "run_manifest.json",
) -> Path:
    """Persist run-manifest JSON in output directory."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    manifest_path = output_path / filename
    with open(manifest_path, "w") as f:
        json.dump(dict(manifest), f, indent=2, sort_keys=True)
    return manifest_path
