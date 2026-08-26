"""Tests for smoke-suite pre/post comparison utility."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _build_run_manifest(
    run_dir: Path, train_loss: float, with_extra_plot: bool = False
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "run_manifest.json"

    checkpoint_paths = [str(run_dir / "model.pt")]
    prediction_paths = [str(run_dir / "predictions" / "1_after_training" / "p1.json")]
    plot_paths = [str(run_dir / "forecasts" / "1_after_training" / "summary.csv")]
    if with_extra_plot:
        plot_paths.append(str(run_dir / "forecasts" / "1_after_training" / "extra.csv"))

    payload = {
        "artifact_root": str(run_dir),
        "status": "success",
        "checkpoint_paths": checkpoint_paths,
        "prediction_paths": prediction_paths,
        "plot_paths": plot_paths,
        "key_metrics": {
            "step5_train_loss": train_loss,
            "step5_eval_loss": train_loss * 1.1,
        },
    }
    _write_json(manifest_path, payload)
    return manifest_path


def _build_suite_manifest(path: Path, model_type: str, run_manifest_path: Path) -> Path:
    payload = {
        "schema_version": "1",
        "suite_label": path.parent.name,
        "runs": [
            {
                "model_type": model_type,
                "output_dir": str(run_manifest_path.parent),
                "run_manifest_path": str(run_manifest_path),
                "status": "success",
                "return_code": 0,
            }
        ],
    }
    _write_json(path, payload)
    return path


def _compare_script_path() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    return (
        repo_root
        / "scripts"
        / "workflows"
        / "forecasting"
        / "compare_regression_smoke_suites.py"
    )


def test_compare_smoke_suites_passes_within_tolerance(tmp_path: Path) -> None:
    baseline_run_manifest = _build_run_manifest(
        tmp_path / "baseline_suite" / "ttm", train_loss=0.85
    )
    candidate_run_manifest = _build_run_manifest(
        tmp_path / "candidate_suite" / "ttm", train_loss=0.90
    )
    baseline_suite = _build_suite_manifest(
        tmp_path / "baseline_suite" / "suite_manifest.json",
        model_type="ttm",
        run_manifest_path=baseline_run_manifest,
    )
    candidate_suite = _build_suite_manifest(
        tmp_path / "candidate_suite" / "suite_manifest.json",
        model_type="ttm",
        run_manifest_path=candidate_run_manifest,
    )
    report_path = tmp_path / "candidate_suite" / "report.json"

    result = subprocess.run(
        [
            sys.executable,
            str(_compare_script_path()),
            "--baseline",
            str(baseline_suite),
            "--candidate",
            str(candidate_suite),
            "--rel-tol",
            "0.25",
            "--report-path",
            str(report_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    assert report_path.exists()


def test_compare_smoke_suites_fails_on_artifact_mismatch(tmp_path: Path) -> None:
    baseline_run_manifest = _build_run_manifest(
        tmp_path / "baseline_suite" / "ttm", train_loss=0.85
    )
    candidate_run_manifest = _build_run_manifest(
        tmp_path / "candidate_suite" / "ttm",
        train_loss=0.86,
        with_extra_plot=True,
    )
    baseline_suite = _build_suite_manifest(
        tmp_path / "baseline_suite" / "suite_manifest.json",
        model_type="ttm",
        run_manifest_path=baseline_run_manifest,
    )
    candidate_suite = _build_suite_manifest(
        tmp_path / "candidate_suite" / "suite_manifest.json",
        model_type="ttm",
        run_manifest_path=candidate_run_manifest,
    )

    result = subprocess.run(
        [
            sys.executable,
            str(_compare_script_path()),
            "--baseline",
            str(baseline_suite),
            "--candidate",
            str(candidate_suite),
            "--rel-tol",
            "0.25",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "plot_paths mismatch" in result.stdout


def test_compare_smoke_suites_recovers_nested_manifest_paths(tmp_path: Path) -> None:
    baseline_run_manifest = _build_run_manifest(
        tmp_path / "baseline_suite" / "ttm" / "nested_run", train_loss=0.85
    )
    candidate_run_manifest = _build_run_manifest(
        tmp_path / "candidate_suite" / "ttm" / "nested_run", train_loss=0.86
    )

    baseline_suite_path = tmp_path / "baseline_suite" / "suite_manifest.json"
    candidate_suite_path = tmp_path / "candidate_suite" / "suite_manifest.json"

    _write_json(
        baseline_suite_path,
        {
            "schema_version": "1",
            "suite_label": "baseline_suite",
            "runs": [
                {
                    "model_type": "ttm",
                    "run_id": "baseline_ttm",
                    "output_dir": str(tmp_path / "baseline_suite" / "ttm"),
                    "run_manifest_path": str(
                        tmp_path / "baseline_suite" / "ttm" / "run_manifest.json"
                    ),
                    "status": "failed_missing_manifest",
                    "return_code": 0,
                }
            ],
        },
    )
    _write_json(
        candidate_suite_path,
        {
            "schema_version": "1",
            "suite_label": "candidate_suite",
            "runs": [
                {
                    "model_type": "ttm",
                    "run_id": "candidate_ttm",
                    "output_dir": str(tmp_path / "candidate_suite" / "ttm"),
                    "run_manifest_path": str(
                        tmp_path / "candidate_suite" / "ttm" / "run_manifest.json"
                    ),
                    "status": "failed_missing_manifest",
                    "return_code": 0,
                }
            ],
        },
    )

    # Ensure nested manifests carry matching run IDs so resolver can select them.
    baseline_payload = json.loads(baseline_run_manifest.read_text())
    baseline_payload["run_id"] = "baseline_ttm"
    _write_json(baseline_run_manifest, baseline_payload)
    candidate_payload = json.loads(candidate_run_manifest.read_text())
    candidate_payload["run_id"] = "candidate_ttm"
    _write_json(candidate_run_manifest, candidate_payload)

    result = subprocess.run(
        [
            sys.executable,
            str(_compare_script_path()),
            "--baseline",
            str(baseline_suite_path),
            "--candidate",
            str(candidate_suite_path),
            "--rel-tol",
            "0.25",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + "\n" + result.stderr
