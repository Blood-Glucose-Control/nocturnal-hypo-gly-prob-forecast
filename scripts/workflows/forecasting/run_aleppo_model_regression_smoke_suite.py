#!/usr/bin/env python3
"""Run one-epoch Aleppo forecasting smoke suite across maintained model families."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SmokeModelSpec:
    model_type: str
    model_config: str
    venv_name: str


MODEL_SPECS: dict[str, SmokeModelSpec] = {
    "ttm": SmokeModelSpec(
        "ttm", "configs/models/ttm/97_regression_smoke_balanced.yaml", "ttm"
    ),
    "chronos2": SmokeModelSpec(
        "chronos2",
        "configs/models/chronos2/97_regression_smoke_balanced.yaml",
        "autogluon",
    ),
    "moment": SmokeModelSpec(
        "moment", "configs/models/moment/97_regression_smoke_balanced.yaml", "moment"
    ),
    "timesfm": SmokeModelSpec(
        "timesfm", "configs/models/timesfm/97_regression_smoke_balanced.yaml", "timesfm"
    ),
    "timegrad": SmokeModelSpec(
        "timegrad",
        "configs/models/timegrad/97_regression_smoke_balanced.yaml",
        "timegrad",
    ),
    "tide": SmokeModelSpec(
        "tide", "configs/models/tide/97_regression_smoke_balanced.yaml", "autogluon"
    ),
    "toto": SmokeModelSpec(
        "toto", "configs/models/toto/97_regression_smoke_balanced.yaml", "toto"
    ),
    "moirai": SmokeModelSpec(
        "moirai", "configs/models/moirai/97_regression_smoke_balanced.yaml", "moirai"
    ),
    "naive_baseline": SmokeModelSpec(
        "naive_baseline", "configs/models/naive_baseline/00_naive.yaml", "autogluon"
    ),
    "statistical": SmokeModelSpec(
        "statistical",
        "configs/models/statistical/97_regression_smoke_balanced.yaml",
        "autogluon",
    ),
    "deepar": SmokeModelSpec(
        "deepar", "configs/models/deepar/97_regression_smoke_balanced.yaml", "autogluon"
    ),
    "patchtst": SmokeModelSpec(
        "patchtst",
        "configs/models/patchtst/97_regression_smoke_balanced.yaml",
        "autogluon",
    ),
    "tft": SmokeModelSpec(
        "tft", "configs/models/tft/97_regression_smoke_balanced.yaml", "autogluon"
    ),
    "tsmixer": SmokeModelSpec(
        "tsmixer", "configs/models/tsmixer/00_iob_cob_smoke.yaml", "darts"
    ),
}

DEFAULT_MODEL_ORDER = [
    "ttm",
    "chronos2",
    "moment",
    "timesfm",
    "timegrad",
    "tide",
    "toto",
    "moirai",
    "naive_baseline",
    "statistical",
    "deepar",
    "patchtst",
    "tft",
    "tsmixer",
]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _default_suite_label() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _parse_models(raw_models: str | None) -> list[str]:
    if raw_models is None or raw_models.strip() == "":
        return DEFAULT_MODEL_ORDER.copy()
    requested = [entry.strip() for entry in raw_models.split(",") if entry.strip()]
    unknown = [entry for entry in requested if entry not in MODEL_SPECS]
    if unknown:
        raise ValueError(
            f"Unknown model(s): {', '.join(unknown)}. "
            f"Known models: {', '.join(DEFAULT_MODEL_ORDER)}"
        )
    return requested


def _load_run_manifest(manifest_path: Path) -> dict[str, Any] | None:
    if not manifest_path.exists():
        return None
    with open(manifest_path) as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Manifest is not a JSON object: {manifest_path}")
    return payload


def _resolve_run_manifest(
    output_dir: Path, run_id: str
) -> tuple[Path, dict[str, Any] | None]:
    direct_manifest_path = output_dir / "run_manifest.json"
    direct_manifest = _load_run_manifest(direct_manifest_path)
    if direct_manifest is not None:
        return direct_manifest_path, direct_manifest

    nested_candidates = sorted(
        output_dir.glob("**/run_manifest.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not nested_candidates:
        return direct_manifest_path, None

    fallback_match: tuple[Path, dict[str, Any]] | None = None
    for candidate_path in nested_candidates:
        candidate_manifest = _load_run_manifest(candidate_path)
        if candidate_manifest is None:
            continue
        if candidate_manifest.get("run_id") == run_id:
            return candidate_path, candidate_manifest
        if fallback_match is None:
            fallback_match = (candidate_path, candidate_manifest)

    if fallback_match is not None:
        return fallback_match
    return direct_manifest_path, None


def _run_single_model(
    *,
    repo_root: Path,
    smoke_script_path: Path,
    suite_label: str,
    suite_dir: Path,
    spec: SmokeModelSpec,
    datasets: str,
    config_dir: str,
    epochs: int,
    skip_steps: str,
    seed: int,
    dry_run: bool,
) -> dict[str, Any]:
    output_dir = suite_dir / spec.model_type
    run_id = f"{suite_label}_{spec.model_type}"
    started = _utc_now_iso()

    env = os.environ.copy()
    env.update(
        {
            "MODEL_TYPE": spec.model_type,
            "MODEL_CONFIG": spec.model_config,
            "VENV_NAME": spec.venv_name,
            "DATASETS": datasets,
            "CONFIG_DIR": config_dir,
            "SKIP_TRAINING": "false",
            "SKIP_STEPS": skip_steps,
            "EPOCHS": str(epochs),
            "RUN_ID": run_id,
            "OUTPUT_BASE_DIR": str(output_dir),
            "SEED": str(seed),
            "PYTHONHASHSEED": str(seed),
            "CUBLAS_WORKSPACE_CONFIG": env.get("CUBLAS_WORKSPACE_CONFIG", ":4096:8"),
        }
    )

    cmd = ["bash", str(smoke_script_path)]
    print(f"\n=== [{spec.model_type}] Starting smoke run ===")
    print(f"Output: {output_dir}")
    print(f"Config: {spec.model_config}")
    print(f"Venv:   {spec.venv_name}")

    if dry_run:
        print("Dry-run command:")
        print("  " + " ".join(cmd))
        return {
            "model_type": spec.model_type,
            "model_config": spec.model_config,
            "venv_name": spec.venv_name,
            "output_dir": str(output_dir),
            "run_manifest_path": str(output_dir / "run_manifest.json"),
            "return_code": 0,
            "status": "dry_run",
            "started_at_utc": started,
            "ended_at_utc": _utc_now_iso(),
            "key_metrics": {},
            "artifact_counts": {},
        }

    result = subprocess.run(cmd, cwd=repo_root, env=env, check=False)
    manifest_path, manifest = _resolve_run_manifest(output_dir, run_id)
    ended = _utc_now_iso()

    manifest_status = None
    key_metrics: dict[str, Any] = {}
    artifact_counts: dict[str, int] = {}
    if manifest is not None:
        manifest_status = manifest.get("status")
        loaded_metrics = manifest.get("key_metrics")
        if isinstance(loaded_metrics, dict):
            key_metrics = {
                str(metric_name): metric_value
                for metric_name, metric_value in loaded_metrics.items()
                if isinstance(metric_value, (int, float))
            }
        for field_name in ("checkpoint_paths", "prediction_paths", "plot_paths"):
            values = manifest.get(field_name, [])
            if isinstance(values, list):
                artifact_counts[field_name] = len(values)

    status = (
        "success"
        if result.returncode == 0 and manifest_status == "success"
        else (
            "failed_missing_manifest"
            if result.returncode == 0 and manifest is None
            else "failed"
        )
    )

    return {
        "model_type": spec.model_type,
        "model_config": spec.model_config,
        "venv_name": spec.venv_name,
        "output_dir": str(output_dir),
        "run_manifest_path": str(manifest_path),
        "return_code": int(result.returncode),
        "status": status,
        "manifest_status": manifest_status,
        "started_at_utc": started,
        "ended_at_utc": ended,
        "key_metrics": key_metrics,
        "artifact_counts": artifact_counts,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one-epoch Aleppo forecasting workflow smoke suite across maintained models."
        )
    )
    parser.add_argument(
        "--suite-label",
        default=_default_suite_label(),
        help="Suite run label used in output path and run IDs.",
    )
    parser.add_argument(
        "--models",
        default=None,
        help=(
            "Comma-separated model IDs to run. Default: all maintained workflow models."
        ),
    )
    parser.add_argument(
        "--output-root",
        default="trained_models/artifacts/regression_smoke/all_models_aleppo",
        help="Root output directory for smoke suites.",
    )
    parser.add_argument(
        "--datasets",
        default="aleppo_2017",
        help="Space-separated dataset list (default: aleppo_2017).",
    )
    parser.add_argument(
        "--config-dir",
        default="configs/data/holdout_smoke_aleppo_ultra",
        help="Holdout config directory.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Epoch override for workflow training step.",
    )
    parser.add_argument(
        "--skip-steps",
        default="7",
        help="Space-separated workflow steps to skip (default: 7 for no resume-training).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Determinism seed exported to workflow environment.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop at first model failure instead of running remaining models.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print per-model commands and write suite manifest without executing workflows.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    repo_root = _repo_root()
    smoke_script_path = (
        repo_root
        / "scripts"
        / "workflows"
        / "forecasting"
        / "forecasting_workflow_regression_smoke.sh"
    )
    if not smoke_script_path.exists():
        parser.error(f"Smoke workflow script not found: {smoke_script_path}")

    selected_models = _parse_models(args.models)
    suite_dir = repo_root / args.output_root / args.suite_label
    suite_dir.mkdir(parents=True, exist_ok=True)

    runs: list[dict[str, Any]] = []
    failures: list[str] = []

    for model_type in selected_models:
        spec = MODEL_SPECS[model_type]
        run_record = _run_single_model(
            repo_root=repo_root,
            smoke_script_path=smoke_script_path,
            suite_label=args.suite_label,
            suite_dir=suite_dir,
            spec=spec,
            datasets=args.datasets,
            config_dir=args.config_dir,
            epochs=args.epochs,
            skip_steps=args.skip_steps,
            seed=args.seed,
            dry_run=args.dry_run,
        )
        runs.append(run_record)

        if run_record["status"] not in {"success", "dry_run"}:
            failures.append(model_type)
            print(
                f"❌ [{model_type}] failed (status={run_record['status']}, "
                f"return_code={run_record['return_code']})"
            )
            if args.fail_fast:
                break
        else:
            print(f"✅ [{model_type}] {run_record['status']}")

    suite_manifest = {
        "schema_version": "1",
        "suite_label": args.suite_label,
        "created_at_utc": _utc_now_iso(),
        "repo_root": str(repo_root),
        "datasets": args.datasets,
        "config_dir": args.config_dir,
        "epochs": args.epochs,
        "skip_steps": args.skip_steps,
        "seed": args.seed,
        "dry_run": args.dry_run,
        "models_requested": selected_models,
        "failed_models": failures,
        "runs": runs,
    }

    suite_manifest_path = suite_dir / "suite_manifest.json"
    with open(suite_manifest_path, "w") as f:
        json.dump(suite_manifest, f, indent=2, sort_keys=True)

    print("\n===================================================================")
    print("Aleppo regression smoke suite complete")
    print("===================================================================")
    print(f"Suite manifest: {suite_manifest_path}")
    if failures:
        print(f"Failed models: {', '.join(failures)}")
        return 1

    print("All models completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
