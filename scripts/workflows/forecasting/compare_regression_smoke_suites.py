#!/usr/bin/env python3
"""Compare two Aleppo smoke-suite manifests for artifact and metric regression."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any


def _load_json_dict(path: Path) -> dict[str, Any]:
    with open(path) as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _normalize_relative_paths(paths: list[str], artifact_root: str) -> list[str]:
    normalized: list[str] = []
    for raw_path in paths:
        rel: str
        try:
            rel = os.path.relpath(raw_path, start=artifact_root)
        except ValueError:
            rel = Path(raw_path).name
        normalized.append(rel.replace("\\", "/"))
    return sorted(normalized)


def _extract_numeric_metrics(manifest: dict[str, Any]) -> dict[str, float]:
    payload = manifest.get("key_metrics", {})
    if not isinstance(payload, dict):
        return {}
    return {
        str(key): float(value)
        for key, value in payload.items()
        if isinstance(value, (int, float))
    }


def _compare_metric(
    baseline: float, candidate: float, rel_tol: float, abs_tol: float
) -> tuple[bool, float, float]:
    diff = abs(candidate - baseline)
    allowed = max(abs_tol, rel_tol * max(abs(baseline), 1.0))
    return diff <= allowed, diff, allowed


def _find_nested_run_manifest(output_dir: Path, run_id: str | None) -> Path | None:
    nested_candidates = sorted(
        output_dir.glob("**/run_manifest.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not nested_candidates:
        return None

    if run_id is None:
        return nested_candidates[0]

    for candidate_path in nested_candidates:
        try:
            candidate_manifest = _load_json_dict(candidate_path)
        except Exception:
            continue
        if candidate_manifest.get("run_id") == run_id:
            return candidate_path
    return nested_candidates[0]


def _resolve_run_manifest_path(run_record: dict[str, Any]) -> Path:
    run_id_raw = run_record.get("run_id")
    run_id = run_id_raw if isinstance(run_id_raw, str) and run_id_raw else None
    output_dir_raw = run_record.get("output_dir")
    output_dir = (
        Path(output_dir_raw)
        if isinstance(output_dir_raw, str) and output_dir_raw
        else None
    )

    explicit = run_record.get("run_manifest_path")
    if isinstance(explicit, str) and explicit:
        explicit_path = Path(explicit)
        if explicit_path.exists():
            return explicit_path
        if output_dir is not None:
            nested = _find_nested_run_manifest(output_dir, run_id)
            if nested is not None:
                return nested
        return explicit_path

    if output_dir is not None:
        direct_path = output_dir / "run_manifest.json"
        if direct_path.exists():
            return direct_path
        nested = _find_nested_run_manifest(output_dir, run_id)
        if nested is not None:
            return nested
        return direct_path

    raise ValueError(
        f"Missing run_manifest_path/output_dir in run record: {run_record}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare smoke-suite outputs. Enforces artifact path parity and "
            "metric similarity tolerance."
        )
    )
    parser.add_argument(
        "--baseline", required=True, help="Baseline suite_manifest.json path."
    )
    parser.add_argument(
        "--candidate", required=True, help="Candidate suite_manifest.json path."
    )
    parser.add_argument(
        "--rel-tol",
        type=float,
        default=0.25,
        help="Relative tolerance for key metrics (default: 0.25).",
    )
    parser.add_argument(
        "--abs-tol",
        type=float,
        default=1e-6,
        help="Absolute tolerance floor for key metrics (default: 1e-6).",
    )
    parser.add_argument(
        "--report-path",
        default=None,
        help="Optional output path for comparison report JSON.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    baseline_suite_path = Path(args.baseline)
    candidate_suite_path = Path(args.candidate)
    baseline_suite = _load_json_dict(baseline_suite_path)
    candidate_suite = _load_json_dict(candidate_suite_path)

    baseline_runs = baseline_suite.get("runs", [])
    candidate_runs = candidate_suite.get("runs", [])
    if not isinstance(baseline_runs, list) or not isinstance(candidate_runs, list):
        parser.error("Both suite manifests must contain a 'runs' list.")

    baseline_by_model = {
        str(run.get("model_type")): run
        for run in baseline_runs
        if isinstance(run, dict)
    }
    candidate_by_model = {
        str(run.get("model_type")): run
        for run in candidate_runs
        if isinstance(run, dict)
    }

    baseline_models = set(baseline_by_model)
    candidate_models = set(candidate_by_model)

    failures: list[str] = []
    model_reports: list[dict[str, Any]] = []

    missing_in_candidate = sorted(baseline_models - candidate_models)
    extra_in_candidate = sorted(candidate_models - baseline_models)
    if missing_in_candidate:
        failures.append(
            "Candidate suite is missing models: " + ", ".join(missing_in_candidate)
        )
    if extra_in_candidate:
        failures.append(
            "Candidate suite has extra models: " + ", ".join(extra_in_candidate)
        )

    models_to_compare = sorted(baseline_models & candidate_models)

    for model_type in models_to_compare:
        baseline_record = baseline_by_model[model_type]
        candidate_record = candidate_by_model[model_type]
        model_failure_reasons: list[str] = []

        baseline_manifest_path = _resolve_run_manifest_path(baseline_record)
        candidate_manifest_path = _resolve_run_manifest_path(candidate_record)

        if not baseline_manifest_path.exists():
            model_failure_reasons.append(
                f"missing baseline run_manifest: {baseline_manifest_path}"
            )
        if not candidate_manifest_path.exists():
            model_failure_reasons.append(
                f"missing candidate run_manifest: {candidate_manifest_path}"
            )

        if not model_failure_reasons:
            baseline_manifest = _load_json_dict(baseline_manifest_path)
            candidate_manifest = _load_json_dict(candidate_manifest_path)

            for label, manifest in (
                ("baseline", baseline_manifest),
                ("candidate", candidate_manifest),
            ):
                status = manifest.get("status")
                if status != "success":
                    model_failure_reasons.append(
                        f"{label} manifest status is '{status}', expected 'success'"
                    )

            for key in ("checkpoint_paths", "prediction_paths", "plot_paths"):
                baseline_paths_raw = baseline_manifest.get(key, [])
                candidate_paths_raw = candidate_manifest.get(key, [])
                if not isinstance(baseline_paths_raw, list) or not isinstance(
                    candidate_paths_raw, list
                ):
                    model_failure_reasons.append(f"{key} is not a list in one manifest")
                    continue

                baseline_rel = _normalize_relative_paths(
                    [str(path) for path in baseline_paths_raw],
                    str(baseline_manifest.get("artifact_root", "")),
                )
                candidate_rel = _normalize_relative_paths(
                    [str(path) for path in candidate_paths_raw],
                    str(candidate_manifest.get("artifact_root", "")),
                )

                if baseline_rel != candidate_rel:
                    model_failure_reasons.append(
                        f"{key} mismatch: baseline={len(baseline_rel)} files, "
                        f"candidate={len(candidate_rel)} files"
                    )

            baseline_metrics = _extract_numeric_metrics(baseline_manifest)
            candidate_metrics = _extract_numeric_metrics(candidate_manifest)
            missing_metric_keys = sorted(set(baseline_metrics) - set(candidate_metrics))
            if missing_metric_keys:
                model_failure_reasons.append(
                    "missing candidate key_metrics: " + ", ".join(missing_metric_keys)
                )

            common_metric_keys = sorted(set(baseline_metrics) & set(candidate_metrics))
            metric_diffs: dict[str, dict[str, float | bool]] = {}
            for metric_key in common_metric_keys:
                baseline_value = baseline_metrics[metric_key]
                candidate_value = candidate_metrics[metric_key]
                if math.isnan(baseline_value) or math.isnan(candidate_value):
                    model_failure_reasons.append(f"{metric_key} contains NaN")
                    continue
                is_ok, diff, allowed = _compare_metric(
                    baseline_value, candidate_value, args.rel_tol, args.abs_tol
                )
                metric_diffs[metric_key] = {
                    "baseline": baseline_value,
                    "candidate": candidate_value,
                    "abs_diff": diff,
                    "allowed_diff": allowed,
                    "within_tolerance": is_ok,
                }
                if not is_ok:
                    model_failure_reasons.append(
                        f"metric {metric_key} drifted beyond tolerance: "
                        f"abs_diff={diff:.6g}, allowed={allowed:.6g}"
                    )

            model_reports.append(
                {
                    "model_type": model_type,
                    "baseline_manifest_path": str(baseline_manifest_path),
                    "candidate_manifest_path": str(candidate_manifest_path),
                    "ok": len(model_failure_reasons) == 0,
                    "failures": model_failure_reasons,
                    "metric_diffs": metric_diffs,
                }
            )
        else:
            model_reports.append(
                {
                    "model_type": model_type,
                    "baseline_manifest_path": str(baseline_manifest_path),
                    "candidate_manifest_path": str(candidate_manifest_path),
                    "ok": False,
                    "failures": model_failure_reasons,
                    "metric_diffs": {},
                }
            )

        if model_failure_reasons:
            failures.append(f"{model_type}: " + " | ".join(model_failure_reasons))

    report = {
        "baseline_suite": str(baseline_suite_path),
        "candidate_suite": str(candidate_suite_path),
        "rel_tol": args.rel_tol,
        "abs_tol": args.abs_tol,
        "model_count_compared": len(models_to_compare),
        "ok": len(failures) == 0,
        "failures": failures,
        "models": model_reports,
    }

    if args.report_path:
        report_path = Path(args.report_path)
    else:
        report_path = candidate_suite_path.parent / "comparison_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    print("Smoke suite comparison report:", report_path)
    if failures:
        print("\nComparison failures:")
        for failure in failures:
            print(" -", failure)
        return 1

    print("✅ Smoke suite comparison passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
