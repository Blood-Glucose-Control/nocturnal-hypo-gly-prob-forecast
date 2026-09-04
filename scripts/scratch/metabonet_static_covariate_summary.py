#!/usr/bin/env python3
# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATIC_PATH = (
    PROJECT_ROOT / "cache/data/metabonet/processed/static_covariates.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "cache/data/metabonet/processed/analysis"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate summary artifacts for Metabonet static covariates. "
            "Always writes pandas-based CSV/JSON outputs; optionally writes HTML profiling."
        )
    )
    parser.add_argument(
        "--static-path",
        type=Path,
        default=DEFAULT_STATIC_PATH,
        help=f"Path to static_covariates.csv (default: {DEFAULT_STATIC_PATH})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for summary outputs (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of top values to keep in each column profile (default: 5).",
    )
    parser.add_argument(
        "--html-profile",
        action="store_true",
        help=(
            "Also generate HTML profiling report via ydata-profiling "
            "(requires the package to be installed)."
        ),
    )
    parser.add_argument(
        "--html-output",
        type=Path,
        default=None,
        help="Optional explicit path for HTML profile output.",
    )
    return parser.parse_args()


def _to_json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    if pd.isna(value):
        return None
    return str(value)


def build_column_profile(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    total_rows = len(df)
    for column in df.columns:
        series = df[column]
        non_null = series.dropna()
        missing_count = int(series.isna().sum())
        non_null_count = int(non_null.shape[0])
        unique_non_null_count = int(non_null.nunique(dropna=True))

        top_counts = non_null.value_counts(dropna=True).head(top_n)
        dominant_value = (
            _to_json_value(top_counts.index[0]) if not top_counts.empty else None
        )
        dominant_count = int(top_counts.iloc[0]) if not top_counts.empty else 0
        dominant_pct = (
            (dominant_count / non_null_count) * 100.0 if non_null_count > 0 else 0.0
        )

        rows.append(
            {
                "column": column,
                "dtype": str(series.dtype),
                "rows": total_rows,
                "non_null_count": non_null_count,
                "missing_count": missing_count,
                "missing_pct": (missing_count / total_rows) * 100.0
                if total_rows > 0
                else 0.0,
                "unique_non_null_count": unique_non_null_count,
                "dominant_value": dominant_value,
                "dominant_count": dominant_count,
                "dominant_pct_non_null": dominant_pct,
                "top_values_json": json.dumps(
                    {
                        _to_json_value(key): int(value)
                        for key, value in top_counts.to_dict().items()
                    },
                    sort_keys=True,
                ),
            }
        )

    return pd.DataFrame(rows).sort_values(by=["missing_pct", "column"], ascending=False)


def write_summary_artifacts(
    *,
    static_path: Path,
    output_dir: Path,
    top_n: int,
) -> tuple[Path, Path]:
    if top_n <= 0:
        raise ValueError("--top-n must be a positive integer.")
    if not static_path.exists():
        raise FileNotFoundError(f"Static covariates file not found: {static_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(static_path, low_memory=False)
    profile_df = build_column_profile(df, top_n=top_n)

    profile_csv_path = output_dir / "static_covariate_column_profile.csv"
    profile_df.to_csv(profile_csv_path, index=False)

    summary_json_path = output_dir / "static_covariate_table_summary.json"
    summary_payload = {
        "static_path": str(static_path),
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "patient_count": int(df["patient_id"].nunique())
        if "patient_id" in df
        else None,
        "columns": list(df.columns),
    }
    summary_json_path.write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return profile_csv_path, summary_json_path


def write_html_profile(
    *,
    static_path: Path,
    html_output_path: Path,
) -> Path:
    try:
        from ydata_profiling import ProfileReport
    except ImportError as exc:
        raise ImportError(
            "ydata-profiling is not installed. Install it first (e.g., pip install ydata-profiling) "
            "or run without --html-profile."
        ) from exc

    df = pd.read_csv(static_path, low_memory=False)
    profile = ProfileReport(
        df,
        title="Metabonet Static Covariates Profile",
        minimal=True,
    )
    html_output_path.parent.mkdir(parents=True, exist_ok=True)
    profile.to_file(str(html_output_path))
    return html_output_path


def main() -> None:
    args = parse_args()
    profile_csv_path, summary_json_path = write_summary_artifacts(
        static_path=args.static_path,
        output_dir=args.output_dir,
        top_n=args.top_n,
    )
    print(f"Wrote column profile: {profile_csv_path}")
    print(f"Wrote table summary: {summary_json_path}")

    if args.html_profile:
        html_output_path = (
            args.html_output
            if args.html_output is not None
            else args.output_dir / "static_covariates_profile.html"
        )
        written_html = write_html_profile(
            static_path=args.static_path,
            html_output_path=html_output_path,
        )
        print(f"Wrote HTML profile: {written_html}")


if __name__ == "__main__":
    main()
