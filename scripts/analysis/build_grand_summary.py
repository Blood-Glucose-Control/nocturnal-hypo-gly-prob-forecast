#!/usr/bin/env python3
"""Build the grand summary tables (model properties + wide results matrix).

Writes the following files under ``--output-dir``:
    * model_properties.csv
    * results_table_512ctx.csv          (main: context_length == 512)
    * results_table_best_ctx.csv        (best across all context lengths)
    * best_runs_long_512ctx.csv
    * best_runs_long_best_ctx.csv
    * missing_combinations_512ctx.csv
    * missing_combinations_best_ctx.csv

Also prints a markdown rendering of the 512-ctx table to stdout.

Example::

    python scripts/analysis/build_grand_summary.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Allow running from the repo root without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.experiments.nocturnal.grand_summary import (  # noqa: E402
    COV_BUCKET_LABELS,
    DATASETS,
    DEFAULT_METRICS,
    MODEL_CLASS_LABELS,
    VALID_COVARIATE_BUCKETS_BY_DATASET,
    build_grand_summary,
)

DEFAULT_SUMMARIES = (
    "experiments/nocturnal_forecasting/summary.csv",
    "experiments/nocturnal_forecasting_ctx_ablation/summary.csv",
)
DEFAULT_OUTPUT_DIR = "results/grand_summary"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--summary-csv",
        action="append",
        default=None,
        help=(
            "Path to a summary.csv (repeatable). "
            f"Default: {' + '.join(DEFAULT_SUMMARIES)}"
        ),
    )
    p.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    p.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASETS),
        help="Datasets to include as super-columns.",
    )
    p.add_argument(
        "--metrics",
        nargs="+",
        default=list(DEFAULT_METRICS),
        help="Metric column order under each dataset.",
    )
    p.add_argument(
        "--forecast-length",
        type=int,
        default=96,
        help="Filter rows to this forecast horizon (default: 96).",
    )
    return p.parse_args()


def _flatten_columns(wide: pd.DataFrame) -> pd.DataFrame:
    """Turn the (dataset, metric) MultiIndex columns into 'dataset__metric' for CSV."""
    flat = wide.copy()
    flat.columns = [f"{ds}__{m}" for ds, m in wide.columns]
    return flat.reset_index()


def _to_markdown(wide: pd.DataFrame, metrics: list[str]) -> str:
    """Render the wide table as grouped markdown — one section per model class."""
    if wide.empty:
        return "_(no rows)_"

    lines: list[str] = []
    datasets = list(dict.fromkeys(ds for ds, _ in wide.columns))

    # Header rows
    super_header = "| Model | Variant | " + " | ".join(
        f"**{ds}**" + " |" * (len(metrics) - 1) for ds in datasets
    )
    sub_header = "| | | " + " | ".join(metrics * len(datasets)) + " |"
    sep = "|---|---|" + "---|" * (len(metrics) * len(datasets))
    lines.append(
        super_header + " |" if not super_header.endswith("|") else super_header
    )
    lines.append(sub_header)
    lines.append(sep)

    last_class = None
    for (cls, model, bucket), row in wide.iterrows():
        if cls != last_class:
            lines.append(
                f"| **{MODEL_CLASS_LABELS.get(cls, cls)}** | | "
                + " | ".join([""] * (len(metrics) * len(datasets)))
                + " |"
            )
            last_class = cls
        cells = []
        for ds in datasets:
            for m in metrics:
                v = row.get((ds, m), float("nan"))
                cells.append("—" if pd.isna(v) else f"{v:.3f}")
        lines.append(
            f"| {model} | {COV_BUCKET_LABELS.get(bucket, bucket)} | "
            + " | ".join(cells)
            + " |"
        )
    return "\n".join(lines)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    summary_paths = args.summary_csv or list(DEFAULT_SUMMARIES)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Variant 1: 512-ctx fair-comparison table
    # ------------------------------------------------------------------
    res_512 = build_grand_summary(
        summary_paths=summary_paths,
        datasets=args.datasets,
        metrics=args.metrics,
        ctx_filter=512,
        forecast_filter=args.forecast_length,
    )
    _flatten_columns(res_512["results_wide"]).to_csv(
        out_dir / "results_table_512ctx.csv", index=False
    )
    res_512["best_runs_long"].to_csv(out_dir / "best_runs_long_512ctx.csv", index=False)
    res_512["missing_combinations"].to_csv(
        out_dir / "missing_combinations_512ctx.csv", index=False
    )
    res_512["misplaced_combinations"].to_csv(
        out_dir / "misplaced_combinations_512ctx.csv", index=False
    )
    res_512["model_properties"].to_csv(out_dir / "model_properties.csv", index=False)

    # ------------------------------------------------------------------
    # Variant 2: best-across-context-lengths
    # ------------------------------------------------------------------
    res_best = build_grand_summary(
        summary_paths=summary_paths,
        datasets=args.datasets,
        metrics=args.metrics,
        ctx_filter=None,
        forecast_filter=args.forecast_length,
    )
    _flatten_columns(res_best["results_wide"]).to_csv(
        out_dir / "results_table_best_ctx.csv", index=False
    )
    res_best["best_runs_long"].to_csv(
        out_dir / "best_runs_long_best_ctx.csv", index=False
    )
    res_best["missing_combinations"].to_csv(
        out_dir / "missing_combinations_best_ctx.csv", index=False
    )
    res_best["misplaced_combinations"].to_csv(
        out_dir / "misplaced_combinations_best_ctx.csv", index=False
    )

    # ------------------------------------------------------------------
    # Console report
    # ------------------------------------------------------------------
    print()
    print("=" * 80)
    print(
        f"GRAND RESULTS TABLE — context_length=512, forecast_length={args.forecast_length}"
    )
    print("=" * 80)
    print(_to_markdown(res_512["results_wide"], args.metrics))
    print()
    print("=" * 80)
    print(f"MISSING COMBINATIONS (512-ctx, fh={args.forecast_length})")
    print("=" * 80)
    miss = res_512["missing_combinations"]
    if miss.empty:
        print("  (none — every capable model × dataset × bucket has a run)")
    else:
        for cls, sub in miss.groupby("model_class", sort=False):
            print(f"\n  [{MODEL_CLASS_LABELS.get(cls, cls)}]")
            for _, r in sub.iterrows():
                print(f"    - {r['model']:<32} {r['dataset']:<18} {r['cov_bucket']}")

    print()
    print("=" * 80)
    print(f"MISPLACED COMBINATIONS (512-ctx, fh={args.forecast_length})")
    print("(runs recorded in a bucket not valid for that dataset)")
    print(
        f"  Dataset covariate availability: { {k: sorted(v) for k, v in VALID_COVARIATE_BUCKETS_BY_DATASET.items()} }"
    )
    print("=" * 80)
    mispl = res_512["misplaced_combinations"]
    if mispl.empty:
        print("  (none — all covariate runs are in the correct dataset bucket)")
    else:
        for cls, sub in mispl.groupby("model_class", sort=False):
            print(f"\n  [{MODEL_CLASS_LABELS.get(cls, cls)}]")
            for _, r in sub.iterrows():
                print(
                    f"    ! {r['model']:<32} {r['dataset']:<18} "
                    f"{r['cov_bucket']:<12} covariates=({r['cov_variant']})"
                )
    print()
    print(f"Outputs written to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
