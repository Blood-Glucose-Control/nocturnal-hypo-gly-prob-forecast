#!/usr/bin/env python3
"""Build the best-runs split-breakdown CSV.

For each (model × dataset × cov_bucket) best run under the requested
holdout ablation (default ``configs/data/holdout_10pct``), re-aggregates
its per-episode metrics separately for the patient-holdout vs
temporal-holdout subsets.

Output: ``<output_dir>/best_runs_split_breakdown_<ablation>[_512ctx].csv``.

Example::

    python scripts/analysis/build_best_split_breakdown.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running from the repo root without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.experiments.nocturnal.grand_summary import DATASETS  # noqa: E402
from src.experiments.nocturnal.grand_summary_split import (  # noqa: E402
    build_best_split_breakdown,
)

DEFAULT_SUMMARIES = (
    "experiments/nocturnal_forecasting/summary.csv",
    "experiments/nocturnal_forecasting_ctx_ablation/summary.csv",
)
DEFAULT_OUTPUT_DIR = "results/grand_summary"
DEFAULT_CONFIG_DIR = "configs/data/holdout_10pct"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary-csv", action="append", default=None)
    p.add_argument("--config-dir", type=str, default=DEFAULT_CONFIG_DIR)
    p.add_argument("--datasets", nargs="+", default=list(DATASETS))
    p.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--forecast-length", type=int, default=96)
    p.add_argument(
        "--context-length",
        type=int,
        default=None,
        help="If set, restrict to runs at this context length (e.g. 512).",
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    summary_paths = args.summary_csv or list(DEFAULT_SUMMARIES)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = build_best_split_breakdown(
        summary_paths=summary_paths,
        config_dir=args.config_dir,
        datasets=args.datasets,
        ctx_filter=args.context_length,
        forecast_filter=args.forecast_length,
    )

    ablation = Path(args.config_dir).name
    suffix = f"_{args.context_length}ctx" if args.context_length else ""
    out_path = out_dir / f"best_runs_split_breakdown_{ablation}{suffix}.csv"
    df.to_csv(out_path, index=False)

    print()
    print("=" * 80)
    print(f"BEST-RUN SPLIT BREAKDOWN (config_dir={args.config_dir})")
    print("=" * 80)
    if df.empty:
        print("(no rows)")
    else:
        cols = [
            "model_class",
            "model",
            "dataset",
            "cov_bucket",
            "split_type",
            "n_patients",
            "n_episodes",
            "n_episodes_with_hypo",
            "rmse",
            "wql",
            "brier_3_9",
        ]
        cols = [c for c in cols if c in df.columns]
        print(df[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nWrote {out_path.resolve()} ({len(df)} rows)")


if __name__ == "__main__":
    main()
