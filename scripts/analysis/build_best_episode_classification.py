#!/usr/bin/env python3
"""Build the best-runs episode-classification breakdown.

For every (model x dataset x cov_bucket) best run under a given holdout
ablation, computes pooled AUROC / AUPRC for the binary task

    y_ep = 1{ any forecast-window step has actual BG < 3.9 mmol/L }

using **two** per-episode score variants:

    s_ep^max  = max_t  P(BG_t < 3.9)
    s_ep^mean = mean_t P(BG_t < 3.9)

Output: ``<output_dir>/best_runs_episode_classification_<ablation>[_Nctx].csv``
with parallel ``*_max`` / ``*_mean`` metric columns.

Example::

    python scripts/analysis/build_best_episode_classification.py --context-length 512
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running from the repo root without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.experiments.nocturnal.episode_classification import (  # noqa: E402
    build_best_episode_classification,
)
from src.experiments.nocturnal.grand_summary import DATASETS  # noqa: E402

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

    df = build_best_episode_classification(
        summary_paths=summary_paths,
        config_dir=args.config_dir,
        datasets=args.datasets,
        ctx_filter=args.context_length,
        forecast_filter=args.forecast_length,
    )

    ablation = Path(args.config_dir).name
    suffix = f"_{args.context_length}ctx" if args.context_length else ""
    out_path = out_dir / f"best_runs_episode_classification_{ablation}{suffix}.csv"
    df.to_csv(out_path, index=False)

    print()
    print("=" * 80)
    print(
        f"BEST-RUN EPISODE CLASSIFICATION (any-step hypo<3.9)  "
        f"config_dir={args.config_dir}"
    )
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
            "n_episodes",
            "n_pos",
            "base_rate",
            "auroc_max",
            "auroc_mean",
            "auprc_max",
            "auprc_mean",
            "auprc_skill_max",
            "auprc_skill_mean",
        ]
        cols = [c for c in cols if c in df.columns]
        # Drop the all-NaN deterministic rows from console output (still in CSV).
        view = df.dropna(subset=["auroc_max"])
        if view.empty:
            print("(no probabilistic runs to display)")
        else:
            print(view[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nWrote {out_path.resolve()} ({len(df)} rows)")


if __name__ == "__main__":
    main()
