#!/usr/bin/env python3
"""Aggregate per-run nocturnal evaluations into summary CSVs.

Walks ``experiments/nocturnal_forecasting/`` (and the context-length ablation
tree, if present), writes ``summary.csv`` next to the runs, and prints
best-run leaderboards to stdout.

This is the intermediate step between
``scripts/experiments/nocturnal_hypo_eval.py`` (which writes per-run output)
and ``scripts/analysis/build_grand_summary.py`` (which reads
``experiments/nocturnal_forecasting/summary.csv``).

Usage
-----
Run from the repository root::

    python scripts/analysis/summarize_experiments.py
    python scripts/analysis/summarize_experiments.py --metric mae
    python scripts/analysis/summarize_experiments.py --root /abs/path/to/experiments
"""

from __future__ import annotations

import argparse
import logging

from src.experiments.base.experiment import ExperimentSummarizer, VALID_METRICS
from src.experiments.nocturnal.summarize import NocturnalSummarizer


class _NocturnalCtxAblationSummarizer(NocturnalSummarizer):
    """NocturnalSummarizer scoped to ``experiments/nocturnal_forecasting_ctx_ablation/``."""

    def __init__(self, experiments_root: str = "experiments") -> None:
        ExperimentSummarizer.__init__(
            self, experiments_root, "nocturnal_forecasting_ctx_ablation"
        )


_SUMMARIZERS = {
    "nocturnal_forecasting": NocturnalSummarizer,
    "nocturnal_forecasting_ctx_ablation": _NocturnalCtxAblationSummarizer,
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize nocturnal-forecasting runs and print best-run leaderboards.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--type",
        choices=list(_SUMMARIZERS) + ["all"],
        default="all",
        dest="experiment_type",
        help="Which experiment tree to summarize.",
    )
    parser.add_argument(
        "--metric",
        choices=sorted(VALID_METRICS),
        default="rmse",
        help="Metric used to rank runs (lower is better).",
    )
    parser.add_argument(
        "--root",
        default="experiments",
        help="Path to the top-level experiments/ directory.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    return parser


def _run(name: str, cls: type[ExperimentSummarizer], root: str, metric: str) -> None:
    """Run a single summarizer: write summary.csv + print leaderboards."""
    print(f"\n=== {name} ===")
    summarizer = cls(experiments_root=root)
    df = summarizer.summarize(metric=metric)
    if df.empty:
        print(f"  no runs found under {root}/{name}/")
        return

    print(f"  summary.csv: {len(df)} runs")

    leaderboards = summarizer.best_runs(metric=metric)
    by_md = leaderboards.get("by_model_dataset")
    by_m = leaderboards.get("by_model")
    if by_md is not None and not by_md.empty:
        print(f"\n  Best run per (model × dataset) by {metric}:")
        cols = [c for c in ["model", "dataset", "mode", metric, "run_id"] if c in by_md.columns]
        print(by_md[cols].to_string(index=False))
    if by_m is not None and not by_m.empty:
        print(f"\n  Global best per model by {metric}:")
        cols = [c for c in ["model", "dataset", "mode", metric, "run_id"] if c in by_m.columns]
        print(by_m[cols].to_string(index=False))


def main() -> None:
    args = _build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    targets = (
        list(_SUMMARIZERS.items())
        if args.experiment_type == "all"
        else [(args.experiment_type, _SUMMARIZERS[args.experiment_type])]
    )
    for name, cls in targets:
        _run(name, cls, args.root, args.metric)


if __name__ == "__main__":
    main()
