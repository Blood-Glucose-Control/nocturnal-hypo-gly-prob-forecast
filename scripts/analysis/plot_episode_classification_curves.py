#!/usr/bin/env python3
# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)

"""Plot ROC and PR curves for the "any-step hypo" episode classification task.

Two figures are produced:
  * ``episode_classification_roc_<ablation>[_Nctx].png``
  * ``episode_classification_pr_<ablation>[_Nctx].png``

Layout: rows = probabilistic model (one curve set per model), columns = dataset.
Per cell, four curves are overlaid: two splits (``patient`` / ``temporal``) ×
two per-episode score variants (``max`` / ``mean``). Split is encoded by
colour, score variant by linestyle (solid = max, dashed = mean).

For each (model × dataset) we pick the ``cov_bucket`` with the highest mean
``auroc_max`` across the two splits in the breakdown CSV produced by
``scripts/analysis/build_best_episode_classification.py``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.experiments.nocturnal.episode_classification import (  # noqa: E402
    load_run_classification_data,
)
from src.experiments.nocturnal.grand_summary import DATASETS  # noqa: E402

DEFAULT_BREAKDOWN = (
    "results/grand_summary/best_runs_episode_classification_holdout_10pct_512ctx.csv"
)
DEFAULT_OUTPUT_DIR = "results/figures"
DEFAULT_CONFIG_DIR = "configs/data/holdout_10pct"

_SPLIT_COLOURS = {"patient": "tab:blue", "temporal": "tab:orange"}
_VARIANT_LINESTYLES = {"max": "-", "mean": "--"}
_VARIANTS = ("max", "mean")
_SPLITS = ("patient", "temporal")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--breakdown-csv", type=str, default=DEFAULT_BREAKDOWN)
    p.add_argument("--config-dir", type=str, default=DEFAULT_CONFIG_DIR)
    p.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--datasets", nargs="+", default=list(DATASETS))
    p.add_argument(
        "--context-suffix",
        type=str,
        default="512ctx",
        help="Suffix used in the output filename (e.g. '512ctx').",
    )
    return p.parse_args()


def _pick_best_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """For each (model, dataset), keep the cov_bucket with highest mean ``auroc_max``."""
    df = df.dropna(subset=["auroc_max"]).copy()
    mean_auroc = (
        df.groupby(["model", "dataset", "cov_bucket"])["auroc_max"]
        .mean()
        .reset_index(name="mean_auroc")
    )
    idx = mean_auroc.groupby(["model", "dataset"])["mean_auroc"].idxmax()
    best = mean_auroc.loc[idx, ["model", "dataset", "cov_bucket"]]
    return df.merge(best, on=["model", "dataset", "cov_bucket"], how="inner")


def _load_curves(rows: pd.DataFrame, config_dir: str) -> dict:
    """Load (fpr, tpr) and (precision, recall) per (model, dataset, split, variant)."""
    out: dict = {}
    cache: dict[str, pd.DataFrame] = {}
    for _, r in rows.iterrows():
        run_path = str(r["run_path"])
        if run_path not in cache:
            try:
                cache[run_path] = load_run_classification_data(
                    run_path,
                    config_dir=config_dir,
                    dataset=str(r["dataset"]),
                )
            except (FileNotFoundError, ValueError) as exc:
                logging.warning("Skipping %s: %s", run_path, exc)
                cache[run_path] = pd.DataFrame()
                continue
        df = cache[run_path]
        if df.empty:
            continue
        sub = df[df["split_type"] == r["split_type"]]
        y = sub["has_hypo"].to_numpy()
        for variant in _VARIANTS:
            s = sub[f"{variant}_phat"].to_numpy()
            valid = ~np.isnan(s)
            if not valid.any() or y[valid].sum() in (0, valid.sum()):
                continue
            fpr, tpr, _ = roc_curve(y[valid], s[valid])
            prec, rec, _ = precision_recall_curve(y[valid], s[valid])
            out[(r["model"], r["dataset"], r["split_type"], variant)] = {
                "fpr": fpr,
                "tpr": tpr,
                "precision": prec,
                "recall": rec,
                "base_rate": float(y[valid].mean()),
                "auroc": float(r[f"auroc_{variant}"]),
                "auprc": float(r[f"auprc_{variant}"]),
                "cov_bucket": str(r["cov_bucket"]),
            }
    return out


def _make_grid(curves: dict, models: list[str], datasets: list[str], kind: str):
    n_rows, n_cols = len(models), len(datasets)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.6 * n_cols + 1.0, 2.4 * n_rows + 0.8),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    for i, model in enumerate(models):
        for j, dataset in enumerate(datasets):
            ax = axes[i][j]
            cell_curves = {
                (split, variant): curves.get((model, dataset, split, variant))
                for split in _SPLITS
                for variant in _VARIANTS
            }
            cov_bucket = next(
                (c["cov_bucket"] for c in cell_curves.values() if c is not None),
                None,
            )
            base_rate = next(
                (c["base_rate"] for c in cell_curves.values() if c is not None),
                None,
            )
            if kind == "roc":
                ax.plot([0, 1], [0, 1], color="grey", lw=0.6, ls=":")
            elif kind == "pr" and base_rate is not None:
                ax.axhline(base_rate, color="grey", lw=0.6, ls=":")
            for (split, variant), c in cell_curves.items():
                if c is None:
                    continue
                colour = _SPLIT_COLOURS[split]
                ls = _VARIANT_LINESTYLES[variant]
                if kind == "roc":
                    ax.plot(
                        c["fpr"],
                        c["tpr"],
                        color=colour,
                        lw=1.2,
                        ls=ls,
                        label=f"{split} {variant} (AUC={c['auroc']:.2f})",
                    )
                else:
                    ax.plot(
                        c["recall"],
                        c["precision"],
                        color=colour,
                        lw=1.2,
                        ls=ls,
                        label=f"{split} {variant} (AP={c['auprc']:.2f})",
                    )
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.tick_params(labelsize=7)
            if i == 0:
                ax.set_title(dataset, fontsize=9)
            if j == 0:
                ax.set_ylabel(
                    f"{model}\n{'TPR' if kind == 'roc' else 'precision'}",
                    fontsize=8,
                )
            if i == n_rows - 1:
                ax.set_xlabel("FPR" if kind == "roc" else "recall", fontsize=8)
            if cov_bucket is not None:
                ax.text(
                    0.98,
                    0.02,
                    cov_bucket,
                    transform=ax.transAxes,
                    fontsize=6,
                    ha="right",
                    va="bottom",
                    color="dimgrey",
                )
            if any(c is not None for c in cell_curves.values()):
                ax.legend(
                    fontsize=6, loc="upper left" if kind == "roc" else "upper right"
                )
            else:
                ax.set_facecolor("#f7f7f7")
    fig.suptitle(
        "Episode-level any-step hypoglycemia "
        + ("ROC curves" if kind == "roc" else "precision–recall curves"),
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    df = pd.read_csv(args.breakdown_csv)
    if "run_path" not in df.columns:
        raise SystemExit(
            f"{args.breakdown_csv} missing run_path column; rebuild with the "
            "current build_best_episode_classification.py."
        )
    df = df[df["dataset"].isin(args.datasets)]
    rows = _pick_best_buckets(df)
    if rows.empty:
        raise SystemExit("No probabilistic rows after filtering.")

    curves = _load_curves(rows, args.config_dir)
    if not curves:
        raise SystemExit("No curves produced.")

    models = sorted({m for (m, _, _, _) in curves})
    datasets = [
        d
        for d in args.datasets
        if any(
            (m, d, s, v) in curves for m in models for s in _SPLITS for v in _VARIANTS
        )
    ]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.context_suffix}" if args.context_suffix else ""

    for kind in ("roc", "pr"):
        fig = _make_grid(curves, models, datasets, kind)
        out_path = out_dir / f"episode_classification_{kind}{suffix}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {out_path.resolve()}")


if __name__ == "__main__":
    main()
