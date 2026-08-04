# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)
"""A6 -- Episode risk-aggregation ablation (max vs mean vs noisy-or).

R2 Q2 asks whether taking the MAX risk across the 8 h horizon makes episode-level
detection over-sensitive to isolated low-probability spikes, versus mean or product
aggregation. We recompute episode-level any-step-hypo detection (AUROC / AUPRC)
under three aggregations of the per-step risk:

  probabilistic score  p_t = P(BG_t < 3.9):
      max      : max_t p_t                 (the paper's default)
      mean     : mean_t p_t
      noisy_or : 1 - prod_t (1 - p_t)      (probabilistic "any step" union)
  deterministic point score  (comparable for ALL models incl. point-only TTM):
      max_point  : -min_t mu_t             (Table 5a default)
      mean_point : -mean_t mu_t

If max is not systematically worse than mean/noisy-or, the max aggregation is
robust (not spike-driven), which is the answer R2 is probing.

Usage:
    python -m scripts.analysis.rebuttal_neurips2026.a6_aggregation

Outputs (scripts/analysis/rebuttal_neurips2026/outputs/):
    a6_aggregation.csv   one row per (dataset, model, aggregation)
    printed markdown-ready summary.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from scripts.analysis.rebuttal_neurips2026 import common as C

OUT_DIR = Path(__file__).resolve().parent / "outputs"

DEFAULT_MODELS = [
    "chronos2",
    "moirai",
    "toto",
    "patchtst",
    "tft",
    "timesfm",
    "ttm",
    "deepar",
]


def _per_step_phypo(qf: np.ndarray, ql: np.ndarray) -> np.ndarray:
    """(N,H) per-step P(BG<3.9) from quantile forecasts (monotone-CDF interp)."""
    n, _, h = qf.shape
    out = np.empty((n, h), dtype=np.float64)
    for i in range(n):
        out[i] = C._p_hypo_by_step(qf[i], ql)
    return out


def _auc_pair(y: np.ndarray, s: np.ndarray) -> tuple[float, float]:
    if y.min() == y.max() or np.all(np.isnan(s)):
        return float("nan"), float("nan")
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))


def run(models: list[str]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    C.save_run_index(OUT_DIR / "run_index.csv")

    rows = []
    for dataset in C.DATASETS:
        for model in models:
            rp = C.best_run_path(model, dataset)
            if rp is None or not rp.exists():
                continue
            r = C.load_run(rp, dataset=dataset, model=model)
            y = r.episodes["has_hypo"].to_numpy().astype(int)
            base = dict(
                dataset=dataset,
                dataset_label=C.DATASET_LABEL[dataset],
                model=model,
                n=len(y),
                n_hypo=int(y.sum()),
            )

            # deterministic point aggregations (all models)
            mu = r.predictions  # (N,H)
            for agg, s in [
                ("max_point", -mu.min(axis=1)),
                ("mean_point", -mu.mean(axis=1)),
            ]:
                au, ap = _auc_pair(y, s)
                rows.append({**base, "aggregation": agg, "auroc": au, "auprc": ap})

            # probabilistic aggregations (models with quantiles)
            if r.quantile_forecasts is not None and r.quantile_levels is not None:
                p = _per_step_phypo(r.quantile_forecasts, r.quantile_levels)  # (N,H)
                aggs = {
                    "max": p.max(axis=1),
                    "mean": p.mean(axis=1),
                    "noisy_or": 1.0 - np.prod(1.0 - p, axis=1),
                }
                for agg, s in aggs.items():
                    au, ap = _auc_pair(y, s)
                    rows.append({**base, "aggregation": agg, "auroc": au, "auprc": ap})

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "a6_aggregation.csv", index=False)
    _print_summary(out)


def _print_summary(out: pd.DataFrame) -> None:
    print("\n" + "=" * 88)
    print("A6 -- RISK-AGGREGATION ABLATION (episode any-step hypo detection)")
    print(
        "    prob: max / mean / noisy_or of P(BG_t<3.9);  point: max=-min mu, mean=-mean mu"
    )
    print("=" * 88)
    for dataset in C.DATASETS:
        sub = out[out.dataset == dataset]
        if sub.empty:
            continue
        print(f"\n### {C.DATASET_LABEL[dataset]}  (n={int(sub['n'].iloc[0])})")
        for model in sub["model"].drop_duplicates():
            cells = []
            for _, r in sub[sub.model == model].iterrows():
                cells.append(
                    f"{r['aggregation']}: AUROC {r['auroc']:.3f} AUPRC {r['auprc']:.3f}"
                )
            print(f"  {model:9s} " + " | ".join(cells))
    # headline: mean delta (max - mean) across datasets/models for prob score
    prob = out[out.aggregation.isin(["max", "mean", "noisy_or"])]
    if not prob.empty:
        piv = prob.pivot_table(
            index=["dataset", "model"], columns="aggregation", values="auroc"
        )
        if {"max", "mean"}.issubset(piv.columns):
            d = (piv["max"] - piv["mean"]).dropna()
            print(
                f"\nAUROC(max) - AUROC(mean): mean={d.mean():+.4f}, "
                f"min={d.min():+.4f}, max={d.max():+.4f}, n={len(d)} "
                f"(>0 => max aggregation is not worse than mean)"
            )
        if {"max", "noisy_or"}.issubset(piv.columns):
            d2 = (piv["max"] - piv["noisy_or"]).dropna()
            print(
                f"AUROC(max) - AUROC(noisy_or): mean={d2.mean():+.4f}, "
                f"min={d2.min():+.4f}, max={d2.max():+.4f}, n={len(d2)}"
            )
    print(f"\nSaved: {OUT_DIR}/a6_aggregation.csv")


def main() -> None:
    ap = argparse.ArgumentParser(description="A6 aggregation ablation")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    args = ap.parse_args()
    run(args.models)


if __name__ == "__main__":
    main()
