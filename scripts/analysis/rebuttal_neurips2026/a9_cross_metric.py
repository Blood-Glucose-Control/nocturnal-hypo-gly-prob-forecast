# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)
"""A9 -- Cross-METRIC dominance (the real "no single architecture dominates").

The paper's claim is that no model wins across the FIVE evaluation axes:
point accuracy (RMSE), probabilistic accuracy (WQL), shape (DILATE-shape),
and hypo detection (AUROC, AUPRC). Detection uses the PROBABILISTIC risk score
max_t P(BG_t < 3.9) -- the score behind Table 5a (confirmed by the authors) --
NOT the point score. Point-only models (ttm/moment) have no P(hypo) and are
excluded from the detection axis (as in the paper).

Outputs, per dataset (Tamborlane de-emphasized; pediatric cohort):
  1. per-metric leaderboard with patient-cluster 95% CIs and per-metric winner;
  2. Spearman rank correlation between the RMSE ranking and every other metric
     ranking (low/negative => the metrics disagree on the best model);
  3. DISSOCIATION tests -- paired patient-cluster bootstrap showing the RMSE
     winner (chronos2) is significantly BETTER on RMSE yet significantly WORSE
     on DILATE-shape and detection than Moirai/Toto (a statistical double
     dissociation = no single architecture dominates).

Usage:
    python -m scripts.analysis.rebuttal_neurips2026.a9_cross_metric
    python -m scripts.analysis.rebuttal_neurips2026.a9_cross_metric --n-boot 10000 --datasets all

Outputs (outputs/):
    a9_metric_leaderboard.csv · a9_rank_correlation.csv · a9_dissociation.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

from scripts.analysis.rebuttal_neurips2026 import common as C

OUT_DIR = Path(__file__).resolve().parent / "outputs"

# Probabilistic models only (detection needs P(hypo)); ttm/moment are point-only.
MODELS = ["chronos2", "moirai", "toto", "timesfm", "patchtst", "tft", "tide", "deepar"]
PRIMARY_DS = ["aleppo_2017", "brown_2019", "lynch_2022"]  # Tamborlane de-emphasized

# metric: (column/score, higher_is_better, stat_fn_builder)
SHAPE_COL = "shape_g001"


def _stat(metric: str):
    if metric == "rmse":
        return C.pooled_rmse, False
    if metric == "wql":
        return (lambda d: C.mean_metric(d, "wql")), False
    if metric == "dilate_shape":
        return (lambda d: C.mean_metric(d, SHAPE_COL)), False
    if metric == "auroc":
        return (lambda d: C.auroc(d, "score_prob")), True
    if metric == "auprc":
        return (lambda d: C.auprc(d, "score_prob")), True
    raise ValueError(metric)


METRICS = ["rmse", "wql", "dilate_shape", "auroc", "auprc"]

# Key dissociation comparisons: (A, B). Delta = stat(A) - stat(B).
DISSOC = [("chronos2", "moirai"), ("chronos2", "toto"), ("chronos2", "patchtst")]


def run(models, datasets, n_boot, seed):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    C.save_run_index(OUT_DIR / "run_index.csv")

    lead, corr_rows, diss_rows = [], [], []
    frames = {}

    def get(ds, m):
        k = (ds, m)
        if k not in frames:
            rp = C.best_run_path(m, ds)
            frames[k] = (
                None
                if rp is None
                else C.load_run(rp, dataset=ds, model=m, compute_phypo=True).episodes
            )
        return frames[k]

    for ds in datasets:
        vals = {}  # metric -> {model: point}
        for metric in METRICS:
            fn, hib = _stat(metric)
            vals[metric] = {}
            for m in models:
                df = get(ds, m)
                if df is None:
                    continue
                pt, lo, hi = C.cluster_bootstrap_ci(df, fn, n_boot=n_boot, seed=seed)
                vals[metric][m] = pt
                lead.append(
                    dict(
                        dataset=ds,
                        dataset_label=C.DATASET_LABEL[ds],
                        model=m,
                        metric=metric,
                        point=pt,
                        lo=lo,
                        hi=hi,
                        higher_is_better=hib,
                    )
                )

        # winners + Spearman vs RMSE ranking
        def ranks(metric):
            fn, hib = _stat(metric)
            s = pd.Series(vals[metric])
            return s.rank(ascending=not hib)  # rank 1 = best

        rmse_rank = ranks("rmse")
        for metric in METRICS:
            rk = ranks(metric)
            common = rmse_rank.index.intersection(rk.index)
            rho, p = spearmanr(rmse_rank[common], rk[common])
            fn, hib = _stat(metric)
            winner = (
                pd.Series(vals[metric]).idxmax()
                if hib
                else pd.Series(vals[metric]).idxmin()
            )
            corr_rows.append(
                dict(
                    dataset=ds,
                    dataset_label=C.DATASET_LABEL[ds],
                    metric=metric,
                    winner=winner,
                    spearman_vs_rmse=float(rho),
                    p=float(p),
                )
            )
        # dissociation paired tests (patient-cluster)
        for a, b in DISSOC:
            da, db = get(ds, a), get(ds, b)
            if da is None or db is None:
                continue
            for metric in METRICS:
                fn, hib = _stat(metric)
                d, lo, hi, pv, n = C.paired_cluster_bootstrap_delta(
                    da, db, fn, n_boot=n_boot, seed=seed
                )
                diss_rows.append(
                    dict(
                        dataset=ds,
                        dataset_label=C.DATASET_LABEL[ds],
                        model_a=a,
                        model_b=b,
                        metric=metric,
                        higher_is_better=hib,
                        delta=d,
                        lo=lo,
                        hi=hi,
                        p=pv,
                        n=n,
                        significant=bool(lo > 0 or hi < 0),
                    )
                )

    lead_df = pd.DataFrame(lead)
    corr_df = pd.DataFrame(corr_rows)
    diss_df = pd.DataFrame(diss_rows)
    lead_df.to_csv(OUT_DIR / "a9_metric_leaderboard.csv", index=False)
    corr_df.to_csv(OUT_DIR / "a9_rank_correlation.csv", index=False)
    diss_df.to_csv(OUT_DIR / "a9_dissociation.csv", index=False)
    _summary(lead_df, corr_df, diss_df, datasets)


def _summary(lead, corr, diss, datasets):
    print("\n" + "=" * 92)
    print("A9 -- CROSS-METRIC WINNERS (best-condition; detection = P(hypo) score)")
    print("=" * 92)
    for ds in datasets:
        c = corr[corr.dataset == ds]
        if c.empty:
            continue
        w = {r.metric: r.winner for r in c.itertuples()}
        print(f"\n### {C.DATASET_LABEL[ds]}")
        print("  winners: " + " | ".join(f"{m}={w.get(m,'?')}" for m in METRICS))
        print(
            "  Spearman(rank vs RMSE): "
            + " | ".join(
                f"{r.metric}={r.spearman_vs_rmse:+.2f}" for r in c.itertuples()
            )
        )
    print("\n" + "=" * 92)
    print(
        "A9 -- DISSOCIATION (paired patient-cluster; delta=A-B; RMSE/WQL/DILATE lower=better, AUROC/AUPRC higher=better)"
    )
    print("=" * 92)
    for ds in datasets:
        s = diss[diss.dataset == ds]
        if s.empty:
            continue
        print(f"\n### {C.DATASET_LABEL[ds]}")
        for (a, b), g in s.groupby(["model_a", "model_b"]):
            cells = []
            for r in g.itertuples():
                mark = "*" if r.significant else " "
                cells.append(f"{r.metric}:{r.delta:+.3f}{mark}")
            print(f"  {a} vs {b}: " + " ".join(cells))
    print(
        "\n  (* = 95% CI excludes 0.  chronos2 sig BETTER on RMSE/WQL but sig WORSE on DILATE/AUROC => no dominance.)"
    )
    print(
        f"\nSaved: {OUT_DIR}/a9_metric_leaderboard.csv, a9_rank_correlation.csv, a9_dissociation.csv"
    )


def main():
    ap = argparse.ArgumentParser(description="A9 cross-metric dominance + significance")
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument(
        "--datasets", nargs="+", default=PRIMARY_DS, help="'all' to include Tamborlane"
    )
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    ds = C.DATASETS if args.datasets == ["all"] else args.datasets
    run(args.models, ds, n_boot=args.n_boot, seed=args.seed)


if __name__ == "__main__":
    main()
