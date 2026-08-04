# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)
"""A7 -- Rank-based significance across models (Friedman + Wilcoxon-Holm + CD).

Complements A1's bootstrap CIs with the rank-based omnibus + post-hoc pipeline
reviewers expect for multi-model benchmarks (Demsar 2006). The statistical UNIT
is the PATIENT (episodes are clustered within patients), not the episode: we
build a patients x models matrix of per-patient mean RMSE, then:

  1. Friedman test per dataset (models as treatments, patients as blocks) --
     omnibus "do models differ at all?".
  2. Average ranks + Nemenyi critical difference (CD) -> data for a per-dataset
     Critical Difference diagram (camera-ready figure; no images in rebuttal).
  3. Pairwise Wilcoxon signed-rank (over patients) with HOLM correction and a
     matched-pairs rank-biserial EFFECT SIZE (crucial given large N: a tiny
     p-value with a negligible effect is not meaningful).

RMSE is the primary metric (per-patient definable). Detection AUROC/AUPRC are
not per-patient definable (single-class patients) -> use A1 cluster-bootstrap CIs.

Usage:
    python -m scripts.analysis.rebuttal_neurips2026.a7_rank_significance

Outputs (scripts/analysis/rebuttal_neurips2026/outputs/):
    a7_friedman.csv     per dataset: chi2, p, n_patients, avg ranks, CD
    a7_posthoc.csv      per dataset x pair: Wilcoxon stat, p_holm, rank_biserial
    printed CD-diagram-ready average ranks.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon

from scripts.analysis.rebuttal_neurips2026 import common as C

OUT_DIR = Path(__file__).resolve().parent / "outputs"

MODELS = [
    "chronos2",
    "moirai",
    "toto",
    "timesfm",
    "patchtst",
    "tft",
    "ttm",
    "tide",
    "deepar",
    "moment",
]

# Nemenyi q_alpha (alpha=0.05, studentized range / sqrt2, inf df) -- Demsar 2006.
_Q05 = {
    2: 1.960,
    3: 2.343,
    4: 2.569,
    5: 2.728,
    6: 2.850,
    7: 2.949,
    8: 3.031,
    9: 3.102,
    10: 3.164,
    11: 3.219,
    12: 3.268,
    13: 3.313,
    14: 3.354,
    15: 3.391,
}


def _per_patient_rmse(dataset: str, models: list[str]) -> pd.DataFrame:
    """patients x models matrix of per-patient mean RMSE (aligned patients)."""
    cols = {}
    for m in models:
        rp = C.best_run_path(m, dataset)
        if rp is None:
            continue
        df = C.load_run(rp, dataset=dataset, model=m, compute_phypo=False).episodes
        cols[m] = df.groupby("pid")["rmse"].mean()
    mat = pd.DataFrame(cols).dropna()  # patients present for all models
    return mat


def _rank_biserial(a: np.ndarray, b: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation effect size for Wilcoxon.

    r = (sum of positive-rank) - (sum of negative-rank) over total rank sum.
    Sign convention: positive => a > b (a worse if metric is RMSE).
    """
    d = a - b
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = rankdata(np.abs(d))
    rp = r[d > 0].sum()
    rn = r[d < 0].sum()
    return float((rp - rn) / r.sum())


def run(models: list[str]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    C.save_run_index(OUT_DIR / "run_index.csv")

    fried_rows, post_rows = [], []
    for dataset in C.DATASETS:
        mat = _per_patient_rmse(dataset, models)
        present = list(mat.columns)
        k, N = len(present), len(mat)
        if k < 3 or N < 5:
            continue
        chi2, p = friedmanchisquare(*[mat[m].to_numpy() for m in present])
        # average ranks (per patient: rank 1 = lowest RMSE)
        ranks = mat.apply(
            lambda row: rankdata(row.to_numpy()), axis=1, result_type="expand"
        )
        ranks.columns = present
        avg_rank = ranks.mean().sort_values()
        cd = _Q05.get(k, 3.2) * np.sqrt(k * (k + 1) / (6.0 * N))
        fried_rows.append(
            dict(
                dataset=dataset,
                dataset_label=C.DATASET_LABEL[dataset],
                chi2=float(chi2),
                p=float(p),
                n_patients=N,
                k_models=k,
                cd=float(cd),
                avg_ranks=";".join(f"{m}:{avg_rank[m]:.2f}" for m in avg_rank.index),
            )
        )
        # pairwise Wilcoxon + Holm
        pairs = list(itertools.combinations(present, 2))
        raw = []
        for a, b in pairs:
            xa, xb = mat[a].to_numpy(), mat[b].to_numpy()
            try:
                stat, pval = wilcoxon(xa, xb, zero_method="wilcox")
            except ValueError:
                stat, pval = np.nan, 1.0
            raw.append(
                (
                    a,
                    b,
                    stat,
                    pval,
                    _rank_biserial(xa, xb),
                    float(np.mean(xa) - np.mean(xb)),
                )
            )
        # Holm correction
        order = np.argsort([r[3] for r in raw])
        m_tests = len(raw)
        holm = [None] * m_tests
        prev = 0.0
        for rank_i, idx in enumerate(order):
            adj = min(1.0, (m_tests - rank_i) * raw[idx][3])
            adj = max(adj, prev)
            holm[idx] = adj
            prev = adj
        for (a, b, stat, pval, rb, dmean), ph in zip(raw, holm):
            post_rows.append(
                dict(
                    dataset=dataset,
                    dataset_label=C.DATASET_LABEL[dataset],
                    model_a=a,
                    model_b=b,
                    wilcoxon_stat=stat,
                    p_raw=pval,
                    p_holm=ph,
                    rank_biserial=rb,
                    mean_rmse_diff=dmean,
                    significant_holm=bool(ph < 0.05),
                )
            )

    fried = pd.DataFrame(fried_rows)
    post = pd.DataFrame(post_rows)
    fried.to_csv(OUT_DIR / "a7_friedman.csv", index=False)
    post.to_csv(OUT_DIR / "a7_posthoc.csv", index=False)
    _print_summary(fried, post)


def _print_summary(fried: pd.DataFrame, post: pd.DataFrame) -> None:
    print("\n" + "=" * 88)
    print("A7 -- FRIEDMAN + NEMENYI (per-patient RMSE; unit = patient)")
    print("=" * 88)
    for r in fried.itertuples():
        print(
            f"\n### {r.dataset_label}: Friedman chi2={r.chi2:.1f}, p={r.p:.2e} "
            f"(N={r.n_patients} patients, k={r.k_models} models)  CD(0.05)={r.cd:.3f} rank units"
        )
        print("  avg ranks (lower=better):")
        for tok in r.avg_ranks.split(";"):
            print(f"    {tok}")
    print("\n" + "=" * 88)
    print("A7 -- PAIRWISE WILCOXON (Holm-corrected) + rank-biserial effect size")
    print("    only significant pairs shown; mean_rmse_diff = A-B (mmol/L)")
    print("=" * 88)
    for dataset in post.dataset.drop_duplicates() if not post.empty else []:
        sub = post[(post.dataset == dataset) & post.significant_holm]
        if sub.empty:
            continue
        lbl = sub.dataset_label.iloc[0]
        print(
            f"\n### {lbl}  ({len(sub)} significant pairs of "
            f"{len(post[post.dataset==dataset])})"
        )
        for r in (
            sub.sort_values("rank_biserial", key=abs, ascending=False)
            .head(8)
            .itertuples()
        ):
            print(
                f"  {r.model_a:9s} vs {r.model_b:9s} dRMSE={r.mean_rmse_diff:+.3f} "
                f"p_holm={r.p_holm:.1e} rank-biserial={r.rank_biserial:+.3f}"
            )
    print(f"\nSaved: {OUT_DIR}/a7_friedman.csv , a7_posthoc.csv")


def main() -> None:
    ap = argparse.ArgumentParser(description="A7 Friedman + Wilcoxon-Holm + CD")
    ap.add_argument("--models", nargs="+", default=MODELS)
    args = ap.parse_args()
    run(args.models)


if __name__ == "__main__":
    main()
