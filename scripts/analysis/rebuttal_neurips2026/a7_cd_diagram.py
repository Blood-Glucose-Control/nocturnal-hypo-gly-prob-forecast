# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)
"""A7-CD -- Critical Difference diagrams (Demsar 2006) from the A7 outputs.

Reads a7_friedman.csv (average ranks + Nemenyi CD per dataset, computed over
PATIENTS) and draws one CD diagram per dataset: models on an average-rank axis,
with horizontal bars connecting cliques of models whose average ranks differ by
less than the critical difference (i.e. NOT significantly different by Nemenyi).

Run A7 first (produces a7_friedman.csv). Figures are for the camera-ready and,
per the ED-track link policy, may be linked anonymously in the rebuttal ONLY in
response to the reviewer's explicit request for significance analysis.

Usage:
    python -m scripts.analysis.rebuttal_neurips2026.a7_cd_diagram

Outputs:
    outputs/figures/cd_<dataset>.png   one CD diagram per dataset
    outputs/figures/cd_all.png          2x2 grid of all four
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


OUT_DIR = Path(__file__).resolve().parent / "outputs"
FIG_DIR = OUT_DIR / "figures"


def _parse_ranks(s: str) -> dict[str, float]:
    return {tok.split(":")[0]: float(tok.split(":")[1]) for tok in s.split(";")}


def _cliques(models_sorted: list[str], ranks: dict[str, float], cd: float):
    """Maximal groups of models whose avg-rank spread <= CD (not sig. different)."""
    cliques = []
    i = 0
    n = len(models_sorted)
    while i < n:
        j = i
        while (
            j + 1 < n and (ranks[models_sorted[j + 1]] - ranks[models_sorted[i]]) <= cd
        ):
            j += 1
        if j > i:  # clique of >=2
            cliques.append((i, j))
        i += 1
    # keep only maximal cliques (drop those contained in another)
    maximal = [
        c
        for c in cliques
        if not any(o != c and o[0] <= c[0] and c[1] <= o[1] for o in cliques)
    ]
    return maximal


def _draw(
    ax,
    dataset_label: str,
    ranks: dict[str, float],
    cd: float,
    chi2: float,
    p: float,
    n_pat: int,
) -> None:
    models = sorted(ranks, key=ranks.get)  # best (lowest rank) first
    k = len(models)
    lo, hi = 1, k
    ax.set_xlim(lo - 0.5, hi + 0.5)
    ax.set_ylim(-(k / 2 + 2), 2.2)
    ax.axis("off")
    # top axis line + ticks
    ax.plot([lo, hi], [0, 0], "k-", lw=1.2)
    for r in range(lo, hi + 1):
        ax.plot([r, r], [0, 0.15], "k-", lw=1)
        ax.text(r, 0.35, str(r), ha="center", va="bottom", fontsize=8)
    ax.text(
        (lo + hi) / 2,
        1.7,
        f"{dataset_label}  (Friedman p={p:.1e}, " f"N={n_pat} pts, CD={cd:.2f})",
        ha="center",
        fontsize=9,
        weight="bold",
    )
    # half split for label placement
    half = (k + 1) // 2
    for idx, m in enumerate(models):
        r = ranks[m]
        if idx < half:  # left labels
            y = -(idx + 1)
            ax.plot([r, r], [0, y], "k-", lw=0.8)
            ax.plot([r, lo - 0.4], [y, y], "k-", lw=0.8)
            ax.text(lo - 0.5, y, f"{m} ({r:.2f})", ha="right", va="center", fontsize=8)
        else:
            y = -(k - idx)
            ax.plot([r, r], [0, y], "k-", lw=0.8)
            ax.plot([r, hi + 0.4], [y, y], "k-", lw=0.8)
            ax.text(hi + 0.5, y, f"{m} ({r:.2f})", ha="left", va="center", fontsize=8)
    # clique bars (not significantly different)
    cl = _cliques(models, ranks, cd)
    yb = -0.25
    for a, b in cl:
        ra, rb = ranks[models[a]], ranks[models[b]]
        ax.plot(
            [ra - 0.05, rb + 0.05],
            [yb, yb],
            "-",
            lw=3.0,
            color="crimson",
            solid_capstyle="round",
        )
        yb -= 0.22
    # CD ruler
    ax.plot([lo, lo + cd], [1.15, 1.15], "k-", lw=1.5)
    ax.plot([lo, lo], [1.08, 1.22], "k-", lw=1)
    ax.plot([lo + cd, lo + cd], [1.08, 1.22], "k-", lw=1)
    ax.text(lo + cd / 2, 1.28, "CD", ha="center", fontsize=7)


def run() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fried = pd.read_csv(OUT_DIR / "a7_friedman.csv")
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, r in zip(axes.ravel(), fried.itertuples()):
        ranks = _parse_ranks(r.avg_ranks)
        _draw(ax, r.dataset_label, ranks, r.cd, r.chi2, r.p, r.n_patients)
        # standalone
        f1, a1 = plt.subplots(figsize=(7, 4.2))
        _draw(a1, r.dataset_label, ranks, r.cd, r.chi2, r.p, r.n_patients)
        f1.tight_layout()
        f1.savefig(FIG_DIR / f"cd_{r.dataset}.png", dpi=150, bbox_inches="tight")
        plt.close(f1)
    fig.suptitle(
        "Critical Difference diagrams — per-patient RMSE ranks "
        "(Nemenyi, alpha=0.05)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(FIG_DIR / "cd_all.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved CD diagrams to {FIG_DIR}/ (cd_all.png + per-dataset).")
    for r in fried.itertuples():
        print(f"  {r.dataset_label}: {FIG_DIR}/cd_{r.dataset}.png")


if __name__ == "__main__":
    run()
