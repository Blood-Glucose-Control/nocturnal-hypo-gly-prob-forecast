# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)
"""A1 -- Bootstrap confidence intervals + paired significance tests.

Answers reviewer requests for uncertainty quantification on the benchmark's
headline claims (R3 W2/Q2: "no statistical significance analysis"; R2:
"differences are intuitive / not established").

For each dataset we load the canonical best run per model (selected by a
filesystem scan of the on-disk runs -- lowest overall_rmse per (model,dataset),
which reproduces the paper's best-condition cells) and compute, over the pooled
episode set, a percentile bootstrap 95% CI (resampling episodes) for:
    - pooled RMSE           (point-forecast accuracy)
    - mean WQL              (probabilistic accuracy)
    - AUROC / AUPRC         (episode-level any-step hypo detection,
                             score = P(hypo) = max_t P(BG_t < 3.9), matching Table 5a)

It then runs paired bootstraps (shared episodes) for the key head-to-head
comparisons that underpin the paper's conclusions.

Usage:
    python -m scripts.analysis.rebuttal_neurips2026.a1_significance             # default
    python -m scripts.analysis.rebuttal_neurips2026.a1_significance --n-boot 10000
    python -m scripts.analysis.rebuttal_neurips2026.a1_significance --quick     # n_boot=500

Outputs (under scripts/analysis/rebuttal_neurips2026/outputs/):
    run_index.csv          provenance: the exact runs selected by the scan
    a1_metric_cis.csv      one row per (dataset, model, metric)
    a1_paired_tests.csv    one row per (dataset, comparison, metric)
    printed markdown-ready summary for pasting into the rebuttal.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts.analysis.rebuttal_neurips2026 import common as C

OUT_DIR = Path(__file__).resolve().parent / "outputs"

# Curated model set covering every headline claim (attention leaders + shape/
# detection leaders + representative DL/foundation baselines).
DEFAULT_MODELS = [
    "chronos2",
    "tft",
    "patchtst",
    "timesfm",
    "ttm",  # attention / RMSE-WQL cluster
    "moirai",
    "toto",  # shape / detection leaders
    "deepar",
    "tide",  # DL baselines
]

# Key head-to-heads: (name, model_a, model_b, metric, score_col). Delta = A - B.
# For RMSE/WQL lower is better; for AUROC/AUPRC higher is better.
PAIRED = [
    ("chronos2_vs_patchtst_RMSE", "chronos2", "patchtst", "rmse", None),
    ("moirai_vs_chronos2_AUROC", "moirai", "chronos2", "auroc", "score_prob"),
    ("toto_vs_chronos2_AUROC", "toto", "chronos2", "auroc", "score_prob"),
    ("moirai_vs_chronos2_AUPRC", "moirai", "chronos2", "auprc", "score_prob"),
    ("patchtst_vs_chronos2_AUROC", "patchtst", "chronos2", "auroc", "score_prob"),
]


def _stat_fn(metric: str, score_col: str | None):
    if metric == "rmse":
        return C.pooled_rmse
    if metric == "wql":
        return lambda d: C.mean_metric(d, "wql")
    if metric == "auroc":
        return lambda d: C.auroc(d, score_col or "score_point")
    if metric == "auprc":
        return lambda d: C.auprc(d, score_col or "score_point")
    raise ValueError(metric)


def run(models: list[str], n_boot: int, seed: int, unit: str = "patient") -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Provenance: persist the exact run selection used for this analysis.
    C.save_run_index(OUT_DIR / "run_index.csv")

    ci_fn = C.cluster_bootstrap_ci if unit == "patient" else C.bootstrap_ci
    delta_fn = (
        C.paired_cluster_bootstrap_delta
        if unit == "patient"
        else C.paired_bootstrap_delta
    )
    tag = "patient" if unit == "patient" else "episode"

    # cache loaded episode frames per (dataset, model)
    frames: dict[tuple[str, str], pd.DataFrame] = {}

    def get_df(dataset: str, model: str) -> pd.DataFrame | None:
        key = (dataset, model)
        if key in frames:
            return frames[key]
        rp = C.best_run_path(model, dataset)
        if rp is None or not rp.exists():
            frames[key] = None
            return None
        # compute_phypo=True so AUROC/AUPRC use the P(hypo) detection score (Table 5a).
        frames[key] = C.load_run(
            rp, dataset=dataset, model=model, compute_phypo=True
        ).episodes
        return frames[key]

    # ---- Table 1: per-model metric CIs ----
    ci_rows = []
    for dataset in C.DATASETS:
        for model in models:
            df = get_df(dataset, model)
            if df is None:
                continue
            for metric, score_col in [
                ("rmse", None),
                ("wql", None),
                ("auroc", "score_prob"),
                ("auprc", "score_prob"),
            ]:
                # Detection metrics require the probabilistic P(hypo) score; skip
                # point-only models (e.g. ttm/moment) that have no quantiles.
                if score_col == "score_prob" and df["score_prob"].isna().all():
                    continue
                pt, lo, hi = ci_fn(
                    df, _stat_fn(metric, score_col), n_boot=n_boot, seed=seed
                )
                ci_rows.append(
                    dict(
                        dataset=dataset,
                        dataset_label=C.DATASET_LABEL[dataset],
                        model=model,
                        metric=metric,
                        point=pt,
                        lo=lo,
                        hi=hi,
                        n=len(df),
                        unit=tag,
                    )
                )
    ci = pd.DataFrame(ci_rows)
    ci.to_csv(OUT_DIR / f"a1_metric_cis_{tag}.csv", index=False)

    # ---- Table 2: paired comparisons ----
    pair_rows = []
    for dataset in C.DATASETS:
        for name, ma, mb, metric, score_col in PAIRED:
            da, db = get_df(dataset, ma), get_df(dataset, mb)
            if da is None or db is None:
                continue
            delta, lo, hi, p, n = delta_fn(
                da, db, _stat_fn(metric, score_col), n_boot=n_boot, seed=seed
            )
            pair_rows.append(
                dict(
                    dataset=dataset,
                    dataset_label=C.DATASET_LABEL[dataset],
                    comparison=name,
                    model_a=ma,
                    model_b=mb,
                    metric=metric,
                    delta=delta,
                    lo=lo,
                    hi=hi,
                    p=p,
                    n=n,
                    unit=tag,
                    significant=bool(lo > 0 or hi < 0),
                )
            )
    pairs = pd.DataFrame(pair_rows)
    pairs.to_csv(OUT_DIR / f"a1_paired_tests_{tag}.csv", index=False)

    _print_summary(ci, pairs)


def _print_summary(ci: pd.DataFrame, pairs: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("A1 -- BOOTSTRAP 95% CIs (pooled episodes; percentile bootstrap)")
    print("=" * 78)
    for dataset in C.DATASETS:
        sub = ci[ci.dataset == dataset]
        if sub.empty:
            continue
        print(f"\n### {C.DATASET_LABEL[dataset]}  (n={int(sub['n'].iloc[0])})")
        for model in sub["model"].drop_duplicates():
            cells = []
            for metric in ["rmse", "wql", "auroc", "auprc"]:
                r = sub[(sub.model == model) & (sub.metric == metric)]
                if r.empty:
                    continue
                r = r.iloc[0]
                cells.append(f"{metric.upper()} {r.point:.3f} [{r.lo:.3f},{r.hi:.3f}]")
            print(f"  {model:10s} " + " | ".join(cells))

    print("\n" + "=" * 78)
    print("A1 -- PAIRED BOOTSTRAP (shared episodes; delta = A - B)")
    print("=" * 78)
    for _, r in pairs.iterrows():
        star = "SIGNIFICANT" if r.significant else "n.s."
        print(
            f"  [{C.DATASET_LABEL[r.dataset]:11s}] {r.comparison:26s} "
            f"delta={r.delta:+.4f} [{r.lo:+.4f},{r.hi:+.4f}] p={r.p:.4f}  {star}"
        )
    print(f"\nSaved: {OUT_DIR/'a1_metric_cis.csv'} , {OUT_DIR/'a1_paired_tests.csv'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="A1 bootstrap CIs + paired significance")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quick", action="store_true", help="n_boot=500 sanity run")
    ap.add_argument(
        "--bootstrap-unit",
        choices=["patient", "episode"],
        default="patient",
        help="resample patients (clustering-honest, default) or episodes",
    )
    args = ap.parse_args()
    n_boot = 500 if args.quick else args.n_boot
    run(args.models, n_boot=n_boot, seed=args.seed, unit=args.bootstrap_unit)


if __name__ == "__main__":
    main()
