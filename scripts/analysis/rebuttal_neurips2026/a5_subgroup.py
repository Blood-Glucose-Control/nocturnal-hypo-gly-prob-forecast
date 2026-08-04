# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)
"""A5 -- Subgroup / fairness analysis (R3 Q4, W5).

Stratifies episode-level performance by patient SEX and by AGE (median split on
the ACTUAL enrollment age -- not age-at-diagnosis, per reviewer note), for the
top models, and reports whether metrics and model RANKINGS are stable across
subgroups (the fairness question).

Metrics per subgroup: pooled RMSE, AUROC, AUPRC (episode any-step hypo,
detection score = P(hypo) = max_t P(BG_t < 3.9), matching the paper's Table 5a).
Age split uses the median enrollment age of the evaluated patients in each
dataset (cohorts differ: Lynch includes children).

Usage:
    python -m scripts.analysis.rebuttal_neurips2026.a5_subgroup

Outputs (scripts/analysis/rebuttal_neurips2026/outputs/):
    a5_subgroup.csv          one row per (dataset, model, axis, group)
    a5_demographics.csv      merged per-episode demographics coverage summary
    printed markdown-ready summary (gaps + ranking stability).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from scripts.analysis.rebuttal_neurips2026 import common as C
from scripts.analysis.rebuttal_neurips2026 import demographics as D

OUT_DIR = Path(__file__).resolve().parent / "outputs"

DEFAULT_MODELS = ["chronos2", "moirai", "toto", "patchtst", "tft"]


def _metrics(df: pd.DataFrame) -> dict:
    return dict(
        n=len(df),
        n_hypo=int(df["has_hypo"].sum()),
        base_rate=float(df["has_hypo"].mean()),
        rmse=C.pooled_rmse(df),
        auroc=C.auroc(df, "score_prob"),
        auprc=C.auprc(df, "score_prob"),
    )


def run(models: list[str], min_group: int = 100) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    C.save_run_index(OUT_DIR / "run_index.csv")

    rows = []
    cov_rows = []
    for dataset in C.DATASETS:
        demo = D.load_demographics(dataset)
        # median enrollment age over evaluated patients (any one model's episodes)
        ref_rp = next(
            (
                C.best_run_path(m, dataset)
                for m in models
                if C.best_run_path(m, dataset)
            ),
            None,
        )
        if ref_rp is None:
            continue
        ref = C.load_run(ref_rp, dataset=dataset).episodes
        ages = demo.set_index("pid")["age"]
        eval_ages = ages.reindex(ref["pid"].unique()).dropna()
        age_median = float(eval_ages.median())

        for model in models:
            rp = C.best_run_path(model, dataset)
            if rp is None or not rp.exists():
                continue
            df = C.load_run(rp, dataset=dataset, model=model).episodes
            merged = df.merge(demo[["pid", "sex", "age"]], on="pid", how="left")
            cov_rows.append(
                dict(
                    dataset=dataset,
                    model=model,
                    n=len(merged),
                    n_missing_sex=int(merged["sex"].isna().sum()),
                    n_missing_age=int(merged["age"].isna().sum()),
                )
            )
            # SEX axis
            for g, sub in merged.groupby("sex"):
                if len(sub) < min_group:
                    continue
                rows.append(
                    dict(
                        dataset=dataset,
                        dataset_label=C.DATASET_LABEL[dataset],
                        model=model,
                        axis="sex",
                        group=str(g),
                        **_metrics(sub),
                    )
                )
            # AGE axis (median split)
            merged["age_group"] = np.where(
                merged["age"] < age_median, "younger", "older"
            )
            merged.loc[merged["age"].isna(), "age_group"] = np.nan
            for g, sub in merged.dropna(subset=["age"]).groupby("age_group"):
                if len(sub) < min_group:
                    continue
                rows.append(
                    dict(
                        dataset=dataset,
                        dataset_label=C.DATASET_LABEL[dataset],
                        model=model,
                        axis="age",
                        group=f"{g}(<{age_median:.0f})"
                        if g == "younger"
                        else f"{g}(>={age_median:.0f})",
                        **_metrics(sub),
                    )
                )

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "a5_subgroup.csv", index=False)
    pd.DataFrame(cov_rows).to_csv(OUT_DIR / "a5_demographics.csv", index=False)
    _print_summary(out)


def _print_summary(out: pd.DataFrame) -> None:
    print("\n" + "=" * 90)
    print(
        "A5 -- SUBGROUP ANALYSIS (true enrollment age + sex; detection score = P(hypo))"
    )
    print("=" * 90)
    for dataset in C.DATASETS:
        sub = out[out.dataset == dataset]
        if sub.empty:
            continue
        print(f"\n### {C.DATASET_LABEL[dataset]}")
        for axis in ["sex", "age"]:
            a = sub[sub.axis == axis]
            if a.empty:
                continue
            groups = list(a["group"].drop_duplicates())
            print(f"  [{axis}] groups: {groups}")
            for model in a["model"].drop_duplicates():
                cells = []
                for g in groups:
                    r = a[(a.model == model) & (a.group == g)]
                    if r.empty:
                        continue
                    r = r.iloc[0]
                    cells.append(
                        f"{g}: AUROC {r.auroc:.3f} AUPRC {r.auprc:.3f} RMSE {r.rmse:.3f} (n={r.n})"
                    )
                print(f"    {model:9s} " + " | ".join(cells))
            # ranking stability across the two groups (by AUROC)
            if len(groups) == 2:
                piv = a.pivot_table(index="model", columns="group", values="auroc")
                if piv.shape[1] == 2 and piv.dropna().shape[0] >= 3:
                    g1, g2 = piv.columns[:2]
                    tau, _ = kendalltau(piv[g1].rank(), piv[g2].rank())
                    print(
                        f"    -> rank stability (Kendall tau, AUROC {g1} vs {g2}): {tau:+.3f}"
                    )
    print(f"\nSaved: {OUT_DIR}/a5_subgroup.csv , {OUT_DIR}/a5_demographics.csv")


def main() -> None:
    ap = argparse.ArgumentParser(description="A5 subgroup / fairness analysis")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--min-group", type=int, default=100)
    args = ap.parse_args()
    run(args.models, min_group=args.min_group)


if __name__ == "__main__":
    main()
