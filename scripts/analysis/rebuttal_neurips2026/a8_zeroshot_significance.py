# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)
"""A8 -- Zero-shot vs fine-tuned significance (uses the rerun ZS runs).

For each foundation model that has both a zero-shot run and a fine-tuned run, we
paired-cluster-bootstrap the RMSE (and WQL) difference on the shared midnight
episodes:  delta = RMSE(zero-shot) - RMSE(fine-tuned best).  delta > 0 and CI
excluding 0 => fine-tuning significantly improves that model. Resampling is by
PATIENT (episodes clustered within patients).

Usage:
    python -m scripts.analysis.rebuttal_neurips2026.a8_zeroshot_significance
    python -m scripts.analysis.rebuttal_neurips2026.a8_zeroshot_significance --n-boot 10000

Outputs (scripts/analysis/rebuttal_neurips2026/outputs/):
    a8_zeroshot_vs_finetuned.csv   per (dataset, model): delta RMSE/WQL + CI + p
    printed markdown-ready summary.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts.analysis.rebuttal_neurips2026 import common as C

OUT_DIR = Path(__file__).resolve().parent / "outputs"

ZS_MODELS = ["chronos2", "moirai", "moment", "timesfm", "ttm", "toto", "sundial"]


def run(models: list[str], n_boot: int, seed: int) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    C.save_run_index(OUT_DIR / "run_index.csv")

    rows = []
    for dataset in C.DATASETS:
        for model in models:
            zs = C.run_path_for_condition(model, dataset, "bg_only", mode="zeroshot")
            ft = C.best_run_path(model, dataset, mode="finetuned")
            if zs is None or ft is None:
                continue
            zs_df = C.load_run(
                zs, dataset=dataset, model=model, compute_phypo=False
            ).episodes
            ft_df = C.load_run(
                ft, dataset=dataset, model=model, compute_phypo=False
            ).episodes
            row = dict(
                dataset=dataset, dataset_label=C.DATASET_LABEL[dataset], model=model
            )
            for metric, col in [("rmse", None), ("wql", "wql")]:
                stat = (
                    C.pooled_rmse
                    if metric == "rmse"
                    else (lambda d: C.mean_metric(d, "wql"))
                )
                d, lo, hi, p, n = C.paired_cluster_bootstrap_delta(
                    zs_df, ft_df, stat, n_boot=n_boot, seed=seed
                )
                row[f"{metric}_zs"] = stat(zs_df)
                row[f"{metric}_ft"] = stat(ft_df)
                row[f"{metric}_delta"] = d
                row[f"{metric}_lo"] = lo
                row[f"{metric}_hi"] = hi
                row[f"{metric}_p"] = p
                row[f"{metric}_sig"] = bool(lo > 0 or hi < 0)
                row["n_patients"] = n
            rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "a8_zeroshot_vs_finetuned.csv", index=False)
    _print_summary(out)


def _print_summary(out: pd.DataFrame) -> None:
    print("\n" + "=" * 90)
    print(
        "A8 -- ZERO-SHOT vs FINE-TUNED (patient-cluster paired bootstrap; delta=ZS-FT)"
    )
    print("    delta>0 & sig => fine-tuning significantly improves the model")
    print("=" * 90)
    for dataset in C.DATASETS:
        sub = out[out.dataset == dataset]
        if sub.empty:
            continue
        print(f"\n### {C.DATASET_LABEL[dataset]}")
        for r in sub.itertuples():
            star = "sig" if r.rmse_sig else "n.s."
            print(
                f"  {r.model:9s} RMSE zs={r.rmse_zs:.3f} ft={r.rmse_ft:.3f} "
                f"delta={r.rmse_delta:+.3f} [{r.rmse_lo:+.3f},{r.rmse_hi:+.3f}] {star}  "
                f"(n={int(r.n_patients)} pts)"
            )
    print(f"\nSaved: {OUT_DIR}/a8_zeroshot_vs_finetuned.csv")


def main() -> None:
    ap = argparse.ArgumentParser(description="A8 zero-shot vs fine-tuned significance")
    ap.add_argument("--models", nargs="+", default=ZS_MODELS)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    run(args.models, n_boot=args.n_boot, seed=args.seed)


if __name__ == "__main__":
    main()
