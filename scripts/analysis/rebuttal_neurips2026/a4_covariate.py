# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)
"""A4 -- BG-only controlled leaderboard + covariate contribution (R3 W3/Q3, AC Q1/Q3).

Two questions the reviewers raise about the covariate ablations:
  (1) Fair comparison: when EVERY model sees the same univariate BG-only input,
      how do architectures rank? (removes the "some models got covariates"
      confound the AC/R3 flagged).
  (2) How much of a model's gain comes from covariates vs architecture?
      delta = RMSE(bg_only) - RMSE(best covariate condition), paired-bootstrapped
      on shared episodes, compared against the architecture spread on bg_only.

Uses the fine-tuned BG-only runs (reruns fill chronos2/moirai/ttm/toto; DL models
are natively bg_only). tft's bg_only per-episode arrays are unavailable on this
branch for 3 datasets -> its published Table-3 G values are shown without CI and
flagged. Point-only models (ttm/moment) have no WQL.

Usage:
    python -m scripts.analysis.rebuttal_neurips2026.a4_covariate
    python -m scripts.analysis.rebuttal_neurips2026.a4_covariate --n-boot 10000

Outputs (scripts/analysis/rebuttal_neurips2026/outputs/):
    a4_bgonly_leaderboard.csv   per (dataset, model): bg-only metrics + 95% CI
    a4_covariate_delta.csv      per (dataset, model): RMSE(bg_only)-RMSE(cov) + CI
    printed markdown-ready summary.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis.rebuttal_neurips2026 import common as C
from scripts.analysis.rebuttal_neurips2026.build_rerun_manifest import PAPER_RMSE

OUT_DIR = Path(__file__).resolve().parent / "outputs"

MODELS = [
    "chronos2",
    "moirai",
    "toto",
    "timesfm",
    "patchtst",
    "tft",
    "ttm",
    "moment",
    "tide",
    "deepar",
]


def _bgonly_run(model: str, dataset: str) -> Path | None:
    """Fine-tuned bg-only run (accept mode finetuned or the DL 'unknown' naming;
    never zero-shot)."""
    return C.run_path_for_condition(
        model, dataset, "bg_only", mode="finetuned"
    ) or C.run_path_for_condition(model, dataset, "bg_only", mode="unknown")


def _covariate_run(model: str, dataset: str):
    """Best covariate-condition run (iob/iob_cob) if the model has one, min RMSE."""
    df = C.scan_runs()
    sub = df[
        (df.model == model)
        & (df.dataset == dataset)
        & df.condition.isin(["iob", "iob_cob"])
        & df.overall_rmse.notna()
    ]
    if sub.empty:
        return None, None
    sub = sub.sort_values("overall_rmse")
    return Path(sub.iloc[0]["run_path"]), sub.iloc[0]["condition"]


def run(models: list[str], n_boot: int, seed: int) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    C.save_run_index(OUT_DIR / "run_index.csv")

    lead, delta = [], []
    for dataset in C.DATASETS:
        for model in models:
            rp = _bgonly_run(model, dataset)
            if rp is None:
                # tft on 3 datasets: published point estimate, no CI
                exp = PAPER_RMSE.get((model, dataset, "bg_only"))
                lead.append(
                    dict(
                        dataset=dataset,
                        dataset_label=C.DATASET_LABEL[dataset],
                        model=model,
                        rmse=exp,
                        rmse_lo=np.nan,
                        rmse_hi=np.nan,
                        wql=np.nan,
                        auroc=np.nan,
                        auprc=np.nan,
                        n=np.nan,
                        source="published(no CI)",
                    )
                )
                continue
            df = C.load_run(rp, dataset=dataset, model=model).episodes
            rmse, rlo, rhi = C.bootstrap_ci(df, C.pooled_rmse, n_boot=n_boot, seed=seed)
            lead.append(
                dict(
                    dataset=dataset,
                    dataset_label=C.DATASET_LABEL[dataset],
                    model=model,
                    rmse=rmse,
                    rmse_lo=rlo,
                    rmse_hi=rhi,
                    wql=C.mean_metric(df, "wql"),
                    auroc=C.auroc(df, "score_point"),
                    auprc=C.auprc(df, "score_point"),
                    n=len(df),
                    source="rerun/survivor",
                )
            )
            # covariate contribution delta (bg_only - covariate); >0 => covariates help
            cov_rp, cov_cond = _covariate_run(model, dataset)
            if cov_rp is not None:
                cov_df = C.load_run(cov_rp, dataset=dataset, model=model).episodes
                d, lo, hi, p, n = C.paired_bootstrap_delta(
                    df, cov_df, C.pooled_rmse, n_boot=n_boot, seed=seed
                )
                delta.append(
                    dict(
                        dataset=dataset,
                        dataset_label=C.DATASET_LABEL[dataset],
                        model=model,
                        cov_condition=cov_cond,
                        rmse_bgonly=C.pooled_rmse(df),
                        rmse_cov=C.pooled_rmse(cov_df),
                        delta=d,
                        lo=lo,
                        hi=hi,
                        p=p,
                        n=n,
                        significant=bool(lo > 0 or hi < 0),
                    )
                )

    lead_df = pd.DataFrame(lead)
    delta_df = pd.DataFrame(delta)
    lead_df.to_csv(OUT_DIR / "a4_bgonly_leaderboard.csv", index=False)
    delta_df.to_csv(OUT_DIR / "a4_covariate_delta.csv", index=False)
    _print_summary(lead_df, delta_df)


def _print_summary(lead: pd.DataFrame, delta: pd.DataFrame) -> None:
    print("\n" + "=" * 88)
    print(
        "A4a -- BG-ONLY CONTROLLED LEADERBOARD (identical univariate input; RMSE 95% CI)"
    )
    print("=" * 88)
    for dataset in C.DATASETS:
        sub = lead[lead.dataset == dataset].sort_values("rmse")
        if sub.empty:
            continue
        print(f"\n### {C.DATASET_LABEL[dataset]}  (bg-only, ranked by RMSE)")
        for i, r in enumerate(sub.itertuples(), 1):
            ci = (
                f"[{r.rmse_lo:.3f},{r.rmse_hi:.3f}]"
                if pd.notna(r.rmse_lo)
                else "(published, no CI)"
            )
            au = f"AUROC {r.auroc:.3f}" if pd.notna(r.auroc) else "AUROC   n/a"
            print(f"  {i:2d}. {r.model:9s} RMSE {r.rmse:.3f} {ci:>17s}  {au}")

    print("\n" + "=" * 88)
    print(
        "A4b -- COVARIATE CONTRIBUTION: RMSE(bg_only) - RMSE(best covariate), paired bootstrap"
    )
    print(
        "    delta>0 => covariates improve RMSE; compare magnitude to architecture spread above"
    )
    print("=" * 88)
    for dataset in C.DATASETS:
        sub = delta[delta.dataset == dataset]
        if sub.empty:
            continue
        print(f"\n### {C.DATASET_LABEL[dataset]}")
        for r in sub.sort_values("delta", ascending=False).itertuples():
            star = "sig" if r.significant else "n.s."
            print(
                f"  {r.model:9s} ({r.cov_condition:7s}) bg={r.rmse_bgonly:.3f} cov={r.rmse_cov:.3f} "
                f"delta={r.delta:+.3f} [{r.lo:+.3f},{r.hi:+.3f}] {star}"
            )
        # architecture spread (best-worst bg-only RMSE) for context
        L = lead[(lead.dataset == dataset) & lead.rmse.notna()]
        spread = L.rmse.max() - L.rmse.min()
        md = sub.delta.median()
        print(
            f"  -> architecture spread on bg-only = {spread:.3f} RMSE; "
            f"median covariate gain = {md:+.3f} RMSE"
        )
    print(f"\nSaved: {OUT_DIR}/a4_bgonly_leaderboard.csv , a4_covariate_delta.csv")


def main() -> None:
    ap = argparse.ArgumentParser(description="A4 bg-only leaderboard + covariate delta")
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    run(args.models, n_boot=args.n_boot, seed=args.seed)


if __name__ == "__main__":
    main()
