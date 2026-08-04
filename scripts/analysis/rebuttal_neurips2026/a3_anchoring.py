# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)
"""A3 / A3b -- Midnight-anchoring sensitivity + overnight intervention timing.

R3's #1 concern (W1/Q1): midnight-anchored nights may include periods where the
patient is awake, eating, or dosing insulin, injecting label noise into the
"nocturnal" framing. We address it two ways, using the processed per-patient
time series (insulin bolus = dose_units/bolus; carbs = food_g where available):

A3b -- WHEN do overnight interventions occur?
    Hour-of-day histogram (0-23) of bolus insulin and carb events across all
    evaluated patients, plus stats on the 8 h forecast window (00:00-08:00):
    fraction of episode-nights with any bolus / any carb in-window, and the
    distribution of in-window bolus hours. Establishes how "quiescent" the
    midnight window actually is.

A3 -- Do conclusions hold on QUIESCENT nights?
    Label each midnight episode "quiescent" (no bolus AND no carb in the 8 h
    window) vs "active"; recompute episode-level metrics per model on the
    quiescent subset and check whether the model RANKING (AUROC) is preserved
    vs the full set (Kendall tau). If rankings hold, the benchmark's conclusions
    are robust to sleep-label noise.

Insulin bolus is available for Aleppo/DCLP3/IOBP2; carbs (food_g) only for
Aleppo. Tamborlane (2008 CGM-only) has no intervention channel -> reported as
coverage-limited.

Usage:
    python -m scripts.analysis.rebuttal_neurips2026.a3_anchoring

Outputs (scripts/analysis/rebuttal_neurips2026/outputs/):
    a3_hour_histogram.csv      hour(0-23) x dataset: bolus_units, n_bolus, n_carb
    a3_episode_activity.csv     per-episode in-window bolus/carb activity flags
    a3_quiescent_metrics.csv    per (dataset, model, subset) metrics + rank tau
    printed markdown-ready summary.
"""

from __future__ import annotations

import argparse
import functools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from scripts.analysis.rebuttal_neurips2026 import common as C

OUT_DIR = Path(__file__).resolve().parent / "outputs"
PROC = Path("/data/shared/cache/data")
WINDOW_H = 8  # forecast horizon hours (96 steps x 5 min)

DEFAULT_MODELS = ["chronos2", "moirai", "toto", "patchtst", "tft"]


@functools.lru_cache(maxsize=2048)
def _patient_frame(dataset: str, pid: str) -> pd.DataFrame | None:
    """Load a processed patient CSV with datetime + insulin/carb signals."""
    f = PROC / dataset / "processed" / f"{pid}_full.csv"
    if not f.exists():
        return None
    df = pd.read_csv(
        f, usecols=lambda c: c in ("datetime", "dose_units", "bolus", "food_g", "cob")
    )
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).set_index("datetime").sort_index()
    # robust intervention signals
    bolus = df["bolus"] if "bolus" in df.columns else df.get("dose_units")
    df["_bolus"] = (
        pd.to_numeric(bolus, errors="coerce").fillna(0.0) if bolus is not None else 0.0
    )
    df["_carb"] = (
        pd.to_numeric(df.get("food_g"), errors="coerce").fillna(0.0)
        if "food_g" in df.columns
        else 0.0
    )
    return df


def _episode_activity(dataset: str, patient_id: str, anchor: pd.Timestamp) -> dict:
    """In-window bolus/carb activity for one episode.

    Wakefulness proxy uses interventions STRICTLY AFTER the anchor (index >
    anchor): a bolus logged exactly at 00:00 is a bedtime dose, not overnight
    wakefulness. Carbs at any point count (announced meals imply the patient is
    awake). Note: closed-loop cohorts (DCLP3/IOBP2) deliver micro-boluses
    automatically, so bolus activity there is only a partial wakefulness signal.
    """
    pf = _patient_frame(dataset, patient_id)
    if pf is None:
        return dict(
            has_bolus=np.nan,
            has_carb=np.nan,
            n_bolus=np.nan,
            n_carb=np.nan,
            first_bolus_hr=np.nan,
            covered=False,
        )
    end = anchor + pd.Timedelta(hours=WINDOW_H)
    w = pf.loc[(pf.index > anchor) & (pf.index < end)]  # strictly after midnight
    b = w[w["_bolus"] > 0]
    c = w[w["_carb"] > 0]
    first_hr = ((b.index[0] - anchor).total_seconds() / 3600.0) if len(b) else np.nan
    return dict(
        has_bolus=bool(len(b) > 0),
        has_carb=bool(len(c) > 0),
        n_bolus=int(len(b)),
        n_carb=int(len(c)),
        first_bolus_hr=first_hr,
        covered=True,
    )


def _hour_histogram(dataset: str, pids: list[str]) -> pd.DataFrame:
    """Aggregate bolus units / bolus count / carb count by clock hour (0-23)."""
    agg = np.zeros((24, 3))  # bolus_units, n_bolus, n_carb
    for pid in pids:
        pf = _patient_frame(dataset, pid)
        if pf is None:
            continue
        hr = pf.index.hour
        b = pf["_bolus"].to_numpy()
        c = pf["_carb"].to_numpy()
        for h in range(24):
            m = hr == h
            agg[h, 0] += float(b[m].sum())
            agg[h, 1] += int((b[m] > 0).sum())
            agg[h, 2] += int((c[m] > 0).sum())
    out = pd.DataFrame(agg, columns=["bolus_units", "n_bolus", "n_carb"])
    out.insert(0, "hour", range(24))
    out.insert(0, "dataset", dataset)
    return out


def run(models: list[str], min_group: int = 100) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    C.save_run_index(OUT_DIR / "run_index.csv")

    hist_all, act_all, metric_rows = [], [], []
    for dataset in C.DATASETS:
        # reference episode set (anchors/patients) from any available model
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
        ref = ref.copy()
        ref["anchor_ts"] = pd.to_datetime(ref["anchor"])
        pids = sorted(ref["pid"].unique())

        # ---- A3b: hour-of-day histogram ----
        hist_all.append(_hour_histogram(dataset, pids))

        # ---- per-episode activity (unique patient+anchor) ----
        uniq = ref[["pid", "anchor_ts"]].drop_duplicates()
        act = {
            (r.pid, r.anchor_ts): _episode_activity(dataset, r.pid, r.anchor_ts)
            for r in uniq.itertuples()
        }
        act_df = pd.DataFrame(
            [
                {"dataset": dataset, "pid": k[0], "anchor": k[1], **v}
                for k, v in act.items()
            ]
        )
        act_all.append(act_df)

        covered = act_df["covered"].mean() if len(act_df) else 0.0
        if covered < 0.5:
            print(
                f"[{C.DATASET_LABEL[dataset]}] intervention data unavailable "
                f"(covered={covered:.0%}) -- skipping quiescent split."
            )
            continue

        # activity flag per (pid, anchor)
        flag = {
            (r.pid, r.anchor): (bool(r.has_bolus) or bool(r.has_carb))
            for r in act_df.itertuples()
            if r.covered
        }

        # ---- A3: quiescent vs active metrics per model ----
        for model in models:
            rp = C.best_run_path(model, dataset)
            if rp is None or not rp.exists():
                continue
            df = C.load_run(rp, dataset=dataset, model=model).episodes.copy()
            df["anchor_ts"] = pd.to_datetime(df["anchor"])
            df["active"] = [
                flag.get((p, a), np.nan) for p, a in zip(df["pid"], df["anchor_ts"])
            ]
            df = df.dropna(subset=["active"])
            for subset, sel in [
                ("full", df),
                ("quiescent", df[~df["active"].astype(bool)]),
                ("active", df[df["active"].astype(bool)]),
            ]:
                if len(sel) < min_group:
                    continue
                metric_rows.append(
                    dict(
                        dataset=dataset,
                        dataset_label=C.DATASET_LABEL[dataset],
                        model=model,
                        subset=subset,
                        n=len(sel),
                        n_hypo=int(sel["has_hypo"].sum()),
                        base_rate=float(sel["has_hypo"].mean()),
                        rmse=C.pooled_rmse(sel),
                        auroc=C.auroc(sel, "score_point"),
                        auprc=C.auprc(sel, "score_point"),
                    )
                )

    hist = pd.concat(hist_all, ignore_index=True) if hist_all else pd.DataFrame()
    acts = pd.concat(act_all, ignore_index=True) if act_all else pd.DataFrame()
    mets = pd.DataFrame(metric_rows)
    hist.to_csv(OUT_DIR / "a3_hour_histogram.csv", index=False)
    acts.to_csv(OUT_DIR / "a3_episode_activity.csv", index=False)
    mets.to_csv(OUT_DIR / "a3_quiescent_metrics.csv", index=False)
    _print_summary(hist, acts, mets)


def _print_summary(hist: pd.DataFrame, acts: pd.DataFrame, mets: pd.DataFrame) -> None:
    print("\n" + "=" * 90)
    print("A3b -- OVERNIGHT INTERVENTION TIMING (in 8 h forecast window 00:00-08:00)")
    print("=" * 90)
    for dataset in C.DATASETS:
        a = acts[acts.dataset == dataset]
        a = a[a["covered"]]
        if a.empty:
            continue
        n = len(a)
        pb = 100.0 * a["has_bolus"].mean()
        pc = 100.0 * a["has_carb"].mean()
        fh = a.loc[a["has_bolus"].astype(bool), "first_bolus_hr"].dropna()
        med = f"{fh.median():.2f}h" if len(fh) else "n/a"
        # share of in-window boluses in first 2h vs last 6h
        hh = hist[hist.dataset == dataset]
        night = hh[hh.hour < WINDOW_H]["n_bolus"].sum()
        day = hh["n_bolus"].sum()
        night_share = 100.0 * night / day if day else float("nan")
        print(
            f"  {C.DATASET_LABEL[dataset]:11s} n={n:5d}  nights w/ bolus in window={pb:4.1f}%  "
            f"carb={pc:4.1f}%  median 1st-bolus={med}  "
            f"| boluses in 00-08h = {night_share:.1f}% of daily boluses"
        )

    print("\n" + "=" * 90)
    print("A3 -- QUIESCENT vs ACTIVE vs FULL (does ranking hold on quiescent nights?)")
    print("=" * 90)
    for dataset in C.DATASETS:
        m = mets[mets.dataset == dataset]
        if m.empty:
            continue
        print(f"\n### {C.DATASET_LABEL[dataset]}")
        for subset in ["full", "quiescent", "active"]:
            s = m[m.subset == subset]
            if s.empty:
                continue
            br = s["base_rate"].iloc[0]
            n = int(s["n"].iloc[0])
            cells = " | ".join(f"{r.model}:{r.auroc:.3f}" for r in s.itertuples())
            print(f"  {subset:9s} (n={n:5d}, hypo={br:.3f})  AUROC  {cells}")
        # rank stability full vs quiescent
        piv = m[m.subset.isin(["full", "quiescent"])].pivot_table(
            index="model", columns="subset", values="auroc"
        )
        if {"full", "quiescent"}.issubset(piv.columns) and piv.dropna().shape[0] >= 3:
            tau, _ = kendalltau(piv["full"].rank(), piv["quiescent"].rank())
            print(
                f"  -> AUROC rank stability full vs quiescent: Kendall tau = {tau:+.3f}"
            )
    print(
        f"\nSaved: {OUT_DIR}/a3_hour_histogram.csv, a3_episode_activity.csv, a3_quiescent_metrics.csv"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="A3/A3b midnight-anchoring sensitivity")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--min-group", type=int, default=100)
    args = ap.parse_args()
    run(args.models, min_group=args.min_group)


if __name__ == "__main__":
    main()
