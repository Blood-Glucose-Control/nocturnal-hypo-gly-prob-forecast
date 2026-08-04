# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)
"""A2 -- Alarm operating-point analysis (clinical utility of the detector).

Answers:
  - R1  : "what mistakes are the models making? precision / false alarms?"
  - R2 Q3: "behaviour if classified via deterministic point forecasts" -- this IS
           the Table 5a score s = -min_t mu_t; we make its alarm behaviour explicit.
  - R3 Q5: "do AUROC/AUPRC gains translate into fewer missed events?"
  - AC Q2: false-positive rates for a real nocturnal alarm system.

Setup (identical to the paper's episode-level detection, Table 5a):
  label  y_i = 1 if any step of the 8 h night has BG < 3.9 mmol/L
  score  s_i = -min_t mu_hat_t   (deterministic point-forecast risk; comparable
                                  across ALL models incl. point-only like TTM)
For each dataset x model we report three operating points:
  - sens>=0.90   high-recall clinical setpoint (miss <=10% of hypo nights)
  - sens>=0.80
  - Youden-J      max (sensitivity + specificity - 1)
reporting sensitivity, specificity, PPV (1 - false-alarm fraction among alarms),
NPV, and false-alarms-per-100-nights (a directly interpretable alarm burden).

A secondary pass uses the probabilistic score s = max_t P(BG_t < 3.9) for models
that emit quantiles.

Usage:
    python -m scripts.analysis.rebuttal_neurips2026.a2_alarm
    python -m scripts.analysis.rebuttal_neurips2026.a2_alarm --bootstrap --n-boot 2000

Outputs (scripts/analysis/rebuttal_neurips2026/outputs/):
    a2_operating_points.csv   one row per (dataset, model, score, operating_point)
    printed markdown-ready summary.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis.rebuttal_neurips2026 import common as C

OUT_DIR = Path(__file__).resolve().parent / "outputs"

# Detection-relevant leaders (RMSE/WQL leaders + shape/detection leaders + a DL ref).
DEFAULT_MODELS = ["chronos2", "moirai", "toto", "patchtst", "tft", "ttm", "deepar"]

# Operating points to report.
SENS_TARGETS = [0.90, 0.80]


def _op_at_threshold(y: np.ndarray, s: np.ndarray, thr: float) -> dict:
    """Confusion-matrix-derived operating metrics at a fixed score threshold."""
    pred = s >= thr
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))
    n = len(y)
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    npv = tn / (tn + fn) if (tn + fn) else float("nan")
    return dict(
        thr=float(thr),
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        sens=sens,
        spec=spec,
        ppv=ppv,
        npv=npv,
        fp_per100=100.0 * fp / n,
        alarms_per100=100.0 * (tp + fp) / n,
    )


def _threshold_for_sensitivity(y: np.ndarray, s: np.ndarray, target: float) -> float:
    """Highest threshold whose sensitivity is >= target (max specificity setpoint)."""
    for t in np.unique(s)[::-1]:  # descending: alarms increase, sensitivity rises
        if _op_at_threshold(y, s, t)["sens"] >= target:
            return float(t)
    return float(s.min())


def _youden_threshold(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import roc_curve

    fpr, tpr, thr = roc_curve(y, s)
    j = tpr - fpr
    return float(thr[int(np.argmax(j))])


def _operating_points(y: np.ndarray, s: np.ndarray) -> list[tuple[str, float]]:
    ops = [
        (f"sens{int(t*100)}", _threshold_for_sensitivity(y, s, t)) for t in SENS_TARGETS
    ]
    ops.append(("youden", _youden_threshold(y, s)))
    return ops


def _bootstrap_op(
    y: np.ndarray,
    s: np.ndarray,
    target: float,
    keys: tuple[str, ...],
    n_boot: int,
    seed: int,
) -> dict:
    """Percentile-bootstrap CIs for chosen metrics at a sens-target operating point.

    Threshold is re-selected inside each resample so the CI reflects setpoint
    uncertainty, not a fixed cut.
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    acc = {k: [] for k in keys}
    for _ in range(n_boot):
        take = rng.choice(idx, size=len(idx), replace=True)
        yb, sb = y[take], s[take]
        if yb.min() == yb.max():
            continue
        thr = _threshold_for_sensitivity(yb, sb, target)
        op = _op_at_threshold(yb, sb, thr)
        for k in keys:
            acc[k].append(op[k])
    out = {}
    for k in keys:
        arr = np.asarray(acc[k], dtype=float)
        out[f"{k}_lo"] = float(np.nanpercentile(arr, 2.5)) if arr.size else float("nan")
        out[f"{k}_hi"] = (
            float(np.nanpercentile(arr, 97.5)) if arr.size else float("nan")
        )
    return out


def run(models: list[str], score: str, bootstrap: bool, n_boot: int, seed: int) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    C.save_run_index(OUT_DIR / "run_index.csv")

    score_col = "score_point" if score == "point" else "score_prob"
    rows = []
    for dataset in C.DATASETS:
        for model in models:
            rp = C.best_run_path(model, dataset)
            if rp is None or not rp.exists():
                continue
            df = C.load_run(rp, dataset=dataset, model=model).episodes
            y = df["has_hypo"].to_numpy().astype(int)
            s = df[score_col].to_numpy()
            if np.all(np.isnan(s)) or y.min() == y.max():
                continue  # point-only under prob score, or degenerate label
            base = dict(
                dataset=dataset,
                dataset_label=C.DATASET_LABEL[dataset],
                model=model,
                score=score,
                n=len(y),
                n_hypo=int(y.sum()),
                base_rate=float(y.mean()),
            )
            for op_name, thr in _operating_points(y, s):
                op = _op_at_threshold(y, s, thr)
                row = {**base, "operating_point": op_name, **op}
                if bootstrap and op_name.startswith("sens"):
                    target = int(op_name[4:]) / 100.0
                    row.update(
                        _bootstrap_op(
                            y,
                            s,
                            target,
                            ("ppv", "fp_per100", "spec"),
                            n_boot=n_boot,
                            seed=seed,
                        )
                    )
                rows.append(row)

    out = pd.DataFrame(rows)
    tag = "point" if score == "point" else "prob"
    out.to_csv(OUT_DIR / f"a2_operating_points_{tag}.csv", index=False)
    _print_summary(out, score)


def _print_summary(out: pd.DataFrame, score: str) -> None:
    print("\n" + "=" * 92)
    print(
        f"A2 -- ALARM OPERATING POINTS  (score = {'-min_t mu (point)' if score=='point' else 'max_t P(BG<3.9) (prob)'})"
    )
    print(
        "    sens=recall, spec=specificity, PPV=1-false-alarm-frac, FP/100=false alarms per 100 nights"
    )
    print("=" * 92)
    for dataset in C.DATASETS:
        sub = out[out.dataset == dataset]
        if sub.empty:
            continue
        br = sub["base_rate"].iloc[0]
        print(
            f"\n### {C.DATASET_LABEL[dataset]}  (n={int(sub['n'].iloc[0])}, hypo base rate={br:.3f})"
        )
        for model in sub["model"].drop_duplicates():
            for _, r in sub[sub.model == model].iterrows():
                extra = ""
                if "ppv_lo" in r and not pd.isna(r.get("ppv_lo", np.nan)):
                    extra = f"  PPV95%[{r['ppv_lo']:.3f},{r['ppv_hi']:.3f}]"
                print(
                    f"  {model:9s} {r['operating_point']:7s} "
                    f"sens={r['sens']:.3f} spec={r['spec']:.3f} PPV={r['ppv']:.3f} "
                    f"NPV={r['npv']:.3f} FP/100={r['fp_per100']:.1f}{extra}"
                )
    print(f"\nSaved: {OUT_DIR}/a2_operating_points_*.csv")


def main() -> None:
    ap = argparse.ArgumentParser(description="A2 alarm operating points")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument(
        "--score",
        choices=["point", "prob"],
        default="point",
        help="risk score: point=-min mu (Table 5a, all models); prob=max P(hypo)",
    )
    ap.add_argument(
        "--bootstrap", action="store_true", help="add 95% CIs at sens setpoints"
    )
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    run(
        args.models,
        score=args.score,
        bootstrap=args.bootstrap,
        n_boot=args.n_boot,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
