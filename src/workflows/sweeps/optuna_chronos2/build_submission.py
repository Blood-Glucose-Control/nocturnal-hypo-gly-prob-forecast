# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)

"""Generate a MetaboNet challenge submission from a trained Chronos-2 model.

Takes a kept Optuna trial's predictor (with its covariate selection) and the
challenge ``template.parquet`` (anchors: ``id``, ``source_file``, ``date``),
builds the input context for every anchor from the MetaboNet ``test.parquet``,
forecasts 24 steps, and writes ``pred_30/60/90/120`` in template row order.

With ``--score`` (live leaderboard), it self-scores the populated rows against
``targets.parquet`` using RMSE/MARD/DTS so you get a local leaderboard estimate
without the full toolkit run. For an official submission, generate the full file
(no ``--limit-groups``) and validate with the toolkit's ``run.py``.

Example (quick subset estimate)::

    python -m src.workflows.sweeps.optuna_chronos2.build_submission \\
        --storage sqlite:///results/optuna/chronos2_metabonet_18h_v3.db \\
        --study-name chronos2_metabonet_18h_v3 \\
        --models-root results/optuna/chronos2_metabonet_18h_v3_models \\
        --template /tmp/gpst/data/live_leaderboard/template.parquet \\
        --targets /tmp/gpst/data/live_leaderboard/targets.parquet \\
        --test-parquet cache/data/metabonet/raw/test.parquet \\
        --out results/optuna/submission_best.parquet --limit-groups 5 --score
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import optuna
import pandas as pd

from .features import (
    DEMOGRAPHIC_KNOWN_COLS,
    SEX_KNOWN_COL,
    TARGET_COL,
    encode_demographics,
    encode_sex,
    engineer_patient_frame,
)
from .metrics import HORIZON_MINUTES, HORIZON_STEPS, dts_error_grid, mard, rmse

logger = logging.getLogger(__name__)

# Re-identified holdout demographics (ethnicity + age_of_diagnosis recovered for
# the CERTAIN annual->IOBP2 matches; these two columns are empty in the holdout
# public parquet). Used only to fill the demographic known covariates for the
# matched subset; unmatched patients fall back to the neutral cohort center.
_REID_DEMOGRAPHICS_PATH = (
    "cache/data/metabonet/raw/annual/outputs/annual_holdout_reid_demographics.csv"
)
_reid_demographics_cache: Optional[Dict[str, Dict[str, object]]] = None


def _first_value(raw: pd.DataFrame, col: str) -> object:
    """First non-null value of a column in a patient frame, else NaN."""
    if col in raw.columns and raw[col].notna().any():
        return raw[col].dropna().iloc[0]
    return np.nan


def _load_reid_demographics() -> Dict[str, Dict[str, object]]:
    """Lazy-load the annual_id -> {ethnicity, age_of_diagnosis} re-id lookup.

    Returns an empty mapping if the artifact is absent (demographics then degrade
    to the neutral cohort center for all holdout patients).
    """
    global _reid_demographics_cache
    if _reid_demographics_cache is None:
        path = Path(_REID_DEMOGRAPHICS_PATH)
        if path.exists():
            df = pd.read_csv(path)
            _reid_demographics_cache = {
                str(r["annual_id"]): {
                    "ethnicity": r.get("ethnicity"),
                    "age_of_diagnosis": r.get("age_of_diagnosis"),
                }
                for _, r in df.iterrows()
            }
            logger.info(
                "Loaded %d re-identified holdout demographics",
                len(_reid_demographics_cache),
            )
        else:
            logger.warning("No re-id demographics at %s; using neutral", path)
            _reid_demographics_cache = {}
    return _reid_demographics_cache


# Columns pulled from test.parquet (target + covariate sources + static).
_TEST_COLUMNS = [
    "id",
    "source_file",
    "date",
    "CGM",
    "basal",
    "bolus",
    "insulin",
    "carbs",
    "heartrate",
    "steps",
    "calories_burned",
    "age",
    "gender",
    "treatment_group",
    "age_of_diagnosis",
    "ethnicity",
]


def _best_trial(study: optuna.Study, trial_number: Optional[int]):
    if trial_number is not None:
        return next(t for t in study.trials if t.number == trial_number)
    complete = [
        t
        for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
    ]
    if not complete:
        raise ValueError("No completed trials to build a submission from.")
    complete.sort(key=lambda t: t.value)
    return complete[0]


def _forecast_group(
    predictor,
    frame: pd.DataFrame,
    anchors: pd.DataFrame,
    *,
    group_key: str,
    context_length: int,
    forecast_length: int,
    interval_mins: int,
    past_cols: Sequence[str],
    known_cols: Sequence[str],
    static_cols: Sequence[str],
    static_row: Dict[str, object],
) -> pd.DataFrame:
    """Forecast every anchor in one (id, source_file) group; return preds."""
    from autogluon.timeseries import (  # pyright: ignore[reportMissingImports]
        TimeSeriesDataFrame,
    )

    step = pd.Timedelta(minutes=interval_mins)
    context_frames: List[pd.DataFrame] = []
    known_frames: List[pd.DataFrame] = []
    item_ids: List[str] = []
    row_index: List[int] = []

    for pos, anchor_ts in zip(anchors.index, anchors["date"]):
        ctx = frame.loc[frame.index <= anchor_ts].tail(context_length)
        if ctx.empty:
            continue  # no history at all -> naive fallback fills this anchor
        # The competition requires a prediction for every anchor, so fill CGM
        # gaps in the context instead of skipping: interpolate interior gaps,
        # then ffill/bfill the edges, then fall back to the window median.
        target = ctx[TARGET_COL].interpolate(method="linear", limit_area="inside")
        target = target.ffill().bfill()
        if target.isna().all():
            continue  # no valid CGM in the window -> naive fallback
        target = target.fillna(target.median())

        item_id = f"{group_key}|{pos}"
        block = pd.DataFrame(index=ctx.index)
        block[TARGET_COL] = target.to_numpy()
        for col in past_cols:
            block[col] = ctx[col].ffill().fillna(0.0).to_numpy()
        for col in known_cols:
            block[col] = ctx[col].to_numpy()
        block["item_id"] = item_id
        block["timestamp"] = block.index
        context_frames.append(block)

        if known_cols:
            future_index = pd.date_range(
                anchor_ts + step, periods=forecast_length, freq=step
            )
            kf = pd.DataFrame(index=future_index)
            minutes = future_index.hour * 60 + future_index.minute
            if "hour_sin" in known_cols:
                kf["hour_sin"] = np.sin(2 * np.pi * minutes / 1440)
            if "hour_cos" in known_cols:
                kf["hour_cos"] = np.cos(2 * np.pi * minutes / 1440)
            # Non-deterministic known covariates (insulin channels routed as
            # known-future) are actually known at forecast time in the holdout:
            # pull their engineered future values from the full patient frame.
            future_frame = frame.reindex(future_index)
            for col in known_cols:
                if col in ("hour_sin", "hour_cos"):
                    continue
                kf[col] = future_frame[col].ffill().fillna(0.0).to_numpy()
            kf["item_id"] = item_id
            kf["timestamp"] = future_index
            known_frames.append(kf)

        item_ids.append(item_id)
        row_index.append(pos)

    if not item_ids:
        return pd.DataFrame(columns=["_row", *[f"pred_{m}" for m in HORIZON_MINUTES]])

    cols = ["item_id", "timestamp", TARGET_COL, *past_cols, *known_cols]
    context_long = pd.concat(context_frames, ignore_index=True)[cols].set_index(
        ["item_id", "timestamp"]
    )
    static_features = None
    if static_cols:
        record: Dict[str, list] = {"item_id": list(item_ids)}
        for col in static_cols:
            value = static_row.get(col, np.nan)
            record[col] = [value] * len(item_ids)
        static_features = pd.DataFrame(record).set_index("item_id")
        for col in static_cols:
            if col in ("gender", "source_file", "treatment_group", "ethnicity"):
                static_features[col] = (
                    static_features[col].astype("string").fillna("unknown")
                )
            else:
                static_features[col] = pd.to_numeric(
                    static_features[col], errors="coerce"
                )

    context_tsdf = TimeSeriesDataFrame(context_long, static_features=static_features)
    known_tsdf = None
    if known_frames:
        known_long = pd.concat(known_frames, ignore_index=True).set_index(
            ["item_id", "timestamp"]
        )
        known_tsdf = TimeSeriesDataFrame(known_long)

    forecast = predictor.predict(context_tsdf, known_covariates=known_tsdf)
    pred_df = forecast["mean"].reset_index()

    out_rows = []
    horizon = max(HORIZON_STEPS.values())
    for item_id, pos in zip(item_ids, row_index):
        sl = pred_df.loc[pred_df["item_id"] == item_id].sort_values("timestamp")
        values = sl["mean"].to_numpy()[:horizon]
        row = {"_row": pos}
        for minutes, s in HORIZON_STEPS.items():
            row[f"pred_{minutes}"] = values[s - 1] if values.size >= s else np.nan
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def _fill_naive_fallback(
    submission: pd.DataFrame,
    test_grouped,
) -> int:
    """Fill still-empty anchors with last-observation-carried-forward.

    The competition requires a prediction for every anchor. Anchors the model
    could not forecast (context entirely missing/degenerate) get the last
    observed CGM before the anchor carried across all horizons; if no prior CGM
    exists, the global median of the already-filled predictions (else 120) is
    used. Returns the number of anchors filled by the fallback.
    """
    pred_cols = [f"pred_{m}" for m in HORIZON_MINUTES]
    missing = submission[submission[pred_cols[0]].isna()]
    if missing.empty:
        return 0
    filled_vals = submission[pred_cols].to_numpy(dtype=float)
    global_default = (
        float(np.nanmedian(filled_vals)) if np.isfinite(filled_vals).any() else 120.0
    )
    for row_pos, arow in missing.iterrows():
        last = global_default
        try:
            raw = test_grouped.get_group((str(arow["id"]), str(arow["source_file"])))
            prior = raw.loc[pd.to_datetime(raw["date"]) <= arow["date"], "CGM"]
            prior = prior[prior.notna() & (prior > 0)]
            if len(prior):
                last = float(prior.iloc[-1])
        except KeyError:
            pass
        for col in pred_cols:
            submission.loc[row_pos, col] = last
    return len(missing)


def build_submission(
    predictor,
    template: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    past_cols: Sequence[str],
    known_cols: Sequence[str],
    static_cols: Sequence[str],
    context_length: int,
    forecast_length: int = 24,
    interval_mins: int = 5,
    limit_groups: Optional[int] = None,
) -> pd.DataFrame:
    """Return ``template`` with ``pred_*`` filled for the model's forecasts."""
    submission = template.copy()
    for minutes in HORIZON_MINUTES:
        submission[f"pred_{minutes}"] = np.nan

    groups = list(template.groupby(["id", "source_file"], sort=False))
    if limit_groups is not None:
        groups = groups[:limit_groups]

    test_df = test_df.copy()
    test_df["id"] = test_df["id"].astype(str)
    test_df["source_file"] = test_df["source_file"].astype(str)
    test_grouped = test_df.groupby(["id", "source_file"], sort=False)

    for n, ((gid, gsource), anchors) in enumerate(groups, start=1):
        key = f"{gid}|{gsource}"
        try:
            raw = test_grouped.get_group((str(gid), str(gsource)))
        except KeyError:
            logger.warning("No test series for %s; leaving NaN", key)
            continue
        raw = raw.rename(columns={"date": "datetime"})
        frame = engineer_patient_frame(raw, interval_mins=interval_mins)
        if SEX_KNOWN_COL in known_cols:
            # Broadcast the patient's sex as a constant known covariate (matches
            # how it is routed at train time). Gender is read from the holdout data.
            gender_val = (
                raw["gender"].dropna().iloc[0]
                if "gender" in raw.columns and raw["gender"].notna().any()
                else np.nan
            )
            frame[SEX_KNOWN_COL] = encode_sex(gender_val)
        demo_cols_needed = [c for c in DEMOGRAPHIC_KNOWN_COLS if c in known_cols]
        if demo_cols_needed:
            # Broadcast the patient's demographics as constant known covariates
            # (matches train-time routing). age/height/weight come from the holdout
            # parquet; ethnicity/age_of_diagnosis come from the re-id lookup (empty
            # in the holdout). Missing values degrade to the neutral cohort center.
            holdout_static: Dict[str, object] = {
                "age": _first_value(raw, "age"),
                "height": _first_value(raw, "height"),
                "weight": _first_value(raw, "weight"),
            }
            reid = _load_reid_demographics().get(str(gid))
            if reid:
                holdout_static["ethnicity"] = reid.get("ethnicity")
                holdout_static["age_of_diagnosis"] = reid.get("age_of_diagnosis")
            for col, val in encode_demographics(holdout_static).items():
                if col in known_cols:
                    frame[col] = val
        static_row = {
            col: (raw[col].dropna().iloc[0] if raw[col].notna().any() else np.nan)
            for col in static_cols
            if col in raw.columns
        }
        preds = _forecast_group(
            predictor,
            frame,
            anchors.reset_index().rename(columns={"index": "_row"}).set_index("_row"),
            group_key=key,
            context_length=context_length,
            forecast_length=forecast_length,
            interval_mins=interval_mins,
            past_cols=past_cols,
            known_cols=known_cols,
            static_cols=static_cols,
            static_row=static_row,
        )
        for _, prow in preds.iterrows():
            row = int(prow["_row"])
            for minutes in HORIZON_MINUTES:
                submission.iat[row, submission.columns.get_loc(f"pred_{minutes}")] = (
                    prow[f"pred_{minutes}"]
                )
        logger.info(
            "Group %d/%d %s: %d anchors forecast", n, len(groups), key, len(preds)
        )

    n_fallback = _fill_naive_fallback(submission, test_grouped)
    if n_fallback:
        logger.info(
            "Naive fallback filled %d anchors the model could not forecast",
            n_fallback,
        )

    return submission


def score_submission(
    submission: pd.DataFrame, targets: pd.DataFrame
) -> Dict[str, Dict[str, float]]:
    """Self-score populated rows against targets (RMSE/MARD/DTS per horizon)."""
    merged = submission.merge(
        targets, on=["id", "source_file", "date"], how="left", suffixes=("", "_t")
    )
    results: Dict[str, Dict[str, float]] = {}
    for minutes in HORIZON_MINUTES:
        pred = merged[f"pred_{minutes}"].to_numpy()
        true = merged[f"target_{minutes}"].to_numpy()
        mask = np.isfinite(pred) & np.isfinite(true)
        if not mask.any():
            continue
        p, t = pred[mask], true[mask]
        zones = dts_error_grid(p, t)
        results[str(minutes)] = {
            "n": int(mask.sum()),
            "rmse": rmse(p, t),
            "mard": mard(p, t),
            "dts_a_b_pct": zones["dts_a_pct"] + zones["dts_b_pct"],
        }
    return results


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage", required=True)
    parser.add_argument("--study-name", required=True)
    parser.add_argument("--models-root", required=True)
    parser.add_argument("--template", required=True)
    parser.add_argument(
        "--test-parquet", default="cache/data/metabonet/raw/test.parquet"
    )
    parser.add_argument("--targets", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--trial", type=int, default=None, help="Trial number (default: best)."
    )
    parser.add_argument("--context-length", type=int, default=None)
    parser.add_argument("--limit-groups", type=int, default=None)
    parser.add_argument("--score", action="store_true")
    args = parser.parse_args()

    from autogluon.timeseries import (  # pyright: ignore[reportMissingImports]
        TimeSeriesPredictor,
    )

    study = optuna.load_study(study_name=args.study_name, storage=args.storage)
    trial = _best_trial(study, args.trial)
    past_cols = list(trial.user_attrs.get("past_cols", []))
    known_cols = list(trial.user_attrs.get("known_cols", []))
    static_cols = list(trial.user_attrs.get("static_cols", []))
    context_length = args.context_length or int(trial.params.get("context_length", 288))
    model_dir = Path(args.models_root) / f"trial_{trial.number:05d}"
    print(
        f"Using trial {trial.number} ({study.metric_names or ['value']}"
        f"={trial.value:.3f}) context={context_length}\n"
        f"  past={past_cols} known={known_cols} static={static_cols}"
    )

    predictor = TimeSeriesPredictor.load(str(model_dir))
    template = pd.read_parquet(args.template)
    template["id"] = template["id"].astype(str)
    template["source_file"] = template["source_file"].astype(str)
    test_df = pd.read_parquet(
        args.test_parquet,
        columns=[c for c in _TEST_COLUMNS],
    )

    submission = build_submission(
        predictor,
        template,
        test_df,
        past_cols=past_cols,
        known_cols=known_cols,
        static_cols=static_cols,
        context_length=context_length,
        limit_groups=args.limit_groups,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    submission.to_parquet(out, index=False)
    n_filled = int(submission["pred_120"].notna().sum())
    print(f"Wrote {out} ({n_filled:,}/{len(submission):,} rows populated)")

    if args.score:
        if not args.targets:
            print("--score requires --targets")
            return 1
        targets = pd.read_parquet(args.targets)
        targets["id"] = targets["id"].astype(str)
        targets["source_file"] = targets["source_file"].astype(str)
        scores = score_submission(submission, targets)
        print("\nLocal self-score (populated rows only):")
        print(f"{'horizon':>8} {'n':>8} {'rmse':>8} {'mard%':>7} {'DTS A+B%':>9}")
        for minutes in HORIZON_MINUTES:
            s = scores.get(str(minutes))
            if s:
                print(
                    f"{minutes:>6}m {s['n']:>8} {s['rmse']:>8.2f} "
                    f"{s['mard']:>7.2f} {s['dts_a_b_pct']:>9.1f}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
