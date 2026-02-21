"""
Experiment script to explore build_midnight_episodes() output on real holdout data.

Run from repo root (src/ must be on PYTHONPATH for bare `data.*` imports):
    PYTHONPATH=src python scripts/scratch/build_midnight_episode_experiment.py
    PYTHONPATH=src python scripts/scratch/build_midnight_episode_experiment.py --boundary-only
    PYTHONPATH=src python scripts/scratch/build_midnight_episode_experiment.py --dataset lynch_2022
    PYTHONPATH=src python scripts/scratch/build_midnight_episode_experiment.py --max-patients 3

#TODO: This can be moved to documentation later to explain episode building behaviour.
"""

import argparse

import numpy as np
import pandas as pd

from src.data.versioning.dataset_registry import DatasetRegistry
from src.data.utils import get_patient_column
from src.evaluation.episode_builders import (
    build_midnight_episodes,
    SAMPLING_INTERVAL_MINUTES,
)

CONTEXT_LENGTH = 512  # ~42.7 hours at 5-min sampling
FORECAST_LENGTH = 72  # 6 hours
HYPO_THRESHOLD = 3.9  # mmol/L
COVARIATE_COLS = ["iob", "cob"]


def describe_episode(ep: dict, idx: int, n_rows: int = 3) -> None:
    ctx = ep["context_df"]
    tgt = ep["target_bg"]

    hypo_steps = int((tgt < HYPO_THRESHOLD).sum())
    hypo_pct = 100 * hypo_steps / len(tgt)

    print(
        f"\n  ep{idx:03d}  anchor={ep['anchor'].date()}  "
        f"ctx_bg=[{ctx['bg_mM'].min():.1f}, {ctx['bg_mM'].max():.1f}] mmol/L  "
        f"tgt_bg=[{tgt.min():.1f}, {tgt.max():.1f}]  "
        f"hypo_steps={hypo_steps}/{len(tgt)} ({hypo_pct:.0f}%)  "
        f"covariates={list(ep['future_covariates'].keys()) or 'none'}"
    )

    print(f"    context_df ({len(ctx)} rows):")
    print(ctx.head(n_rows).to_string(max_cols=None).replace("\n", "\n    "))
    print("    ...")
    print(ctx.tail(n_rows).to_string(max_cols=None).replace("\n", "\n    "))

    # Show target BG as a small table
    tgt_index = pd.date_range(ep["anchor"], periods=len(tgt), freq="5min")
    tgt_df = pd.DataFrame({"bg_mM": tgt}, index=tgt_index)
    print(f"\n    target_bg ({len(tgt_df)} rows, anchor={ep['anchor']}):")
    print(tgt_df.head(n_rows).to_string().replace("\n", "\n    "))
    print("    ...")
    print(tgt_df.tail(n_rows).to_string().replace("\n", "\n    "))


def bg_gap_summary(
    patient_df: pd.DataFrame, interval_mins: int = SAMPLING_INTERVAL_MINUTES
) -> None:
    """Print a quick summary of BG gap sizes in the raw data before reindexing."""
    bg = patient_df["bg_mM"].sort_index()
    freq = f"{interval_mins}min"
    grid = pd.date_range(
        bg.index.min().floor(freq), bg.index.max().floor(freq), freq=freq
    )
    reindexed = bg.reindex(grid)

    nan_mask = reindexed.isna()
    if not nan_mask.any():
        print("    BG gaps: none")
        return

    # Count consecutive-NaN run lengths
    run_lengths = []
    count = 0
    for v in nan_mask:
        if v:
            count += 1
        elif count > 0:
            run_lengths.append(count)
            count = 0
    if count > 0:
        run_lengths.append(count)

    run_lengths = np.array(run_lengths)
    short = (run_lengths <= 2).sum()
    long_ = (run_lengths > 2).sum()
    print(
        f"    BG gaps: {len(run_lengths)} runs  "
        f"(<=2 steps / fillable: {short}, >2 steps / skip-worthy: {long_})  "
        f"longest={run_lengths.max()} steps ({run_lengths.max() * interval_mins} min)"
    )


def run_patient(patient_id, patient_df: pd.DataFrame) -> None:
    print(f"\n{'─' * 60}")
    print(
        f"Patient: {patient_id}  rows={len(patient_df)}  "
        f"span={patient_df.index.min().date()} -> {patient_df.index.max().date()}"
    )
    print(f"  Columns: {list(patient_df.columns)}")
    print(
        f"  BG: min={patient_df['bg_mM'].min():.2f}  "
        f"max={patient_df['bg_mM'].max():.2f}  "
        f"pct_nan={100 * patient_df['bg_mM'].isna().mean():.1f}%"
    )
    bg_gap_summary(patient_df)

    # -- Run episode builder with interpolation enabled (default) ------------
    avail_covs = [c for c in COVARIATE_COLS if c in patient_df.columns]
    episodes, skip_stats = build_midnight_episodes(
        patient_df,
        context_length=CONTEXT_LENGTH,
        forecast_length=FORECAST_LENGTH,
        covariate_cols=avail_covs,
        max_bg_gap_steps=2,
    )

    print("\n  Skip stats (max_bg_gap_steps=2):")
    print(f"    total_anchors      = {skip_stats['total_anchors']}")
    print(f"    skipped_bg_nan     = {skip_stats['skipped_bg_nan']}")
    print(f"    interpolated       = {skip_stats['interpolated_episodes']}")
    print(f"    valid episodes     = {len(episodes)}")
    if skip_stats["skipped_anchors"]:
        dates = [str(a.date()) for a in skip_stats["skipped_anchors"][:5]]
        suffix = "..." if len(skip_stats["skipped_anchors"]) > 5 else ""
        print(f"    skipped on dates   = {dates}{suffix}")

    # -- Run again with interpolation disabled to measure rescue rate --------
    episodes_noint, _ = build_midnight_episodes(
        patient_df,
        context_length=CONTEXT_LENGTH,
        forecast_length=FORECAST_LENGTH,
        covariate_cols=avail_covs,
        max_bg_gap_steps=0,
    )
    rescued = len(episodes) - len(episodes_noint)
    print(
        f"\n  Episodes without interpolation (max_bg_gap_steps=0): {len(episodes_noint)}"
    )
    if rescued > 0:
        print(f"  -> interpolation rescued {rescued} episode(s)")

    # -- Sample episodes -----------------------------------------------------
    if episodes:
        n = len(episodes)
        first_n = min(2, n)
        last_n = min(2, max(0, n - first_n))

        print(f"\n  First {first_n} episode(s) (of {n}):")
        for i in range(first_n):
            describe_episode(episodes[i], i)

        if last_n > 0:
            last_start = n - last_n
            print(f"\n  Last {last_n} episode(s) (of {n}):")
            for i in range(last_start, n):
                describe_episode(episodes[i], i)


def run_boundary_experiment() -> None:
    """
    Synthetic experiment: data starts and ends exactly on midnight.

    With context_length=12 (1h) and forecast_length=6 (30min):

      Data:  2024-01-01 00:00  ->  2024-01-04 00:00  (inclusive)
      Grid:  288 steps of 5 min

    Anchors that *could* exist: Jan 1, Jan 2, Jan 3, Jan 4 midnights.

    - Jan 1 midnight: no context before it -> skipped (earliest anchor constraint)
    - Jan 4 midnight: no forecast after it -> skipped (latest anchor constraint)
    - Jan 2 and Jan 3: should be valid

    This shows the half-open boundary: the data endpoints are midnights, but
    those endpoint midnights themselves are excluded as anchors.
    """
    ctx_len = 12  # 1 hour of context at 5-min sampling
    fct_len = 6  # 30 minutes of forecast

    # Build exactly midnight-to-midnight data (Jan 1 00:00 through Jan 4 00:00 inclusive)
    start = pd.Timestamp("2024-01-01 00:00")
    end = pd.Timestamp("2024-01-04 00:00")
    index = pd.date_range(start, end, freq="5min")
    df = pd.DataFrame({"bg_mM": np.linspace(5.0, 9.0, len(index))}, index=index)

    print("\n" + "=" * 60)
    print("BOUNDARY EXPERIMENT: data starts and ends on midnight")
    print("=" * 60)
    print(f"  Data span : {df.index[0]}  ->  {df.index[-1]}")
    print(f"  Steps     : {len(df)}  ({len(df) * 5 / 60:.1f} h)")
    print(
        f"  context_length={ctx_len} ({ctx_len * 5} min),  "
        f"forecast_length={fct_len} ({fct_len * 5} min)"
    )

    episodes, skip_stats = build_midnight_episodes(
        df, context_length=ctx_len, forecast_length=fct_len
    )

    print(f"\n  total_anchors    = {skip_stats['total_anchors']}")
    print(f"  skipped_bg_nan   = {skip_stats['skipped_bg_nan']}")
    print(f"  valid episodes   = {len(episodes)}")
    print(f"  anchors          = {[ep['anchor'] for ep in episodes]}")

    print("\n  expected: Jan 1 and Jan 4 midnight are EXCLUDED")
    print("            Jan 2 and Jan 3 midnight are INCLUDED")

    for i, ep in enumerate(episodes):
        describe_episode(ep, i, n_rows=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="brown_2019")
    parser.add_argument("--config-dir", default="configs/data/holdout_5pct")
    parser.add_argument("--max-patients", type=int, default=5)
    parser.add_argument(
        "--boundary-only",
        action="store_true",
        help="Only run the boundary experiment, skip real data load",
    )
    args = parser.parse_args()

    run_boundary_experiment()

    if args.boundary_only:
        return

    print(f"Dataset : {args.dataset}")
    print(f"Config  : {args.config_dir}")
    print(
        f"Context : {CONTEXT_LENGTH} steps ({CONTEXT_LENGTH * SAMPLING_INTERVAL_MINUTES / 60:.1f} h)"
    )
    print(
        f"Forecast: {FORECAST_LENGTH} steps ({FORECAST_LENGTH * SAMPLING_INTERVAL_MINUTES / 60:.1f} h)"
    )

    registry = DatasetRegistry(holdout_config_dir=args.config_dir)
    holdout_data = registry.load_holdout_data_only(args.dataset)

    patient_col = get_patient_column(holdout_data)
    patients = holdout_data[patient_col].unique()[: args.max_patients]
    print(
        f"\nPatients in holdout: {holdout_data[patient_col].nunique()}  "
        f"(showing {len(patients)})"
    )

    for patient_id in patients:
        patient_df = holdout_data[holdout_data[patient_col] == patient_id].copy()

        if not isinstance(patient_df.index, pd.DatetimeIndex):
            time_col = "datetime" if "datetime" in patient_df.columns else None
            if time_col:
                patient_df[time_col] = pd.to_datetime(patient_df[time_col])
                patient_df = patient_df.set_index(time_col).sort_index()

        run_patient(patient_id, patient_df)

    print(f"\n{'=' * 60}")


if __name__ == "__main__":
    main()
