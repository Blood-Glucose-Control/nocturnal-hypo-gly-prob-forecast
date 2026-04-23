#!/usr/bin/env python3
# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)

"""
Compare Lynch 2022 data loading: new txt-based pipeline vs old cached SAS data.

Checks:
  1. Patient coverage: patient IDs present in each source
  2. Row counts / BG coverage per patient
  3. Timestamp quality: fraction of timestamps on exact 5-min boundaries
  4. Potential midnight episode count for holdout patients
  5. BG value alignment between our loader and the IOBP2Adapter (BabelBetes-compatible)

Usage:
    python scripts/data_processing_scripts/compare_lynch_data_sources.py
"""

import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

# Allow running from the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.cache_manager import get_cache_manager
from src.data.diabetes_datasets.lynch_2022.babelbetes_adapter import IOBP2Adapter
from src.data.diabetes_datasets.lynch_2022.data_cleaner import (
    clean_lynch2022_train_data,
    load_lynch2022_raw_dataset,
)
from src.data.utils.patient_id import format_patient_id

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

MG_DL_TO_MMOL = 1 / 18.0
HOLDOUT_CONFIG = Path("configs/data/holdout_10pct/lynch_2022.yaml")
EPISODE_WINDOW_HOURS = 8
EPISODE_FREQ_MIN = 5
EPISODE_LEN = EPISODE_WINDOW_HOURS * 60 // EPISODE_FREQ_MIN  # 96 steps


def _load_holdout_patients() -> list[str]:
    with open(HOLDOUT_CONFIG) as f:
        cfg = yaml.safe_load(f)
    return cfg["patient_config"]["holdout_patients"]


def _count_potential_midnight_episodes(df: pd.DataFrame, p_num: str) -> int:
    """
    Count nights where at least EPISODE_LEN rows exist after midnight.
    Timestamps must already be on 5-min boundaries.
    """
    if df.empty:
        return 0
    idx = pd.DatetimeIndex(df.index if isinstance(df.index, pd.DatetimeIndex) else df["datetime"])
    midnights = pd.date_range(idx.normalize().min(), idx.normalize().max(), freq="D")
    count = 0
    for midnight in midnights:
        window_end = midnight + pd.Timedelta(hours=EPISODE_WINDOW_HOURS)
        mask = (idx >= midnight) & (idx < window_end)
        if mask.sum() >= EPISODE_LEN:
            count += 1
    return count


def _pct_on_boundary(timestamps: pd.DatetimeIndex, freq_min: int = 5) -> float:
    """Fraction of timestamps that fall exactly on a 5-min boundary."""
    if len(timestamps) == 0:
        return 0.0
    seconds = timestamps.second + timestamps.microsecond / 1e6
    minute_mod = timestamps.minute % freq_min
    on_boundary = (minute_mod == 0) & (seconds == 0)
    return on_boundary.mean() * 100


def section(title: str):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print("=" * 70)


def main():
    cache_manager = get_cache_manager()
    raw_root = cache_manager.get_absolute_path_by_type("lynch_2022", "raw")
    txt_base = raw_root / "IOBP2 RCT Public Dataset" / "Data Tables"
    holdout_patients = _load_holdout_patients()

    # ------------------------------------------------------------------ #
    # 1. Load via new txt-based pipeline                                  #
    # ------------------------------------------------------------------ #
    section("1. Loading via NEW txt-based pipeline")
    print(f"   Source: {txt_base}")
    raw_new = load_lynch2022_raw_dataset(txt_base)
    cleaned_new = clean_lynch2022_train_data(raw_new)
    new_patients = set(cleaned_new["p_num"].unique())
    print(f"   Patients: {len(new_patients)}")
    print(f"   Total rows: {len(cleaned_new):,}")
    print(f"   Date range: {cleaned_new['datetime'].min()} → {cleaned_new['datetime'].max()}")
    ts_new = pd.DatetimeIndex(cleaned_new["datetime"])
    print(f"   Timestamps on 5-min boundary: {_pct_on_boundary(ts_new):.1f}%")

    # ------------------------------------------------------------------ #
    # 2. Check old processed cache (if it exists)                         #
    # ------------------------------------------------------------------ #
    section("2. Old processed cache (pre-fix)")
    processed_dir = cache_manager.get_processed_data_path("lynch_2022")
    old_cache_files = sorted(processed_dir.glob("lyn_*_full.csv"))
    if old_cache_files:
        print(f"   Found {len(old_cache_files)} cached patient files")
        # Sample first holdout patient
        sample_pid = holdout_patients[0]
        sample_path = processed_dir / f"{sample_pid}_full.csv"
        if sample_path.exists():
            old_df = pd.read_csv(sample_path, index_col="datetime", parse_dates=True)
            ts_old = old_df.index
            print(f"   Sample patient ({sample_pid}): {len(old_df)} rows")
            print(f"   Timestamps on 5-min boundary: {_pct_on_boundary(ts_old):.1f}%")
            print(f"   First 3 timestamps: {list(ts_old[:3])}")
    else:
        print("   No cached files found (cache already cleared)")

    # ------------------------------------------------------------------ #
    # 3. BabelBetes-adapter comparison (CGM alignment)                    #
    # ------------------------------------------------------------------ #
    section("3. BabelBetes adapter vs our loader — CGM alignment")
    adapter = IOBP2Adapter(raw_root / "IOBP2 RCT Public Dataset")
    bb_cgm = adapter.extract_cgm_history()
    print(f"   BabelBetes patients: {len(bb_cgm)}")

    # Compare for a sample holdout patient (use raw numeric ID)
    sample_pid_str = holdout_patients[0]  # e.g. "lyn_372"
    raw_id = sample_pid_str.split("_")[1]  # "372"
    if raw_id in bb_cgm:
        bb_pt = bb_cgm[raw_id].set_index("datetime")["cgm_mgdl"]
        our_pt = cleaned_new[cleaned_new["p_num"] == sample_pid_str].set_index("datetime")["bg_mM"]
        our_pt_mgdl = our_pt * 18.0

        # Round both to nearest 5-min for alignment comparison
        bb_rounded = bb_pt.copy()
        bb_rounded.index = bb_rounded.index.round("5min")
        bb_rounded = bb_rounded[~bb_rounded.index.duplicated(keep="first")]
        our_rounded = our_pt_mgdl.copy()
        our_rounded.index = our_rounded.index.round("5min")
        our_rounded = our_rounded[~our_rounded.index.duplicated(keep="first")]

        common_idx = bb_rounded.index.intersection(our_rounded.index)
        if len(common_idx) > 0:
            diff = (our_rounded.loc[common_idx] - bb_rounded.loc[common_idx]).abs()
            print(f"   Patient {sample_pid_str}: {len(common_idx)} aligned timestamps")
            print(f"   Mean |ΔBGM| (mg/dL): {diff.mean():.3f}")
            print(f"   Max  |ΔBGM| (mg/dL): {diff.max():.3f}")
            threshold_ok = (diff <= 0.5).mean() * 100
            print(f"   Within 0.5 mg/dL: {threshold_ok:.1f}%  (expect ~100%: same raw CGMVal, no timestamp shift)")
        else:
            print(f"   No common timestamps for {sample_pid_str}")
    else:
        print(f"   Patient {raw_id} not found in BabelBetes adapter output")

    # ------------------------------------------------------------------ #
    # 4. Potential midnight episode counts — holdout patients             #
    # ------------------------------------------------------------------ #
    section("4. Potential midnight episodes — holdout patients")
    print(f"   Checking {len(holdout_patients)} holdout patients...")
    episode_results = []
    for p_num in holdout_patients:
        pt_df = cleaned_new[cleaned_new["p_num"] == p_num].set_index("datetime")
        n_episodes = _count_potential_midnight_episodes(pt_df, p_num)
        episode_results.append({"patient": p_num, "rows": len(pt_df), "potential_episodes": n_episodes})

    ep_df = pd.DataFrame(episode_results)
    total_eps = ep_df["potential_episodes"].sum()
    covered = (ep_df["potential_episodes"] > 0).sum()
    print(f"   Total potential episodes: {total_eps}")
    print(f"   Patients with ≥1 episode: {covered} / {len(holdout_patients)}")
    print(f"\n   Per-patient breakdown:")
    print(ep_df.to_string(index=False))

    # ------------------------------------------------------------------ #
    # 5. Summary                                                          #
    # ------------------------------------------------------------------ #
    section("5. Summary")
    # Note: ts_pct checks the pre-aggregation cleaning stage (0.5% is expected here;
    # timestamps are rounded to 5-min boundaries by preprocessing_pipeline).
    # The processed cache at cache/data/lynch_2022/processed/ should show 100%.
    processed_dir = cache_manager.get_processed_data_path("lynch_2022")
    sample_processed_path = processed_dir / f"{holdout_patients[0]}_full.csv"
    if sample_processed_path.exists():
        proc_df = pd.read_csv(sample_processed_path, index_col="datetime", parse_dates=True)
        ts_proc = proc_df.index
        proc_pct = _pct_on_boundary(ts_proc)
        status = "PASS" if proc_pct == 100.0 else "FAIL"
        print(f"   Processed cache timestamp alignment [{status}]: {proc_pct:.1f}% on 5-min boundary")
    else:
        print("   Processed cache not found — run Lynch2022DataLoader(use_cached=False) first")
    status = "PASS" if total_eps >= 1000 else ("WARN" if total_eps >= 100 else "FAIL")
    print(f"   Episode count [{status}]: {total_eps} potential holdout episodes")
    print(f"   (Pre-fix count was ~57; expect ≥2000 after cache regeneration)")


if __name__ == "__main__":
    main()
