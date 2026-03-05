# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)
# For commercial licensing, contact: cjrisi/christopher AT uwaterloo/gluroo DOT ca/com

"""
Visualize Chronos-2 forecasts for specific midnight-anchored episodes.

Generates per-episode and combined comparison plots for two Brown 2019
episodes (bro_149::ep056, bro_171::ep087). Each plot shows:
- Context window (last 2 h) + 8 h forecast horizon
- True BG vs predicted BG mean
- 10th–90th quantile bands
- IOB on a secondary axis (if trained with covariates)
- 3.9 mmol/L hypoglycemia threshold

Usage:
    python scripts/visualization/visualize_chronos2_episodes.py

    # Override defaults
    python scripts/visualization/visualize_chronos2_episodes.py \
        --predictor-path trained_models/artifacts/chronos2/<run_dir> \
        --output-dir images/figures/chronos2_episodes
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Ensure repo root is importable ──────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# ── Constants ───────────────────────────────────────────────────────────
HYPO_THRESHOLD = 3.9  # mmol/L
CONTEXT_LENGTH = 512
FORECAST_HORIZON = 96  # 8 hours at 5-min intervals
INTERVAL_MINS = 5

# Episodes to visualize
EPISODES = {
    "bro_149": {
        "episode_id": "bro_149::ep056",
        "anchor": "2018-07-29 23:55:00",
        "midnight": "2018-07-30 00:00:00",
    },
    "bro_171": {
        "episode_id": "bro_171::ep087",
        "anchor": "2018-08-23 23:55:00",
        "midnight": "2018-08-24 00:00:00",
    },
}

# Default predictor path (BG + IOB + insulin_availability, 50k steps)
DEFAULT_PREDICTOR_PATH = str(
    REPO_ROOT
    / "trained_models"
    / "artifacts"
    / "chronos2"
    / "2026-02-28_05:57_RID20260228_055715_392456_holdout_workflow"
)

# Alternative: BG-only predictor (50k steps)
# DEFAULT_PREDICTOR_PATH = str(
#     REPO_ROOT / "trained_models" / "artifacts" / "chronos2"
#     / "2026-02-28_05:54_RID20260228_055400_391511_holdout_workflow"
# )

DEFAULT_OUTPUT_DIR = str(REPO_ROOT / "images" / "figures" / "chronos2_episodes")


# ── Data Loading & Prediction ──────────────────────────────────────────


def load_patient_data(pid: str) -> pd.DataFrame:
    """Load and concatenate train + validation data for a Brown 2019 patient."""
    from src.data.diabetes_datasets.data_loader import get_loader

    loader = get_loader(data_source_name="brown_2019", dataset_type="train", use_cached=True)

    dfs = []
    if pid in loader.train_data:
        dfs.append(loader.train_data[pid])
    if pid in loader.validation_data:
        dfs.append(loader.validation_data[pid])

    if not dfs:
        raise ValueError(f"Patient {pid} not found in Brown 2019 dataset")

    patient_df = pd.concat(dfs).sort_index()
    return patient_df


def generate_predictions(
    predictor_path: str,
) -> dict[str, dict]:
    """Run Chronos-2 inference for specified episodes.

    Returns a dict keyed by patient ID with context, target, predictions,
    and quantile information.
    """
    from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

    print(f"Loading predictor from: {predictor_path}")
    predictor = TimeSeriesPredictor.load(predictor_path)

    # Load model config to check for covariates
    config_path = os.path.join(predictor_path, "model_config.yaml")
    covariate_cols: list[str] = []
    if os.path.exists(config_path):
        import yaml

        with open(config_path) as f:
            model_config = yaml.safe_load(f)
        covariate_cols = model_config.get("covariate_cols", [])
        print(f"  Covariates from config: {covariate_cols}")

    results: dict[str, dict] = {}

    for pid, ep_info in EPISODES.items():
        midnight = pd.Timestamp(ep_info["midnight"])
        print(f"\nProcessing {pid} (midnight={midnight}) …")

        patient_df = load_patient_data(pid)

        # Context: 512 steps ending at 23:55 (inclusive)
        ctx_end = midnight - pd.Timedelta(minutes=5)
        ctx_start = midnight - pd.Timedelta(minutes=5 * CONTEXT_LENGTH)
        fh_end = midnight + pd.Timedelta(minutes=5 * FORECAST_HORIZON)

        context_bg = patient_df.loc[ctx_start:ctx_end, "bg_mM"].values[-CONTEXT_LENGTH:]
        target_bg = patient_df.loc[midnight:fh_end, "bg_mM"].values[:FORECAST_HORIZON]

        # Build TimeSeriesDataFrame for context
        timestamps = pd.date_range(
            end=midnight - pd.Timedelta(minutes=5),
            periods=CONTEXT_LENGTH,
            freq="5min",
        )
        data: dict[str, object] = {
            "item_id": [pid] * CONTEXT_LENGTH,
            "timestamp": timestamps,
            "target": context_bg,
        }

        # Add covariates if the model was trained with them
        for cov_col in covariate_cols:
            if cov_col in patient_df.columns:
                cov_vals = patient_df.loc[ctx_start:ctx_end, cov_col].values[
                    -CONTEXT_LENGTH:
                ]
                data[cov_col] = cov_vals

        tsdf = TimeSeriesDataFrame.from_data_frame(pd.DataFrame(data))

        # Build known_covariates for the forecast horizon
        known_covariates = None
        active_covs = [c for c in covariate_cols if c in patient_df.columns]
        if active_covs:
            fh_timestamps = pd.date_range(
                start=midnight, periods=FORECAST_HORIZON, freq="5min"
            )
            kc_data: dict[str, object] = {
                "item_id": [pid] * FORECAST_HORIZON,
                "timestamp": fh_timestamps,
            }
            for cov_col in active_covs:
                kc_data[cov_col] = patient_df.loc[midnight:fh_end, cov_col].values[
                    :FORECAST_HORIZON
                ]
            known_covariates = TimeSeriesDataFrame.from_data_frame(
                pd.DataFrame(kc_data)
            )

        # Predict — AutoGluon returns columns: "mean", "0.1", "0.5", "0.9"
        predictions = predictor.predict(tsdf, known_covariates=known_covariates)

        mean = predictions["mean"].values
        q10 = predictions["0.1"].values
        q90 = predictions["0.9"].values

        # RMSE over available target
        n = min(len(mean), len(target_bg))
        rmse = float(np.sqrt(np.nanmean((mean[:n] - target_bg[:n]) ** 2)))

        print(f"  {pid}: RMSE={rmse:.3f}, pred_len={len(mean)}, target_len={len(target_bg)}")

        # Collect IOB for the forecast window (for plotting)
        forecast_iob = None
        if "iob" in patient_df.columns:
            forecast_iob = patient_df.loc[midnight:fh_end, "iob"].values[
                :FORECAST_HORIZON
            ].tolist()

        # IOB in context (for plotting)
        context_iob = None
        if "iob" in patient_df.columns:
            context_iob = patient_df.loc[ctx_start:ctx_end, "iob"].values[
                -CONTEXT_LENGTH:
            ].tolist()

        results[pid] = {
            "episode_id": ep_info["episode_id"],
            "anchor": ep_info["anchor"],
            "midnight": ep_info["midnight"],
            "context_bg": context_bg.tolist(),
            "target_bg": target_bg.tolist(),
            "mean": mean.tolist(),
            "q10": q10.tolist(),
            "q90": q90.tolist(),
            "rmse": rmse,
            "context_iob": context_iob,
            "forecast_iob": forecast_iob,
        }

    return results


# ── Visualization ───────────────────────────────────────────────────────


def _time_axis(n: int, offset_hours: float = 0.0) -> np.ndarray:
    """Return time axis in hours relative to midnight."""
    return np.arange(n) * INTERVAL_MINS / 60 + offset_hours


def plot_single_episode(
    pid: str,
    ep: dict,
    output_dir: str,
    context_hours: float = 2.0,
) -> str:
    """Generate a single-episode plot with context + forecast.

    Shows the last `context_hours` of the context window and the full
    8 h forecast horizon.

    Returns path to saved PNG.
    """
    matplotlib.use("Agg")

    ctx_bg = np.array(ep["context_bg"])
    target_bg = np.array(ep["target_bg"])
    pred_mean = np.array(ep["mean"])
    pred_q10 = np.array(ep["q10"])
    pred_q90 = np.array(ep["q90"])

    # Number of context points to show (last N hours before midnight)
    ctx_show = int(context_hours * 60 / INTERVAL_MINS)
    ctx_bg_show = ctx_bg[-ctx_show:]

    # Time axes (hours relative to midnight)
    t_ctx = _time_axis(ctx_show, offset_hours=-context_hours)
    t_fh = _time_axis(len(pred_mean), offset_hours=0.0)
    t_target = _time_axis(len(target_bg), offset_hours=0.0)

    fig, ax1 = plt.subplots(figsize=(6, 5))

    # ── Context BG ──────────────────────────────────────────────────
    ax1.plot(
        t_ctx,
        ctx_bg_show,
        color="black",
        linewidth=2.0,
        label="Actual BG (context)",
        zorder=10,
    )

    # ── Target (ground truth) BG ────────────────────────────────────
    ax1.plot(
        t_target,
        target_bg,
        color="black",
        linewidth=2.0,
        linestyle="--",
        label="Actual BG (target)",
        zorder=10,
    )

    # ── Predicted mean ──────────────────────────────────────────────
    ax1.plot(
        t_fh,
        pred_mean,
        color="steelblue",
        linewidth=2.0,
        label=f"Chronos-2 Predicted (RMSE={ep['rmse']:.2f})",
        zorder=8,
    )

    # ── Quantile band ───────────────────────────────────────────────
    ax1.fill_between(
        t_fh,
        pred_q10,
        pred_q90,
        color="steelblue",
        alpha=0.15,
        label="10th–90th percentile",
        zorder=3,
    )

    # ── Hypo threshold ──────────────────────────────────────────────
    ax1.axhline(
        y=HYPO_THRESHOLD,
        color="red",
        linestyle=":",
        linewidth=1.2,
        alpha=0.6,
        label=f"Hypo threshold ({HYPO_THRESHOLD} mmol/L)",
    )

    # ── Midnight line ───────────────────────────────────────────────
    ax1.axvline(x=0, color="gray", linestyle="-", linewidth=0.8, alpha=0.5)
    ax1.text(
        0.05,
        ax1.get_ylim()[1] * 0.95,
        "Midnight",
        fontsize=8,
        color="gray",
        ha="left",
        va="top",
    )

    # ── IOB on secondary axis ───────────────────────────────────────
    has_iob = ep.get("forecast_iob") is not None and ep.get("context_iob") is not None
    if has_iob:
        ax2 = ax1.twinx()
        ctx_iob = np.array(ep["context_iob"])[-ctx_show:]
        fh_iob = np.array(ep["forecast_iob"])

        ax2.plot(
            t_ctx,
            ctx_iob,
            color="gray",
            linestyle="--",
            linewidth=1.0,
            alpha=0.45,
        )
        ax2.plot(
            t_fh[: len(fh_iob)],
            fh_iob,
            color="gray",
            linestyle="--",
            linewidth=1.0,
            alpha=0.45,
            label="IOB (U)",
        )
        ax2.set_ylabel("IOB (U)", fontsize=10, color="gray")
        ax2.tick_params(axis="y", labelcolor="gray", labelsize=8)
        max_iob = max(np.max(ctx_iob), np.max(fh_iob)) if len(fh_iob) > 0 else 1
        ax2.set_ylim(0, max(max_iob * 1.6, 0.5))

    # ── Labels & formatting ─────────────────────────────────────────
    ax1.set_xlabel("Hours relative to midnight", fontsize=11)
    ax1.set_ylabel("Blood Glucose (mmol/L)", fontsize=11)
    ax1.set_title(
        f"Chronos-2 Forecast — {ep['episode_id']}\n"
        f"Anchor: {ep['anchor']}  |  RMSE: {ep['rmse']:.3f} mmol/L",
        fontsize=13,
        fontweight="bold",
    )
    ax1.set_xlim(-context_hours, FORECAST_HORIZON * INTERVAL_MINS / 60)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis="both", labelsize=9)

    # Combined legend
    handles1, labels1 = ax1.get_legend_handles_labels()
    if has_iob:
        handles2, labels2 = ax2.get_legend_handles_labels()
        handles1 += handles2
        labels1 += labels2
    ax1.legend(handles1, labels1, loc="upper right", fontsize=8, framealpha=0.9)

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    fname = f"chronos2_{pid}_{ep['episode_id'].replace('::', '_')}.png"
    out_path = os.path.join(output_dir, fname)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")
    return out_path


def plot_combined(
    results: dict[str, dict],
    output_dir: str,
    context_hours: float = 2.0,
) -> str:
    """Generate a 1x2 side-by-side comparison of both episodes.

    Returns path to saved PNG.
    """
    matplotlib.use("Agg")

    fig, axes = plt.subplots(2, 1, figsize=(6, 9), sharey=True)

    for ax_idx, (pid, ep) in enumerate(results.items()):
        ax1 = axes[ax_idx]

        ctx_bg = np.array(ep["context_bg"])
        target_bg = np.array(ep["target_bg"])
        pred_mean = np.array(ep["mean"])
        pred_q10 = np.array(ep["q10"])
        pred_q90 = np.array(ep["q90"])

        ctx_show = int(context_hours * 60 / INTERVAL_MINS)
        ctx_bg_show = ctx_bg[-ctx_show:]

        t_ctx = _time_axis(ctx_show, offset_hours=-context_hours)
        t_fh = _time_axis(len(pred_mean), offset_hours=0.0)
        t_target = _time_axis(len(target_bg), offset_hours=0.0)

        # Context BG
        ax1.plot(t_ctx, ctx_bg_show, "k-", linewidth=2.0, label="Actual BG (context)", zorder=10)

        # Target (ground truth)
        ax1.plot(t_target, target_bg, "k--", linewidth=2.0, label="Actual BG (target)", zorder=10)

        # Predicted mean
        ax1.plot(
            t_fh,
            pred_mean,
            color="steelblue",
            linewidth=2.0,
            label=f"Predicted (RMSE={ep['rmse']:.2f})",
            zorder=8,
        )

        # Quantile band
        ax1.fill_between(
            t_fh, pred_q10, pred_q90,
            color="steelblue", alpha=0.15,
            label="10th–90th pctl", zorder=3,
        )

        # Hypo threshold
        ax1.axhline(y=HYPO_THRESHOLD, color="red", linestyle=":", linewidth=1.2, alpha=0.6)

        # Midnight
        ax1.axvline(x=0, color="gray", linestyle="-", linewidth=0.8, alpha=0.4)

        # IOB secondary axis
        has_iob = ep.get("forecast_iob") is not None and ep.get("context_iob") is not None
        if has_iob:
            ax2 = ax1.twinx()
            ctx_iob = np.array(ep["context_iob"])[-ctx_show:]
            fh_iob = np.array(ep["forecast_iob"])
            ax2.plot(t_ctx, ctx_iob, color="gray", linestyle="--", linewidth=0.9, alpha=0.4)
            ax2.plot(t_fh[: len(fh_iob)], fh_iob, color="gray", linestyle="--", linewidth=0.9, alpha=0.4, label="IOB")
            ax2.set_ylabel("IOB (U)", fontsize=9, color="gray")
            ax2.tick_params(axis="y", labelcolor="gray", labelsize=7)
            max_iob = max(np.max(ctx_iob), np.max(fh_iob)) if len(fh_iob) > 0 else 1
            ax2.set_ylim(0, max(max_iob * 1.6, 0.5))

        ax1.set_xlabel("Hours relative to midnight", fontsize=10)
        ax1.set_ylabel("Blood Glucose (mmol/L)", fontsize=10)
        ax1.set_title(
            f"{ep['episode_id']}\nAnchor: {ep['anchor']}  |  RMSE: {ep['rmse']:.3f}",
            fontsize=11,
            fontweight="bold",
        )
        ax1.set_xlim(-context_hours, FORECAST_HORIZON * INTERVAL_MINS / 60)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis="both", labelsize=8)

        if ax_idx == 0:
            handles, labels = ax1.get_legend_handles_labels()
            ax1.legend(handles, labels, loc="upper right", fontsize=7, framealpha=0.9)

    fig.subplots_adjust(left=0.12, right=0.88, top=0.95, bottom=0.07, hspace=0.35)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "chronos2_combined_episodes.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved combined: {out_path}")
    return out_path


# ── CLI ─────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize Chronos-2 forecasts for selected Brown 2019 episodes."
    )
    parser.add_argument(
        "--predictor-path",
        default=DEFAULT_PREDICTOR_PATH,
        help="Path to the trained AutoGluon TimeSeriesPredictor directory.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save plots and JSON results.",
    )
    parser.add_argument(
        "--context-hours",
        type=float,
        default=2.0,
        help="Hours of context to show before midnight (default: 2.0).",
    )
    parser.add_argument(
        "--skip-predict",
        action="store_true",
        help="Skip prediction; load results from existing JSON instead.",
    )
    parser.add_argument(
        "--results-json",
        default=None,
        help="Path to precomputed results JSON (used with --skip-predict).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    json_path = args.results_json or os.path.join(
        args.output_dir, "chronos2_episode_results.json"
    )

    # ── Generate or load predictions ────────────────────────────────
    if args.skip_predict:
        print(f"Loading precomputed results from {json_path}")
        with open(json_path) as f:
            results = json.load(f)
    else:
        results = generate_predictions(args.predictor_path)

        # Save results
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved results JSON: {json_path}")

    # ── Plot both episodes in a single figure ─────────────────────
    print("\nGenerating combined figure …")
    plot_combined(results, args.output_dir, context_hours=args.context_hours)

    print("\nDone. All outputs in:", args.output_dir)


if __name__ == "__main__":
    main()
