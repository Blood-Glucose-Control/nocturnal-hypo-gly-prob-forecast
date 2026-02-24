#!/usr/bin/env python3
"""
TiDE Registry Visualization: Patient-Level vs Temporal Holdout

Generates best-30 episode plots for two holdout groups:
  1. Patient holdout: 8 patients the model has NEVER seen
  2. Temporal holdout: 160 training patients, but evaluated on held-out time periods

Each plot shows 30 episodes in a 6x5 grid:
  - Ground truth BG (black)
  - TiDE prediction (blue)
  - IOB trajectory (gray, secondary axis)
  - Midnight boundary (purple vertical line)
  - Hypo threshold at 3.9 mM (red dashed)

USAGE:
    python scripts/visualize_tide_registry.py --model-path models/tide_registry/best_1406592
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor  # noqa: E402
from src.data.versioning.dataset_registry import DatasetRegistry  # noqa: E402
from src.data.models import ColumnNames  # noqa: E402
from src.models.chronos2.utils import convert_to_patient_dict  # noqa: E402
from src.evaluation.episode_builders import build_midnight_episodes  # noqa: E402

# Configuration
INTERVAL_MINS = 5
CONTEXT_LENGTH = 512
FORECAST_HORIZON = 72
CONTEXT_HOURS_TO_SHOW = 3
TARGET_COL = ColumnNames.BG.value
IOB_COL = ColumnNames.IOB.value
PATIENT_COL = "p_num"
TIME_COL = "datetime"
NUM_DISPLAY = 30

# 8 patient-level holdout patients from configs/data/holdout_5pct/brown_2019.yaml
PATIENT_HOLDOUT_IDS = {
    "bro_164",
    "bro_62",
    "bro_152",
    "bro_35",
    "bro_88",
    "bro_31",
    "bro_134",
    "bro_3",
}


def format_episodes_for_prediction(episodes, forecast_horizon):
    """Format episodes for AutoGluon prediction."""
    train_data_list = []
    known_cov_list = []

    for i, ep in enumerate(episodes):
        item_id = f"ep_{i:04d}"

        df = ep["context_df"].copy()
        df["item_id"] = item_id
        df["timestamp"] = df.index
        df["target"] = df[TARGET_COL]
        df["iob"] = df[IOB_COL].ffill().fillna(0) if IOB_COL in df.columns else 0.0
        train_data_list.append(df[["item_id", "timestamp", "target", "iob"]])

        future_covs = ep.get("future_covariates", {})
        future_iob = future_covs.get(IOB_COL, np.zeros(forecast_horizon))
        future_iob = pd.Series(future_iob).ffill().fillna(0).to_numpy()
        future_timestamps = pd.date_range(
            ep["anchor"], periods=forecast_horizon, freq=f"{INTERVAL_MINS}min"
        )
        future_df = pd.DataFrame(
            {
                "item_id": item_id,
                "timestamp": future_timestamps,
                "iob": future_iob[:forecast_horizon],
            }
        )
        known_cov_list.append(future_df)

    train_combined = pd.concat(train_data_list, ignore_index=True)
    train_combined = train_combined.set_index(["item_id", "timestamp"])
    train_data = TimeSeriesDataFrame(train_combined)

    known_combined = pd.concat(known_cov_list, ignore_index=True)
    known_combined = known_combined.set_index(["item_id", "timestamp"])
    known_covariates = TimeSeriesDataFrame(known_combined)

    return train_data, known_covariates


def compute_episode_metrics(predictor, episodes, forecast_horizon):
    """Run predictions and compute per-episode RMSE + discontinuity."""
    if not episodes:
        return []

    ts_data, known_cov = format_episodes_for_prediction(episodes, forecast_horizon)
    predictions = predictor.predict(ts_data, known_covariates=known_cov)

    metrics = []
    for i, ep in enumerate(episodes):
        item_id = f"ep_{i:04d}"
        if item_id not in predictions.index.get_level_values(0):
            continue

        pred = predictions.loc[item_id]["mean"].values
        actual = ep["target_bg"][: len(pred)]

        rmse = np.sqrt(np.mean((pred - actual) ** 2))
        last_context = ep["context_df"][TARGET_COL].iloc[-1]
        discontinuity = abs(last_context - pred[0])

        future_covs = ep.get("future_covariates", {})
        future_iob = future_covs.get(IOB_COL, np.zeros(forecast_horizon))
        mean_iob = np.mean(future_iob)

        metrics.append(
            {
                "episode": ep,
                "pred": pred,
                "rmse": rmse,
                "discontinuity": discontinuity,
                "mean_iob": mean_iob,
                "last_context_bg": last_context,
                "patient_id": ep.get("patient_id", "?"),
            }
        )

    return metrics


def plot_30_episodes(selected, title, filename, output_dir):
    """Plot 30 episodes in a 6x5 grid."""
    n = min(len(selected), NUM_DISPLAY)
    fig, axes = plt.subplots(6, 5, figsize=(25, 30))
    axes_flat = axes.flatten()

    context_steps = int(CONTEXT_HOURS_TO_SHOW * 60 / INTERVAL_MINS)
    forecast_time_hours = np.arange(FORECAST_HORIZON) * INTERVAL_MINS / 60
    context_time_hours = -np.arange(context_steps, 0, -1) * INTERVAL_MINS / 60

    avg_rmse = np.mean([e["rmse"] for e in selected[:n]])
    avg_discont = np.mean([e["discontinuity"] for e in selected[:n]])

    for plot_idx in range(30):
        ax = axes_flat[plot_idx]
        ax2 = ax.twinx()

        if plot_idx >= n:
            ax.set_visible(False)
            ax2.set_visible(False)
            continue

        em = selected[plot_idx]
        ep = em["episode"]

        # Context BG (last N hours)
        context_bg = ep["context_df"][TARGET_COL].iloc[-context_steps:].to_numpy()
        forecast_bg = ep["target_bg"]

        # Context (dashed)
        ax.plot(
            context_time_hours,
            context_bg,
            "k--",
            linewidth=1.5,
            alpha=0.6,
            label="Context BG",
            zorder=9,
        )

        # Forecast ground truth (solid black)
        ax.plot(
            forecast_time_hours,
            forecast_bg,
            "k-",
            linewidth=2.5,
            label="Actual BG",
            zorder=10,
        )

        # Connect context to forecast
        boundary_time = np.array([context_time_hours[-1], forecast_time_hours[0]])
        boundary_bg = np.array([context_bg[-1], forecast_bg[0]])
        ax.plot(boundary_time, boundary_bg, "k-", linewidth=2.0, alpha=0.8, zorder=9.5)

        # TiDE prediction
        pred = em["pred"]
        ax.plot(
            forecast_time_hours[: len(pred)],
            pred,
            color="dodgerblue",
            linewidth=2.0,
            alpha=0.9,
            label="TiDE Prediction",
            zorder=5,
        )

        # IOB trajectory
        future_covs = ep.get("future_covariates", {})
        future_iob = future_covs.get(IOB_COL)
        if future_iob is not None and IOB_COL in ep["context_df"].columns:
            context_iob = (
                ep["context_df"][IOB_COL].iloc[-context_steps:].fillna(0).to_numpy()
            )
            ax2.plot(
                context_time_hours,
                context_iob,
                color="gray",
                linestyle=":",
                linewidth=1.0,
                alpha=0.4,
            )
            ax2.plot(
                forecast_time_hours[: len(future_iob)],
                future_iob,
                color="gray",
                linestyle="--",
                linewidth=1.2,
                alpha=0.5,
                label="IOB",
            )
            ax2.set_ylabel("IOB (U)", fontsize=7, color="gray")
            ax2.tick_params(axis="y", labelcolor="gray", labelsize=6)
            all_iob = np.concatenate([context_iob, future_iob])
            max_iob = max(all_iob) if len(all_iob) > 0 else 0
            ax2.set_ylim(0, max_iob * 1.5 if max_iob > 0 else 1)

        # Midnight boundary
        ax.axvline(
            x=0, color="purple", linestyle="-", alpha=0.5, linewidth=2, label="Midnight"
        )

        # Hypo threshold
        ax.axhline(y=3.9, color="red", linestyle=":", alpha=0.3, linewidth=1)

        ax.set_xlabel("Hours from Midnight", fontsize=7)
        ax.set_ylabel("BG (mmol/L)", fontsize=7)
        ax.set_title(
            f'#{plot_idx+1} {em["patient_id"]} ({ep["anchor"].strftime("%Y-%m-%d")}) '
            f"IOB={em['mean_iob']:.1f}U "
            f"RMSE={em['rmse']:.2f} "
            f"\u0394={em['discontinuity']:.2f}",
            fontsize=7,
        )
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-CONTEXT_HOURS_TO_SHOW, 6)
        ax.tick_params(axis="both", labelsize=6)

        if plot_idx == 0:
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=6)

    plt.suptitle(
        f"{title}\n"
        f"Avg RMSE={avg_rmse:.3f} mM | Avg Discontinuity={avg_discont:.3f} mM | "
        f"N={n} episodes",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()

    output_path = output_dir / filename
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")

    return avg_rmse, avg_discont


def main():
    parser = argparse.ArgumentParser(description="Visualize TiDE registry results")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained predictor (e.g., models/tide_registry/best_1406592)",
    )
    parser.add_argument("--holdout-dir", type=str, default="configs/data/holdout_5pct")
    parser.add_argument("--dataset", type=str, default="brown_2019")
    parser.add_argument("--max-episodes", type=int, default=500)
    args = parser.parse_args()

    model_path = PROJECT_ROOT / args.model_path
    output_dir = model_path / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("TiDE REGISTRY VISUALIZATION")
    print("Patient-Level vs Temporal Holdout")
    print("=" * 70)

    # =========================================================================
    # Load model
    # =========================================================================
    print(f"\nLoading model from: {model_path}")
    predictor = TimeSeriesPredictor.load(str(model_path))

    # =========================================================================
    # Load holdout data
    # =========================================================================
    print(f"\nLoading holdout data via registry ({args.holdout_dir})...")
    registry = DatasetRegistry(holdout_config_dir=args.holdout_dir)
    _, holdout_df = registry.load_dataset_with_split(args.dataset)

    holdout_patient_dict = convert_to_patient_dict(holdout_df, PATIENT_COL, TIME_COL)
    print(f"Total holdout patients: {len(holdout_patient_dict)}")

    # Split into patient-level vs temporal holdout
    patient_holdout = {
        pid: pdf
        for pid, pdf in holdout_patient_dict.items()
        if pid in PATIENT_HOLDOUT_IDS
    }
    temporal_holdout = {
        pid: pdf
        for pid, pdf in holdout_patient_dict.items()
        if pid not in PATIENT_HOLDOUT_IDS
    }

    print(
        f"  Patient-level holdout: {len(patient_holdout)} patients "
        f"(never seen: {sorted(patient_holdout.keys())})"
    )
    print(
        f"  Temporal holdout: {len(temporal_holdout)} patients "
        f"(seen during training, eval on held-out time)"
    )

    # =========================================================================
    # Build episodes for both groups
    # =========================================================================
    def build_episodes_for_group(patient_dict, group_name, max_eps):
        episodes = []
        for pid, pdf in patient_dict.items():
            eps, skip_stats = build_midnight_episodes(
                pdf,
                context_length=CONTEXT_LENGTH,
                forecast_length=FORECAST_HORIZON,
                target_col=TARGET_COL,
                covariate_cols=[IOB_COL],
                interval_mins=INTERVAL_MINS,
            )
            # Tag each episode with its patient ID
            for ep in eps:
                ep["patient_id"] = pid
            episodes.extend(eps)
            if len(episodes) >= max_eps:
                break
        episodes = episodes[:max_eps]
        print(f"  {group_name}: {len(episodes)} episodes")
        return episodes

    print("\nBuilding midnight-anchored episodes...")
    patient_episodes = build_episodes_for_group(
        patient_holdout, "Patient holdout", args.max_episodes
    )
    temporal_episodes = build_episodes_for_group(
        temporal_holdout, "Temporal holdout", args.max_episodes
    )

    # =========================================================================
    # Predict + compute metrics for both groups
    # =========================================================================
    print("\nRunning predictions...")

    print("  Patient holdout predictions...")
    patient_metrics = compute_episode_metrics(
        predictor, patient_episodes, FORECAST_HORIZON
    )
    print(f"    {len(patient_metrics)} episodes evaluated")

    print("  Temporal holdout predictions...")
    temporal_metrics = compute_episode_metrics(
        predictor, temporal_episodes, FORECAST_HORIZON
    )
    print(f"    {len(temporal_metrics)} episodes evaluated")

    # =========================================================================
    # Generate plots: Best 30 for each group
    # =========================================================================
    print(f"\n{'='*70}")
    print("GENERATING PLOTS")
    print("=" * 70)

    results = {}

    for group_name, metrics, label in [
        (
            "patient",
            patient_metrics,
            "Patient-Level Holdout (NEVER seen during training)",
        ),
        (
            "temporal",
            temporal_metrics,
            "Temporal Holdout (seen patients, held-out time)",
        ),
    ]:
        if not metrics:
            print(f"\n  WARNING: No episodes for {group_name} holdout, skipping.")
            continue

        # Sort by RMSE ascending (best first)
        sorted_best = sorted(metrics, key=lambda x: x["rmse"])
        sorted_worst = sorted(metrics, key=lambda x: x["rmse"], reverse=True)

        print(f"\n--- {label} ---")

        # Best 30
        avg_rmse_best, avg_disc_best = plot_30_episodes(
            sorted_best[:NUM_DISPLAY],
            f"Best 30 Episodes: {label}",
            f"{group_name}_best30.png",
            output_dir,
        )
        results[f"{group_name}_best30"] = {
            "rmse": avg_rmse_best,
            "discont": avg_disc_best,
        }

        # Worst 30
        avg_rmse_worst, avg_disc_worst = plot_30_episodes(
            sorted_worst[:NUM_DISPLAY],
            f"Worst 30 Episodes: {label}",
            f"{group_name}_worst30.png",
            output_dir,
        )
        results[f"{group_name}_worst30"] = {
            "rmse": avg_rmse_worst,
            "discont": avg_disc_worst,
        }

    # =========================================================================
    # Summary
    # =========================================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print("=" * 70)

    for group_name, metrics, label in [
        ("patient", patient_metrics, "Patient holdout"),
        ("temporal", temporal_metrics, "Temporal holdout"),
    ]:
        if not metrics:
            continue
        all_rmse = np.mean([m["rmse"] for m in metrics])
        all_disc = np.mean([m["discontinuity"] for m in metrics])
        all_var = np.mean([np.var(m["pred"]) for m in metrics])
        print(f"\n  {label} ({len(metrics)} episodes):")
        print(f"    RMSE:          {all_rmse:.4f} mM")
        print(
            f"    Discontinuity: {all_disc:.4f} mM "
            f"{'PASS' if all_disc < 0.2 else 'FAIL'}"
        )
        print(f"    Variance:      {all_var:.4f}")

        if f"{group_name}_best30" in results:
            r = results[f"{group_name}_best30"]
            print(f"    Best 30:  RMSE={r['rmse']:.3f}, Disc={r['discont']:.3f}")
        if f"{group_name}_worst30" in results:
            r = results[f"{group_name}_worst30"]
            print(f"    Worst 30: RMSE={r['rmse']:.3f}, Disc={r['discont']:.3f}")

    print(f"\nPlots saved to: {output_dir}")
    print("  patient_best30.png  — Best 30 on never-seen patients")
    print("  patient_worst30.png — Worst 30 on never-seen patients")
    print("  temporal_best30.png — Best 30 on held-out time periods")
    print("  temporal_worst30.png — Worst 30 on held-out time periods")


if __name__ == "__main__":
    main()
