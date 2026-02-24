#!/usr/bin/env python3
"""
TiDE Uncertainty Visualization: Probabilistic Forecasts with Confidence Intervals

Generates SEPARATE plots for Manual vs HPO models with uncertainty bands:
  - Manual model: mean + 10-90% confidence interval
  - HPO model: mean + 10-90% confidence interval

This allows investigating:
  1. Does uncertainty increase at the boundary (midnight)?
  2. Do quantiles also show discontinuity?
  3. Is the HPO model more/less certain than manual?

Visualizes:
  - Ground truth BG (black, bold)
  - Model mean prediction (colored line)
  - 10-90% confidence interval (shaded region)
  - IOB trajectory (gray, secondary axis)
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor  # noqa: E402
from src.data.diabetes_datasets.data_loader import get_loader  # noqa: E402
from src.data.models import ColumnNames  # noqa: E402

# ============================================================
# Configuration
# ============================================================
INTERVAL_MINS = 5
CONTEXT_LENGTH_DEFAULT = 144
CONTEXT_LENGTH_SCALED = 512
FORECAST_HORIZON = 72
CONTEXT_HOURS_TO_SHOW = 3  # Show 3 hours of context before forecast
TARGET_COL = ColumnNames.BG.value
IOB_COL = ColumnNames.IOB.value
NUM_DISPLAY = 30
MIN_IOB_THRESHOLD = 0.5

# Model registry
MODELS = {
    "tide_manual": {
        "label": "TiDE Manual (512c, 256d)",
        "color": "blue",
        "linestyle": "-",
        "linewidth": 2.0,
        "alpha": 0.8,
        "context_length": CONTEXT_LENGTH_SCALED,
        "path": "models/tide_validation/scaled",
        "zorder": 2,
    },
    "tide_hpo": {
        "label": "TiDE HPO (Random Search)",
        "color": "darkorange",
        "linestyle": "-",
        "linewidth": 2.0,
        "alpha": 0.9,
        "context_length": CONTEXT_LENGTH_SCALED,
        "path": "models/tide_hpo",  # First HPO run (non-Bayesian)
        "zorder": 3,
    },
}

# Which sets of 30 to generate
SELECTIONS = [
    {
        "name": "best30",
        "title_prefix": "Top 30 Episodes (Best Manual TiDE RMSE)",
        "sort_key": "rmse_tide_manual",
        "ascending": True,
        "filename": "tide_uncertainty_best30.png",  # Will append _manual.png and _hpo.png
    },
]


def build_midnight_episodes(
    patient_df, target_col, iob_col, interval_mins, context_len, horizon
):
    """Build midnight-anchored episodes with IOB data.

    Note: Episodes with any NaN values are filtered out, so only complete
    continuous segments are used for evaluation.
    """
    df = patient_df.sort_index()
    df = df[~df.index.duplicated(keep="last")]

    has_iob = iob_col in df.columns and df[iob_col].notna().any()

    freq = f"{interval_mins}min"
    grid = pd.date_range(
        df.index.min().floor(freq), df.index.max().floor(freq), freq=freq
    )
    df = df.reindex(grid)

    dt = pd.Timedelta(minutes=interval_mins)
    earliest = df.index.min() + context_len * dt
    latest = df.index.max() - (horizon - 1) * dt

    first_midnight = earliest.normalize()
    if first_midnight < earliest:
        first_midnight += pd.Timedelta(days=1)

    last_midnight = latest.normalize()
    if last_midnight < first_midnight:
        return []

    episodes = []
    for anchor in pd.date_range(first_midnight, last_midnight, freq="D"):
        window_start = anchor - context_len * dt
        window_end = anchor + horizon * dt
        window_index = pd.date_range(
            window_start, window_end, freq=freq, inclusive="left"
        )

        cols_to_get = [target_col]
        if has_iob:
            cols_to_get.append(iob_col)

        window_df = df.reindex(window_index)[cols_to_get]

        if window_df[target_col].isna().any():
            continue

        context_df = window_df.iloc[:context_len].copy()
        forecast_df = window_df.iloc[context_len:].copy()

        target_bg = forecast_df[target_col].to_numpy()

        future_iob = None
        if has_iob:
            if context_df[iob_col].isna().mean() > 0.5:
                continue
            context_df[iob_col] = context_df[iob_col].ffill().fillna(0)
            future_iob = forecast_df[iob_col].ffill().fillna(0).to_numpy()

        episodes.append(
            {
                "anchor": anchor,
                "context_df": context_df,
                "target_bg": target_bg,
                "future_iob": future_iob,
            }
        )

    return episodes


def format_for_autogluon(episodes, target_col, iob_col, context_len):
    """Format episodes for AutoGluon prediction with IOB."""
    train_data_list = []
    known_cov_list = []

    for i, ep in enumerate(episodes):
        item_id = f"ep_{i:03d}"

        # Adjust context to match model's context length
        context_df = ep["context_df"].copy()
        if len(context_df) > context_len:
            context_df = context_df.iloc[-context_len:]

        df = context_df.copy()
        df["item_id"] = item_id
        df["timestamp"] = df.index
        df["target"] = df[target_col]
        df["iob"] = df[iob_col]

        train_data_list.append(df[["item_id", "timestamp", "target", "iob"]])

        future_timestamps = pd.date_range(
            ep["anchor"], periods=FORECAST_HORIZON, freq=f"{INTERVAL_MINS}min"
        )
        future_df = pd.DataFrame(
            {
                "item_id": item_id,
                "timestamp": future_timestamps,
                "iob": ep["future_iob"],
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


def plot_single_model_with_uncertainty(
    selected_episodes, model_key, selection_config, output_dir
):
    """Plot 30 episodes for a SINGLE model with uncertainty bands (10-90% CI)."""
    fig, axes = plt.subplots(6, 5, figsize=(25, 30))
    axes_flat = axes.flatten()

    cfg = MODELS[model_key]

    # Time arrays for forecast and context
    context_steps = int(CONTEXT_HOURS_TO_SHOW * 60 / INTERVAL_MINS)
    forecast_time_hours = np.arange(FORECAST_HORIZON) * INTERVAL_MINS / 60
    context_time_hours = -np.arange(context_steps, 0, -1) * INTERVAL_MINS / 60

    # Compute aggregate metrics
    avg_rmse = np.mean([e[f"rmse_{model_key}"] for e in selected_episodes])
    avg_discont = np.mean([e[f"discont_{model_key}"] for e in selected_episodes])
    avg_boundary_unc = np.mean(
        [e[f"boundary_unc_{model_key}"] for e in selected_episodes]
    )

    for plot_idx, em in enumerate(selected_episodes):
        ax = axes_flat[plot_idx]
        ax2 = ax.twinx()

        ep = em["episode"]

        # Extract context BG (last N hours before midnight)
        context_bg = ep["context_df"][TARGET_COL].iloc[-context_steps:].to_numpy()
        forecast_bg = ep["target_bg"]

        # Plot as single continuous line, but style differently before/after midnight
        # Context portion (dashed)
        ax.plot(
            context_time_hours,
            context_bg,
            "k--",
            linewidth=1.5,
            alpha=0.6,
            label="Context BG",
            zorder=9,
        )

        # Forecast portion (solid)
        ax.plot(
            forecast_time_hours,
            forecast_bg,
            "k-",
            linewidth=2.5,
            label="Actual BG",
            zorder=10,
        )

        # Add connecting segment at boundary to ensure continuity
        boundary_time = np.array([context_time_hours[-1], forecast_time_hours[0]])
        boundary_bg = np.array([context_bg[-1], forecast_bg[0]])
        ax.plot(boundary_time, boundary_bg, "k-", linewidth=2.0, alpha=0.8, zorder=9.5)

        # IOB trajectory on secondary axis (both context and forecast)
        if ep["future_iob"] is not None:
            # Context IOB
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
            # Forecast IOB
            ax2.plot(
                forecast_time_hours,
                ep["future_iob"],
                color="gray",
                linestyle="--",
                linewidth=1.2,
                alpha=0.5,
                label="IOB",
            )
            ax2.set_ylabel("IOB (U)", fontsize=7, color="gray")
            ax2.tick_params(axis="y", labelcolor="gray", labelsize=6)
            all_iob = np.concatenate([context_iob, ep["future_iob"]])
            max_iob = max(all_iob)
            ax2.set_ylim(0, max_iob * 1.5 if max_iob > 0 else 1)

        # Mark the boundary (midnight) - this is where forecast starts
        ax.axvline(
            x=0, color="purple", linestyle="-", alpha=0.5, linewidth=2, label="Midnight"
        )

        # Plot model prediction with uncertainty bands
        pred_mean = em[f"pred_{model_key}"]
        pred_10 = em[f"pred_10_{model_key}"]
        pred_90 = em[f"pred_90_{model_key}"]

        # Uncertainty band (10-90% CI)
        ax.fill_between(
            forecast_time_hours[: len(pred_mean)],
            pred_10,
            pred_90,
            color=cfg["color"],
            alpha=0.2,
            label="10-90% CI",
            zorder=1,
        )

        # Mean prediction
        ax.plot(
            forecast_time_hours[: len(pred_mean)],
            pred_mean,
            color=cfg["color"],
            linestyle=cfg["linestyle"],
            linewidth=cfg["linewidth"],
            alpha=cfg["alpha"],
            label=f"{cfg['label']} (mean)",
            zorder=cfg["zorder"],
        )

        # Hypo threshold (across both context and forecast)
        ax.axhline(y=3.9, color="red", linestyle=":", alpha=0.3, linewidth=1)

        ax.set_xlabel("Hours from Midnight", fontsize=7)
        ax.set_ylabel("BG (mmol/L)", fontsize=7)

        # Title: show RMSE, discontinuity, and boundary uncertainty
        rmse = em[f"rmse_{model_key}"]
        discont = em[f"discont_{model_key}"]
        boundary_unc = em[f"boundary_unc_{model_key}"]
        ax.set_title(
            f'#{plot_idx+1} ({ep["anchor"].strftime("%Y-%m-%d")}) '
            f"IOB={em['mean_iob']:.1f}U "
            f"RMSE={rmse:.2f} "
            f"Δ={discont:.2f} "
            f"Unc={boundary_unc:.2f}",
            fontsize=7,
        )
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-CONTEXT_HOURS_TO_SHOW, 6)  # Show context before midnight
        ax.tick_params(axis="both", labelsize=6)

        # Legend only on first subplot
        if plot_idx == 0:
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=6)

    # Build subtitle with metrics
    subtitle = (
        f"{cfg['label']}: RMSE={avg_rmse:.3f}, "
        f"Discontinuity={avg_discont:.3f}, "
        f"Avg Boundary Uncertainty={avg_boundary_unc:.3f}"
    )

    plt.suptitle(
        f'{selection_config["title_prefix"]} - {cfg["label"]}\n{subtitle}',
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )

    plt.tight_layout()

    # Generate filename with model name
    base_filename = selection_config["filename"].replace(".png", "")
    model_suffix = model_key.replace("tide_", "")
    output_path = output_dir / f"{base_filename}_{model_suffix}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")

    return avg_rmse, avg_discont, avg_boundary_unc


def main():
    print("=" * 70)
    print("TiDE HPO COMPARISON VISUALIZATION")
    print("Comparing Manual Scaled vs First HPO (Random Search)")
    print("=" * 70)

    output_dir = PROJECT_ROOT / "models/tide_hpo"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # Verify model paths
    # ============================================================
    print("\nModel paths:")
    model_keys = list(MODELS.keys())
    all_exist = True
    for mk in model_keys:
        cfg = MODELS[mk]
        full_path = PROJECT_ROOT / cfg["path"]
        exists = full_path.exists()
        print(f"  {cfg['label']:30s} -> {cfg['path']} {'OK' if exists else 'MISSING'}")
        if not exists:
            all_exist = False

    if not all_exist:
        print("\nERROR: Some models are missing. Aborting.")
        return

    # ============================================================
    # Load validation data + build episodes (use max context length)
    # ============================================================
    print("\n" + "=" * 70)
    print("LOADING VALIDATION DATA")
    print("=" * 70)

    loader = get_loader(
        data_source_name="brown_2019",
        dataset_type="train",
        use_cached=True,
    )
    val_data = loader.validation_data
    print(f"Validation patients: {len(val_data)}")

    print(
        f"\nBuilding midnight-anchored episodes (using {CONTEXT_LENGTH_SCALED} context)..."
    )
    all_episodes = []
    for pid, pdf in val_data.items():
        eps = build_midnight_episodes(
            pdf,
            TARGET_COL,
            IOB_COL,
            INTERVAL_MINS,
            CONTEXT_LENGTH_SCALED,
            FORECAST_HORIZON,
        )
        all_episodes.extend(eps)
    print(f"Total episodes: {len(all_episodes)}")

    # ============================================================
    # Load all models
    # ============================================================
    print("\n" + "=" * 70)
    print("LOADING MODELS")
    print("=" * 70)

    predictors = {}
    for mk in model_keys:
        cfg = MODELS[mk]
        print(f"Loading {cfg['label']}...")
        predictors[mk] = TimeSeriesPredictor.load(str(PROJECT_ROOT / cfg["path"]))

    # ============================================================
    # Make predictions with both models
    # ============================================================
    print("\n" + "=" * 70)
    print("MAKING PREDICTIONS")
    print("=" * 70)

    predictions = {}
    for mk in model_keys:
        cfg = MODELS[mk]
        print(f"  {cfg['label']}...")

        # Format data with correct context length for this model
        ts_val, known_cov = format_for_autogluon(
            all_episodes, TARGET_COL, IOB_COL, cfg["context_length"]
        )

        predictions[mk] = predictors[mk].predict(ts_val, known_covariates=known_cov)

    # ============================================================
    # Compute per-episode metrics with discontinuity measurement
    # ============================================================
    print("\n" + "=" * 70)
    print("COMPUTING PER-EPISODE METRICS")
    print("=" * 70)

    episode_metrics = []
    for i, ep in enumerate(all_episodes):
        item_id = f"ep_{i:03d}"

        # Check all predictions exist
        skip = False
        for mk in model_keys:
            if item_id not in predictions[mk].index.get_level_values(0):
                skip = True
                break
        if skip:
            continue

        actual = ep["target_bg"]
        mean_iob = np.mean(ep["future_iob"]) if ep["future_iob"] is not None else 0

        # Get last context BG for discontinuity measurement
        last_context_bg = ep["context_df"][TARGET_COL].iloc[-1]

        entry = {
            "idx": i,
            "episode": ep,
            "item_id": item_id,
            "mean_iob": mean_iob,
            "last_context_bg": last_context_bg,
        }

        for mk in model_keys:
            pred_mean = predictions[mk].loc[item_id]["mean"].values
            pred_10 = predictions[mk].loc[item_id]["0.1"].values  # 10th percentile
            pred_90 = predictions[mk].loc[item_id]["0.9"].values  # 90th percentile

            rmse = np.sqrt(np.mean((pred_mean[: len(actual)] - actual) ** 2))

            # Discontinuity = |last_context_bg - first_forecast|
            discontinuity = abs(last_context_bg - pred_mean[0])

            # Uncertainty width at boundary
            boundary_uncertainty = pred_90[0] - pred_10[0]

            entry[f"pred_{mk}"] = pred_mean
            entry[f"pred_10_{mk}"] = pred_10
            entry[f"pred_90_{mk}"] = pred_90
            entry[f"rmse_{mk}"] = rmse
            entry[f"discont_{mk}"] = discontinuity
            entry[f"boundary_unc_{mk}"] = boundary_uncertainty

        episode_metrics.append(entry)

    print(f"Episodes with all predictions: {len(episode_metrics)}")

    # Filter: meaningful IOB
    filtered = [e for e in episode_metrics if e["mean_iob"] >= MIN_IOB_THRESHOLD]
    print(f"Episodes with mean IOB >= {MIN_IOB_THRESHOLD}: {len(filtered)}")

    # ============================================================
    # Generate plots
    # ============================================================
    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)

    all_results = {}

    for sel in SELECTIONS:
        print(f"\n--- {sel['title_prefix']} ---")

        sorted_eps = sorted(
            filtered, key=lambda x: x[sel["sort_key"]], reverse=not sel["ascending"]
        )
        selected = sorted_eps[:NUM_DISPLAY]

        if len(selected) < NUM_DISPLAY:
            print(
                f"  WARNING: Only {len(selected)} episodes available (wanted {NUM_DISPLAY})"
            )

        # Generate SEPARATE plots for each model with uncertainty bands
        results = {}
        for mk in model_keys:
            print(f"  Plotting {MODELS[mk]['label']}...")
            rmse, discont, boundary_unc = plot_single_model_with_uncertainty(
                selected, mk, sel, output_dir
            )
            results[mk] = {
                "rmse": rmse,
                "discont": discont,
                "boundary_unc": boundary_unc,
            }
        all_results[sel["name"]] = results

    # ============================================================
    # Summary table
    # ============================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Full validation set metrics
    print(f"\n{'--- Full validation set metrics ---':^80s}")
    print(f"{'Model':<30s}  {'RMSE':>10s}  {'Discontinuity':>15s}  {'Variance':>10s}")
    print("-" * 70)

    for mk in model_keys:
        all_rmse = np.mean([e[f"rmse_{mk}"] for e in episode_metrics])
        all_discont = np.mean([e[f"discont_{mk}"] for e in episode_metrics])

        # Compute variance across all predictions
        all_preds = np.concatenate([e[f"pred_{mk}"] for e in episode_metrics])
        variance = np.var(all_preds)

        print(
            f"{MODELS[mk]['label']:<30s}  {all_rmse:>10.4f}  {all_discont:>15.4f}  {variance:>10.4f}"
        )

    # Selected subset metrics
    for sel in SELECTIONS:
        print(f"\n{'--- ' + sel['title_prefix'] + ' ---':^80s}")
        print(
            f"{'Model':<30s}  {'RMSE':>10s}  {'Discontinuity':>15s}  {'Boundary Unc':>15s}"
        )
        print("-" * 80)

        for mk in model_keys:
            rmse = all_results[sel["name"]][mk]["rmse"]
            discont = all_results[sel["name"]][mk]["discont"]
            boundary_unc = all_results[sel["name"]][mk]["boundary_unc"]
            print(
                f"{MODELS[mk]['label']:<30s}  {rmse:>10.4f}  {discont:>15.4f}  {boundary_unc:>15.4f}"
            )

    print(f"\nOutput directory: {output_dir}")
    print("Files generated (2 per selection - one for each model):")
    for sel in SELECTIONS:
        for mk in model_keys:
            model_suffix = mk.replace("tide_", "")
            filename = sel["filename"].replace(".png", f"_{model_suffix}.png")
            print(f"  - {filename}")

    print("\n" + "=" * 70)
    print("DISCONTINUITY VALIDATION")
    print("=" * 70)

    for mk in model_keys:
        all_discont = np.mean([e[f"discont_{mk}"] for e in episode_metrics])
        status = "✅ PASS" if all_discont < 0.2 else "❌ FAIL"
        print(f"{MODELS[mk]['label']:<30s}: {all_discont:.4f} mM {status}")


if __name__ == "__main__":
    main()
