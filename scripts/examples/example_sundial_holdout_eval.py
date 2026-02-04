#!/usr/bin/env python3
"""
Sundial Zero-Shot Holdout Evaluation.

Minimal script to evaluate Sundial on holdout data for demo results.

Usage:
    python scripts/examples/example_sundial_holdout_eval.py --dataset kaggle_brisT1D
    python scripts/examples/example_sundial_holdout_eval.py --dataset kaggle_brisT1D --config-dir configs/data/holdout
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data.versioning.dataset_registry import DatasetRegistry
from src.data.preprocessing.time_processing import iter_daily_context_forecast_splits
from src.models.sundial import SundialForecaster, SundialConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Sundial zero-shot holdout evaluation")
    parser.add_argument("--dataset", type=str, default="kaggle_brisT1D", help="Dataset name")
    parser.add_argument("--config-dir", type=str, default="configs/data/holdout", help="Holdout config dir")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--num-samples", type=int, default=100, help="Sundial num_samples")
    parser.add_argument("--forecast-hours", type=int, default=6, help="Hours to forecast")
    args = parser.parse_args()

    # Setup
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
        args.output_dir = f"./trained_models/artifacts/sundial_eval/{timestamp}_{args.dataset}"

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    prediction_length = args.forecast_hours * 12  # 5-min intervals

    logger.info("=" * 60)
    logger.info("SUNDIAL ZERO-SHOT HOLDOUT EVALUATION")
    logger.info("=" * 60)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Forecast: {args.forecast_hours} hours ({prediction_length} steps)")
    logger.info(f"Output: {args.output_dir}")

    # Load holdout data
    logger.info("\n--- Loading Holdout Data ---")
    registry = DatasetRegistry(holdout_config_dir=args.config_dir)
    holdout_data = registry.load_holdout_data_only(args.dataset)

    patient_col = "p_num" if "p_num" in holdout_data.columns else "id"
    patients = holdout_data[patient_col].unique()
    logger.info(f"Holdout patients: {list(patients)}")
    logger.info(f"Total samples: {len(holdout_data):,}")

    # Initialize model
    logger.info("\n--- Initializing Sundial ---")
    config = SundialConfig(num_samples=args.num_samples)
    model = SundialForecaster(config)

    # Evaluate each patient
    logger.info("\n--- Running Evaluation ---")
    all_results = []
    plot_examples = []  # Store some examples for plotting

    for patient_id in patients:
        patient_df = holdout_data[holdout_data[patient_col] == patient_id].copy()
        if "datetime" in patient_df.columns:
            patient_df = patient_df.set_index("datetime")

        patient_preds = []
        patient_targets = []
        patient_contexts = []

        for daytime, nocturnal in iter_daily_context_forecast_splits(patient_df):
            context = daytime["bg_mM"].values
            target = nocturnal["bg_mM"].values[:prediction_length]

            # Skip invalid windows
            if np.isnan(context).any() or np.isnan(target).any():
                continue
            if len(context) < 10 or len(target) < prediction_length:
                continue

            # Predict
            pred = model.predict(context, batch_size=prediction_length)
            patient_preds.append(pred)
            patient_targets.append(target)
            patient_contexts.append(context)

            # Store first valid example per patient for plotting
            if len(plot_examples) < 8 and len(patient_preds) == 1:
                # Get IOB if available
                iob_context = daytime["iob"].values if "iob" in daytime.columns else None
                iob_target = nocturnal["iob"].values[:prediction_length] if "iob" in nocturnal.columns else None

                plot_examples.append({
                    "patient": patient_id,
                    "context": context,
                    "target": target,
                    "pred": pred,
                    "iob_context": iob_context,
                    "iob_target": iob_target,
                })

        if not patient_preds:
            logger.warning(f"  {patient_id}: No valid windows")
            continue

        # Aggregate patient metrics
        preds = np.concatenate(patient_preds)
        targets = np.concatenate(patient_targets)
        metrics = model._compute_metrics(preds, targets)

        all_results.append({
            "patient": patient_id,
            "days": len(patient_preds),
            **metrics,
        })

        logger.info(f"  {patient_id}: RMSE={metrics['rmse']:.3f}, MAE={metrics['mae']:.3f} ({len(patient_preds)} days)")

    # Overall metrics (aggregate from per-patient, no duplicate inference)
    overall = {}
    if all_results:
        total_days = sum(r['days'] for r in all_results)
        overall_rmse = np.sqrt(sum(r['rmse']**2 * r['days'] for r in all_results) / total_days)
        overall_mae = sum(r['mae'] * r['days'] for r in all_results) / total_days
        overall_mape = sum(r['mape'] * r['days'] for r in all_results) / total_days
        overall = {"rmse": overall_rmse, "mae": overall_mae, "mape": overall_mape, "mse": overall_rmse**2}

        logger.info("\n" + "=" * 60)
        logger.info("OVERALL RESULTS")
        logger.info("=" * 60)
        logger.info(f"RMSE: {overall['rmse']:.3f}")
        logger.info(f"MAE:  {overall['mae']:.3f}")
        logger.info(f"MAPE: {overall['mape']:.1f}%")
        logger.info(f"Total days: {sum(r['days'] for r in all_results)}")

    # Save results
    results = {
        "model": "sundial-base-128m",
        "dataset": args.dataset,
        "timestamp": datetime.now().isoformat(),
        "config": {"num_samples": args.num_samples, "forecast_hours": args.forecast_hours},
        "overall": overall,
        "per_patient": all_results,
    }

    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to: {results_file}")

    # Save CSV
    if all_results:
        pd.DataFrame(all_results).to_csv(output_path / "patient_metrics.csv", index=False)

    # Generate plots
    if plot_examples:
        logger.info("\n--- Generating Plots ---")
        n = len(plot_examples)
        ncols = min(4, n)
        nrows = int(np.ceil(n / ncols))

        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        axes = np.array(axes).flatten() if n > 1 else [axes]

        # Show last 3 hours of context + 6 hour forecast
        context_hours_to_show = 3
        context_steps_to_show = context_hours_to_show * 12  # 5-min intervals

        for i, ex in enumerate(plot_examples):
            ax = axes[i]
            ctx_full, tgt, pred = ex["context"], ex["target"], ex["pred"]
            iob_ctx_full, iob_tgt = ex.get("iob_context"), ex.get("iob_target")

            # Truncate context to last N hours
            ctx = ctx_full[-context_steps_to_show:] if len(ctx_full) > context_steps_to_show else ctx_full
            if iob_ctx_full is not None:
                iob_ctx = iob_ctx_full[-context_steps_to_show:] if len(iob_ctx_full) > context_steps_to_show else iob_ctx_full
            else:
                iob_ctx = None

            # Time axis in hours (0 = forecast start)
            t_ctx = (np.arange(len(ctx)) - len(ctx)) / 12  # negative hours before forecast
            t_pred = np.arange(len(tgt)) / 12  # positive hours after forecast start

            # Plot BG
            ax.plot(t_ctx, ctx, "b-", lw=1.5, label="BG (context)")
            ax.plot(t_pred, tgt, "g-", lw=2, label="BG (actual)")
            ax.plot(t_pred, pred, "r--", lw=2, alpha=0.8, label="BG (forecast)")

            ax.axvline(0, color="gray", ls=":", lw=1.5, label="Forecast start")
            ax.axhline(3.9, color="crimson", ls="--", alpha=0.4, lw=1, label="Hypo threshold")

            ax.set_ylabel("BG (mmol/L)", fontsize=9)
            ax.set_ylim(0, 18)

            # Plot IOB on secondary axis if available
            if iob_ctx is not None and iob_tgt is not None:
                ax2 = ax.twinx()
                ax2.plot(t_ctx, iob_ctx, "c-", alpha=0.5, lw=1, label="IOB")
                ax2.plot(t_pred, iob_tgt, "c-", alpha=0.5, lw=1)
                ax2.set_ylabel("IOB (U)", fontsize=8, color="cyan")
                ax2.tick_params(axis="y", labelcolor="cyan", labelsize=7)
                ax2.set_ylim(0, max(np.nanmax(iob_ctx), np.nanmax(iob_tgt)) * 1.2 + 0.1)

            rmse = model._compute_metrics(pred, tgt)["rmse"]
            ax.set_title(f"{ex['patient']}: RMSE={rmse:.2f} mmol/L", fontsize=10)
            ax.set_xlabel("Hours", fontsize=9)
            ax.tick_params(labelsize=8)
            ax.grid(alpha=0.3)

        for j in range(n, len(axes)):
            axes[j].set_visible(False)

        # Single legend for first plot
        axes[0].legend(fontsize=7, loc="upper right")
        fig.suptitle(f"Sundial Zero-Shot Forecasts - {args.dataset}\n(Context: {context_hours_to_show}h, Forecast: {args.forecast_hours}h)",
                     fontsize=12, fontweight="bold")
        plt.tight_layout()

        plot_file = output_path / "forecasts.png"
        plt.savefig(plot_file, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Plot saved to: {plot_file}")

    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
