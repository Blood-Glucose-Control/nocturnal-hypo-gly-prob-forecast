#!/usr/bin/env python3
r"""
Nocturnal Hypoglycemia Evaluation Script.

Evaluates models on the midnight-anchored nocturnal forecasting task — a clinically
meaningful evaluation where context ends at midnight and the model forecasts 6 hours
of overnight blood glucose when hypoglycemia risk is highest.

This is fundamentally different from sliding-window evaluation (sliding_window_eval.py),
which measures general forecast accuracy across all times of day. These two evaluation
modes produce DIFFERENT RMSE numbers and must never be compared on the same leaderboard.

Usage:
    python scripts/experiments/nocturnal_hypo_eval.py --model ttm --dataset brown_2019
    # sundial zero-shot (no checkpoint):
    python scripts/experiments/nocturnal_hypo_eval.py \
        --model sundial \
        --dataset tamborlane_2008 \
        --config-dir configs/data/holdout_10pct \
        --context-length 512 \
        --forecast-length 96 \
        --cuda-device 0

    # timesfm ft-shot (no checkpoint):
    python scripts/experiments/nocturnal_hypo_eval.py \
        --model timesfm \
        --dataset tamborlane_2008 \
        --config-dir configs/data/holdout_10pct \
        --context-length 512 \
        --forecast-length 96 \
        --cuda-device 1

        --checkpoint trained_models/artifacts/timesfm/2026-02-27_05:37_RID20260227_053718_211403_holdout_workflow/resumed_training/model.pt \

    # ttm zero-shot (no checkpoint):
    python scripts/experiments/nocturnal_hypo_eval.py \
        --model ttm \
        --dataset tamborlane_2008 \
        --config-dir configs/data/holdout_10pct \
        --context-length 512 \
        --forecast-length 96 \
        --cuda-device 0

        --checkpoint trained_models/artifacts/ttm/2026-02-27_03:53_RID20260227_035316_193673_holdout_workflow/model.pt \

    python scripts/experiments/nocturnal_hypo_eval.py \
        --model ttm \
        --context-length 512 \
        --forecast-length 96

    # TimeGrad — after first 10-epoch training run (aleppo_2017):
    python scripts/experiments/nocturnal_hypo_eval.py \
        --model timegrad \
        --dataset aleppo_2017 \
        --config-dir configs/data/holdout_10pct \
        --model-config configs/models/timegrad/cgm_only.yaml \
        --context-length 512 \
        --forecast-length 48 \
        --cuda-device 1

    # TimeGrad — after second 10-epoch resumed training run (lynch_2022, epochs 11–20):
    python scripts/experiments/nocturnal_hypo_eval.py \
        --model timegrad \
        --dataset aleppo_2017 \
        --config-dir configs/data/holdout_10pct \
        --model-config configs/models/timegrad/cgm_only.yaml \
        --context-length 512 \
        --forecast-length 96 \
        --checkpoint trained_models/artifacts/timegrad/2026-02-24_01:12_RID20260224_011201_2800320_holdout_workflow/resumed_training/model.pt

    # Chronos-2 fine-tuned (checkpoint with model.pt/ directory):
    python scripts/experiments/nocturnal_hypo_eval.py \
        --model chronos2 \
        --dataset tamborlane_2008 \
        --config-dir configs/data/holdout_10pct \
        --context-length 512 \
        --forecast-length 96 \
        --cuda-device 1 \
        --checkpoint trained_models/artifacts/chronos2/2026-02-28_05:54_RID20260228_055400_391511_holdout_workflow/resumed_training/model.pt

    # TiDE fine-tuned:
    python scripts/experiments/nocturnal_hypo_eval.py \
        --model tide \
        --dataset tamborlane_2008 \
        --config-dir configs/data/holdout_10pct \
        --context-length 512 \
        --forecast-length 96 \
        --cuda-device 1 \
        --checkpoint trained_models/artifacts/tide/2026-02-28_21:28_RID20260228_212852_496983_holdout_workflow/model.pt
        """

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from src.data.versioning.dataset_registry import DatasetRegistry
from src.data.utils import get_patient_column
from src.evaluation.nocturnal import (
    evaluate_nocturnal_forecasting,
    plot_best_worst_episodes,
    STEPS_PER_HOUR,
)
from src.models import create_model_and_config
from src.utils import get_git_commit_hash, setup_file_logging, load_yaml_config

# Configure root logger for console output
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def compute_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """Compute evaluation metrics."""
    return compute_regression_metrics(predictions, targets)


def evaluate_nocturnal_forecasting(
    model: BaseTimeSeriesFoundationModel,
    holdout_data: pd.DataFrame,
    context_length: int,
    forecast_length: int,
    target_col: str = "bg_mM",
    covariate_cols: Optional[List[str]] = None,
    interval_mins: int = SAMPLING_INTERVAL_MINUTES,
) -> Dict[str, Any]:
    """Evaluate model on midnight-anchored nocturnal forecasting task.

    Builds midnight episodes per patient and evaluates each episode individually.
    This per-episode approach ensures compatibility with models like TTM that
    use ForecastDFDataset internally (which creates sliding windows rather than
    treating pre-windowed inputs as separate samples).

    Args:
        model: Model implementing predict(data) -> np.ndarray.
        holdout_data: Flat DataFrame with all holdout patients.
        context_length: Context window size in steps.
        forecast_length: Forecast horizon in steps.
        target_col: BG column name.
        covariate_cols: Covariate column names (e.g., ["iob"]).
        interval_mins: Sampling interval in minutes.

    Returns:
        Dict with overall_rmse, total_episodes, per_patient, per_episode.
    """
    patient_col = get_patient_column(holdout_data)
    patients = holdout_data[patient_col].unique()

    # --- Phase 1: Build episodes for all patients ---
    episode_metadata = []
    context_dfs = []
    future_cov_dfs = []

    for patient_id in patients:
        patient_df = holdout_data[holdout_data[patient_col] == patient_id].copy()

        # Set DatetimeIndex if not already set
        if not isinstance(patient_df.index, pd.DatetimeIndex):
            time_col = "datetime" if "datetime" in patient_df.columns else None
            if time_col:
                patient_df[time_col] = pd.to_datetime(patient_df[time_col])
                patient_df = patient_df.set_index(time_col).sort_index()
            else:
                logger.warning(
                    "Patient %s: no datetime column or DatetimeIndex, skipping",
                    patient_id,
                )
                continue

        episodes, skip_stats = build_midnight_episodes(
            patient_df,
            context_length=context_length,
            forecast_length=forecast_length,
            target_col=target_col,
            covariate_cols=covariate_cols,
            interval_mins=interval_mins,
        )

        if not episodes:
            logger.info("  Patient %s: no valid midnight episodes", patient_id)
            if skip_stats["skipped_bg_nan"] > 0:
                logger.debug(
                    "    Skipped %d/%d anchors due to missing BG",
                    skip_stats["skipped_bg_nan"],
                    skip_stats["total_anchors"],
                )
            continue

        for i, ep in enumerate(episodes):
            # Use :: delimiter to avoid collision if patient_id contains "_ep"
            ep_id = f"{patient_id}::ep{i:03d}"

            ctx = ep["context_df"].copy()
            # Convert DatetimeIndex to column for models that expect it (e.g., TTM)
            ctx = ctx.reset_index(names="datetime")
            # Ensure patient identifier column is present for models like TTM
            # which expect 'p_num' as an ID column.  If the original dataset
            # used a different patient column (e.g. 'id'), the value is still
            # available via patient_id so we always add 'p_num' here.
            ctx["p_num"] = patient_id
            ctx["episode_id"] = ep_id
            # Add group column for panel data batching (required by some models)
            ctx["group"] = ep_id
            context_dfs.append(ctx)

            # Future covariates panel (if any covariates available)
            if ep["future_covariates"]:
                future_ts = pd.date_range(
                    ep["anchor"],
                    periods=forecast_length,
                    freq=f"{interval_mins}min",
                )
                future_data = {
                    cov_col: cov_vals
                    for cov_col, cov_vals in ep["future_covariates"].items()
                }
                # Future covariates should also carry the patient id so that
                # any model preprocessing that groups by id works correctly.
                future_data["p_num"] = patient_id
                future_data["episode_id"] = ep_id
                future_data["datetime"] = future_ts
                future_data["group"] = ep_id
                future_cov_dfs.append(pd.DataFrame(future_data))

            episode_metadata.append(
                {
                    "episode_id": ep_id,
                    "patient_id": str(patient_id),
                    "anchor": ep["anchor"],
                    "target_bg": ep["target_bg"],
                }
            )

    if not episode_metadata:
        logger.warning("No valid midnight episodes found across all patients")
        return {
            "overall_rmse": float("nan"),
            "total_episodes": 0,
            "per_patient": [],
            "per_episode": [],
        }

    # --- Phase 2: Iterate and call predict() per episode ---
    # Note: We iterate per-episode rather than batching because models like TTM
    # use ForecastDFDataset which creates sliding windows over the data rather
    # than treating pre-windowed episodes as separate samples.
    logger.info(
        "Evaluating %d midnight episodes across %d patients",
        len(episode_metadata),
        len(set(m["patient_id"] for m in episode_metadata)),
    )

    all_episode_results = []
    patient_episodes = {}

    for ctx_df, meta in tqdm(
        zip(context_dfs, episode_metadata),
        total=len(episode_metadata),
        desc="Evaluating episodes",
        unit="ep",
    ):
        target = meta["target_bg"]

        # Skip episodes that don't have the full forecast horizon
        if len(target) < forecast_length:
            continue

        # Ensure the context frame has a datetime index for models that
        # require it (TiDE) while still retaining the 'datetime' column for
        # others (TTM).  The context df was originally reset_index in
        # episode construction, so we put the column back on the index.
        if "datetime" in ctx_df.columns and not isinstance(
            ctx_df.index, pd.DatetimeIndex
        ):
            ctx_df = ctx_df.set_index("datetime", drop=False)

        # Predict using unified interface
        raw_pred = model.predict(ctx_df)

        # Convert heterogeneous outputs (DataFrame, Series, list, ndarray)
        # into a flat numpy float array representing the forecast values.
        if isinstance(raw_pred, pd.DataFrame):
            # prefer the target column but fall back to numeric columns
            if target_col in raw_pred.columns:
                pred_vals = raw_pred[target_col].values
            else:
                num_cols = raw_pred.select_dtypes(include=[np.number]).columns
                if len(num_cols) == 0:
                    raise ValueError("Prediction DataFrame contains no numeric columns")
                pred_vals = raw_pred[num_cols[0]].values
        elif isinstance(raw_pred, pd.Series):
            pred_vals = raw_pred.values
        else:
            pred_vals = np.asarray(raw_pred)

        # Ensure numeric dtype (will raise if values cannot be converted)
        try:
            pred = np.asarray(pred_vals, dtype=float)
        except Exception as e:  # pragma: no cover - defensive
            logger.error("Failed to convert predictions to float: %s", e)
            raise

        # After conversion we may still have extra dims (e.g. 2D from DataFrame
        # flattening) so apply earlier logic.
        if pred.ndim == 3:
            # Shape (batch, forecast_length, channels) -> (forecast_length,)
            pred = pred[0, :, 0]
        elif pred.ndim == 2:
            # Shape (batch, forecast_length) -> (forecast_length,)
            pred = pred.flatten()

        target = meta["target_bg"]

        # Truncate prediction to match target length if needed
        if len(pred) > len(target):
            pred = pred[: len(target)]

        ep_rmse = float(np.sqrt(np.mean((pred - target) ** 2)))

        # Extract context BG values for plotting
        context_bg = ctx_df[target_col].values if target_col in ctx_df.columns else None

        all_episode_results.append(
            {
                "patient_id": meta["patient_id"],
                "anchor": meta["anchor"].isoformat(),
                "rmse": ep_rmse,
                "pred": pred.tolist(),
                "target_bg": target.tolist(),
                "context_bg": context_bg.tolist() if context_bg is not None else None,
            }
        )

        pid = meta["patient_id"]
        if pid not in patient_episodes:
            patient_episodes[pid] = []
        patient_episodes[pid].append((pred, target))

    # Per-patient aggregate
    all_patient_results = []
    for pid, ep_list in patient_episodes.items():
        preds = np.concatenate([p for p, _ in ep_list])
        targets = np.concatenate([t for _, t in ep_list])
        metrics = compute_metrics(preds, targets)

        all_patient_results.append(
            {
                "patient_id": pid,
                "episodes": len(ep_list),
                **metrics,
            }
        )

        logger.info(
            "  Patient %s: RMSE=%.3f, MAE=%.3f (%d midnight episodes)",
            pid,
            metrics["rmse"],
            metrics["mae"],
            len(ep_list),
        )

    # Overall metrics (concatenated predictions)
    all_preds = np.concatenate([np.array(ep["pred"]) for ep in all_episode_results])
    all_targets = np.concatenate(
        [np.array(ep["target_bg"]) for ep in all_episode_results]
    )
    overall_rmse = float(np.sqrt(np.mean((all_preds - all_targets) ** 2)))

    logger.info(
        "Nocturnal evaluation: %.4f RMSE over %d midnight episodes",
        overall_rmse,
        len(all_episode_results),
    )

    return {
        "overall_rmse": overall_rmse,
        "total_episodes": len(all_episode_results),
        "per_patient": all_patient_results,
        "per_episode": all_episode_results,
    }


def setup_output_directory(
    model_name: str,
    dataset_name: str,
    context_length: int,
    forecast_length: int,
    checkpoint: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Path:
    """Create and return output directory path."""
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
        mode = "finetuned" if checkpoint else "zeroshot"
        output_dir = (
            f"./experiments/nocturnal_forecasting/"
            f"{context_length}ctx_{forecast_length}fh/{model_name}/"
            f"{timestamp}_{dataset_name}_{mode}"
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def save_experiment_config(
    args: argparse.Namespace,
    model_config: Dict[str, Any],
    output_path: Path,
) -> None:
    """Save experiment configuration for reproducibility."""
    cmd_parts = ["python", "scripts/experiments/nocturnal_hypo_eval.py"]
    cmd_parts.extend(["--model", args.model])
    cmd_parts.extend(["--dataset", args.dataset])
    cmd_parts.extend(["--config-dir", args.config_dir])
    cmd_parts.extend(["--context-length", str(args.context_length)])
    cmd_parts.extend(["--forecast-length", str(args.forecast_length)])
    cmd_parts.extend(["--cuda-device", str(args.cuda_device)])
    if args.checkpoint:
        cmd_parts.extend(["--checkpoint", args.checkpoint])
    if args.model_config:
        cmd_parts.extend(["--model-config", args.model_config])

    config = {
        "evaluation_type": "nocturnal_hypoglycemia",
        "cli_args": {
            "model": args.model,
            "dataset": args.dataset,
            "config_dir": args.config_dir,
            "checkpoint": args.checkpoint,
            "context_length": args.context_length,
            "forecast_length": args.forecast_length,
            "cuda_device": args.cuda_device,
            "model_config": args.model_config,
            "output_dir": args.output_dir,
        },
        "model_config": model_config,
        "environment": {
            "git_commit": get_git_commit_hash(),
            "python_version": sys.version.split()[0],
            "timestamp": datetime.now().isoformat(),
        },
        "reproducibility_command": " ".join(cmd_parts),
    }

    config_file = output_path / "experiment_config.json"
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Experiment config saved to: {config_file}")


def save_results(results: Dict[str, Any], output_path: Path) -> None:
    """Save evaluation results to JSON file."""
    results_file = output_path / "nocturnal_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to: {results_file}")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Nocturnal hypoglycemia evaluation (midnight-anchored forecasting)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ttm",
        choices=["sundial", "ttm", "chronos2", "moirai", "timegrad", "timesfm", "tide"],
        help="Model type to use for evaluation",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default=None,
        help="Path to model config YAML file",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="brown_2019",
        help="Dataset name",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="configs/data/holdout_10pct",
        help="Holdout config directory",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to fine-tuned checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=512,
        help="Context window length in steps (default: 512)",
    )
    parser.add_argument(
        "--forecast-length",
        type=int,
        default=72,
        help="Forecast horizon in steps (default: 72 = 6 hours)",
    )
    parser.add_argument(
        "--covariate-cols",
        type=str,
        nargs="*",
        default=None,
        help="Covariate column names (e.g., iob cob)",
    )
    parser.add_argument(
        "--cuda-device",
        type=int,
        default=1,
        help="CUDA device ID to use (default: 1)",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    # Set CUDA device to avoid DataParallel issues
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)

    # Load config from file if provided
    config_dict = load_yaml_config(args.model_config) if args.model_config else {}

    # Prepare model kwargs
    model_kwargs = {**config_dict}
    if args.context_length is not None:
        model_kwargs["context_length"] = args.context_length
    if args.forecast_length is not None:
        model_kwargs["forecast_length"] = args.forecast_length

    # Initialize model
    logger.info("=" * 60)
    logger.info("NOCTURNAL HYPOGLYCEMIA EVALUATION")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Mode: {'Fine-tuned' if args.checkpoint else 'Zero-shot'}")

    logger.info(f"\n--- Initializing {args.model.upper()} ---")
    model, config = create_model_and_config(
        args.model, checkpoint=args.checkpoint, **model_kwargs
    )

    context_length = config.context_length
    forecast_length = config.forecast_length

    # Warn if user asked for a different forecast length than the model's
    # configuration.  This commonly happens when loading a checkpoint that was
    # trained with a shorter horizon (e.g. 72) but the CLI requested a longer
    # one (e.g. 96).  TiDE factory code refuses to increase the horizon beyond
    # the trained value, so the printed forecast_length will remain the smaller
    # number.  We surface that here so it's harder to miss.
    if args.forecast_length is not None and args.forecast_length != forecast_length:
        if args.forecast_length > forecast_length:
            logger.warning(
                "Requested forecast-length %d is larger than the model's trained "
                "value (%d); using %d (training horizon).",
                args.forecast_length,
                forecast_length,
                forecast_length,
            )
        else:
            logger.info(
                "Overriding model forecast_length %d -> %d as requested.",
                forecast_length,
                args.forecast_length,
            )
            forecast_length = args.forecast_length

    logger.info(f"Dataset: {args.dataset}")
    logger.info(
        f"Context: {context_length} steps ({context_length / STEPS_PER_HOUR:.1f} hours)"
    )
    logger.info(
        f"Forecast: {forecast_length} steps ({forecast_length / STEPS_PER_HOUR:.1f} hours)"
    )

    # Setup output directory
    output_path = setup_output_directory(
        args.model,
        args.dataset,
        context_length,
        forecast_length,
        args.checkpoint,
        args.output_dir,
    )

    log_file = setup_file_logging(output_path, "nocturnal_evaluation.log")
    logger.info(f"Output: {output_path}")
    logger.info(f"Log file: {log_file}")

    # Re-log configuration to file
    logger.info("=" * 60)
    logger.info("EVALUATION CONFIGURATION")
    logger.info("=" * 60)
    logger.info("Evaluation type: Nocturnal Hypoglycemia (midnight-anchored)")
    logger.info(f"Model: {args.model}")
    logger.info(f"Mode: {'Fine-tuned' if args.checkpoint else 'Zero-shot'}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(
        f"Context: {context_length} steps ({context_length / STEPS_PER_HOUR:.1f} hours)"
    )
    logger.info(
        f"Forecast: {forecast_length} steps ({forecast_length / STEPS_PER_HOUR:.1f} hours)"
    )

    # Load holdout data
    logger.info("\n--- Loading Holdout Data ---")
    registry = DatasetRegistry(holdout_config_dir=args.config_dir)
    holdout_data = registry.load_holdout_data_only(args.dataset)

    patient_col = get_patient_column(holdout_data)
    patients = holdout_data[patient_col].unique()
    logger.info(f"Holdout patients: {list(patients)}")
    logger.info(f"Total samples: {len(holdout_data):,}")

    # Inform user if any requested covariates are absent in the dataset.
    if args.covariate_cols:
        missing = [c for c in args.covariate_cols if c not in holdout_data.columns]
        if missing:
            logger.warning(
                "Dataset is missing requested covariate columns %s; "
                "these will be filled with zeros during prediction.",
                missing,
            )

    # Save experiment configuration
    save_experiment_config(
        args,
        {
            "context_length": context_length,
            "forecast_length": forecast_length,
            **config_dict,
        },
        output_path,
    )

    # Run nocturnal evaluation
    logger.info("\n--- Running Nocturnal Evaluation ---")
    results = evaluate_nocturnal_forecasting(
        model=model,
        holdout_data=holdout_data,
        context_length=context_length,
        forecast_length=forecast_length,
        covariate_cols=args.covariate_cols,
    )

    # Log overall results
    logger.info("\n")
    logger.info("=" * 60)
    logger.info("OVERALL RESULTS")
    logger.info("=" * 60)
    logger.info(f"Overall RMSE: {results['overall_rmse']:.4f}")
    logger.info(f"Total midnight episodes: {results['total_episodes']}")

    # Prepare full results
    full_results = {
        "evaluation_type": "nocturnal_hypoglycemia",
        "model": args.model,
        "mode": "fine-tuned" if args.checkpoint else "zero-shot",
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "context_length": context_length,
            "forecast_length": forecast_length,
            **config_dict,
        },
        **results,
    }

    # Save results
    save_results(full_results, output_path)

    # Generate best/worst episode plots
    logger.info("\n--- Generating Plots ---")
    plot_best_worst_episodes(
        per_episode=results["per_episode"],
        output_path=output_path,
        model_name=args.model,
        dataset_name=args.dataset,
        is_finetuned=bool(args.checkpoint),
    )

    logger.info("\n")
    logger.info("=" * 60)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
