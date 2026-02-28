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
        """

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Any, Dict, List

from src.data.versioning.dataset_registry import DatasetRegistry
from src.data.utils import get_patient_column
from src.evaluation.nocturnal import evaluate_nocturnal_forecasting, plot_best_worst_episodes
from src.models import create_model_and_config
from src.utils import get_git_commit_hash, setup_file_logging, load_yaml_config

# Constants
SAMPLING_INTERVAL_MINUTES = 5
STEPS_PER_HOUR = 60 // SAMPLING_INTERVAL_MINUTES

# Configure root logger for console output
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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

    # Auto-detect covariates from model config if not explicitly specified
    if args.covariate_cols is None and hasattr(config, "covariate_cols") and config.covariate_cols:
        args.covariate_cols = config.covariate_cols
        logger.info(f"Using covariates from model config: {args.covariate_cols}")

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
