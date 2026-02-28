#!/usr/bin/env python3
"""
End-to-end TiDE holdout workflow: training + nocturnal evaluation.

Combines training and evaluation into a single holdout workflow,
structured to match example_holdout_generic_workflow.py.

TiDE trains from scratch via AutoGluon — no pretrained weights, so Step 4
(zero-shot evaluation) is skipped. Evaluation uses midnight-anchored nocturnal
episodes via evaluate_nocturnal_forecasting().

Workflow Steps:
  1. Check holdout configs exist
  2. Validate holdout configs
  3. Load and combine training data
  -- (Step 4 skipped: TiDE trains from scratch, no zero-shot) --
  5. Train TiDE model from scratch
  6. Load model from checkpoint (verify save/load works)
  7. Resume training on loaded model

Each evaluation phase (5, 6, 7) runs evaluate_nocturnal_forecasting() and
saves results JSON + example forecast plots.

Usage:
    # Default: train + evaluate on Brown 2019
    python scripts/examples/example_tide_holdout_workflow.py --datasets brown_2019

    # With YAML config override
    python scripts/examples/example_tide_holdout_workflow.py \\
        --datasets brown_2019 \\
        --model-config configs/models/tide/default.yaml

    # Skip training (evaluate existing model)
    python scripts/examples/example_tide_holdout_workflow.py \\
        --datasets brown_2019 \\
        --skip-training \\
        --output-dir trained_models/artifacts/tide/20260226_123456
"""

import argparse
import json
import logging
import shutil
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.versioning.dataset_registry import DatasetRegistry
from src.data.preprocessing.dataset_combiner import (
    combine_datasets_for_training,
    print_dataset_column_table,
)
from src.models.tide import TiDEForecaster
from src.models.tide.config import TiDEConfig, create_default_tide_config
from src.utils.config_loader import load_yaml_config

from scripts.examples.holdout_eval import evaluate_nocturnal_forecasting

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Suppress verbose internal logging
logging.getLogger("src.data.preprocessing").setLevel(logging.WARNING)
logging.getLogger("src.data.diabetes_datasets").setLevel(logging.WARNING)
logging.getLogger("src.models").setLevel(logging.WARNING)
logging.getLogger("src.utils").setLevel(logging.WARNING)


# =============================================================================
# CONFIGURATION
# =============================================================================


def build_tide_config(args: argparse.Namespace, yaml_config: Optional[Dict]) -> TiDEConfig:
    """Build TiDEConfig with priority: CLI arg > YAML value > TiDEConfig default.

    Uses create_default_tide_config() from src/models/tide/config.py which
    applies validated defaults before any overrides.
    """
    overrides = dict(yaml_config) if yaml_config else {}

    # CLI args override YAML (only if explicitly provided)
    if args.time_limit is not None:
        overrides["time_limit"] = args.time_limit
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.context_length is not None:
        overrides["context_length"] = args.context_length
    if args.forecast_length is not None:
        overrides["forecast_length"] = args.forecast_length
    if args.learning_rate is not None:
        overrides["lr"] = args.learning_rate

    return create_default_tide_config(**overrides)


def _log_dir_size(dir_path: Path) -> None:
    """Log total size of a directory."""
    total = sum(f.stat().st_size for f in dir_path.rglob("*") if f.is_file())
    logger.info(f"  Directory size: {total / (1024 * 1024):.2f} MB")


# =============================================================================
# EVALUATION HELPER
# =============================================================================


def _evaluate_and_save(
    model: TiDEForecaster,
    holdout_data: pd.DataFrame,
    holdout_config,
    holdout_type: str,
    phase_name: str,
    output_dir: str,
    dataset_name: str,
) -> Optional[Dict[str, Any]]:
    """Run nocturnal evaluation and save results + plots.

    Replaces the generic workflow's _evaluate_and_plot() which uses
    _generate_forecasts() (incompatible ndarray interface). Instead calls
    evaluate_nocturnal_forecasting() from holdout_eval.py which handles
    TiDE's panel DataFrame predict natively.
    """
    logger.info("-" * 40)
    logger.info(f"Evaluating for phase: {phase_name} (holdout_type={holdout_type})")
    logger.info("-" * 40)

    config = model.config

    # Filter holdout data by type
    patient_holdout_ids = (
        holdout_config.patient_config.holdout_patients
        if holdout_config.patient_config
        else []
    )

    filtered_data = holdout_data.copy()
    if holdout_type == "temporal":
        filtered_data = filtered_data[
            ~filtered_data["p_num"].isin(patient_holdout_ids)
        ]
        logger.info(
            f"  Filtered to temporal holdout: {filtered_data.shape} "
            f"({filtered_data['p_num'].nunique()} patients)"
        )
    elif holdout_type == "patient":
        filtered_data = filtered_data[
            filtered_data["p_num"].isin(patient_holdout_ids)
        ]
        logger.info(
            f"  Filtered to patient holdout: {filtered_data.shape} "
            f"({filtered_data['p_num'].nunique()} patients)"
        )
    else:
        logger.info(
            f"  Using both holdout types: {filtered_data.shape} "
            f"({filtered_data['p_num'].nunique()} patients)"
        )

    try:
        # Run nocturnal evaluation
        results = evaluate_nocturnal_forecasting(
            model=model,
            holdout_data=filtered_data,
            context_length=config.context_length,
            forecast_length=config.forecast_length,
            target_col=config.target_col,
            covariate_cols=config.covariate_cols,
            interval_mins=config.interval_mins,
        )

        logger.info(
            f"  Overall RMSE: {results['overall_rmse']:.4f} "
            f"({results['total_episodes']} episodes)"
        )

        # Per-patient summary
        str_holdout_ids = [str(pid) for pid in patient_holdout_ids]
        for p in results["per_patient"]:
            tag = " [unseen]" if p["patient_id"] in str_holdout_ids else ""
            logger.info(
                f"    Patient {p['patient_id']}: "
                f"RMSE={p['rmse']:.3f}, MAE={p['mae']:.3f}, "
                f"episodes={p['episodes']}{tag}"
            )

        # Save results JSON to predictions/{phase_name}/
        predictions_dir = Path(output_dir) / "predictions" / phase_name
        predictions_dir.mkdir(parents=True, exist_ok=True)

        results_file = (
            predictions_dir / f"{dataset_name}_{holdout_type}_nocturnal.json"
        )

        save_data = {
            "phase": phase_name,
            "dataset": dataset_name,
            "holdout_type": holdout_type,
            "overall_rmse": results["overall_rmse"],
            "total_episodes": results["total_episodes"],
            "per_patient": results["per_patient"],
            "config": {
                "context_length": config.context_length,
                "forecast_length": config.forecast_length,
                "encoder_hidden_dim": config.encoder_hidden_dim,
                "scaling": config.scaling,
                "covariate_cols": config.covariate_cols,
                "time_limit": config.time_limit,
            },
            "timestamp": datetime.now().isoformat(),
        }

        with open(results_file, "w") as f:
            json.dump(save_data, f, indent=2, default=str)
        logger.info(f"  Results saved to: {results_file}")

        # Generate example forecast plots
        _plot_example_episodes(
            results=results,
            config=config,
            phase_name=phase_name,
            output_dir=output_dir,
            dataset_name=dataset_name,
        )

        return results

    except Exception as e:
        logger.error(f"  Failed to evaluate: {e}")
        traceback.print_exc()
        return None


def _select_representative_episodes(
    per_episode: List[Dict[str, Any]],
) -> List[Tuple[str, Dict[str, Any]]]:
    """Select 6 representative episodes: 2 best, 2 median, 2 worst by RMSE.

    Returns list of (tag, episode) tuples where tag is "best"/"median"/"worst".
    """
    if len(per_episode) <= 6:
        return [("", ep) for ep in per_episode]

    scored = []
    for ep in per_episode:
        pred = np.array(ep["pred"])
        target = np.array(ep["target_bg"])
        rmse = float(np.sqrt(np.mean((pred - target) ** 2)))
        scored.append((rmse, ep))
    scored.sort(key=lambda x: x[0])

    n = len(scored)
    mid = n // 2

    return [
        ("best", scored[0][1]),
        ("best", scored[1][1]),
        ("median", scored[mid - 1][1]),
        ("median", scored[mid][1]),
        ("worst", scored[-2][1]),
        ("worst", scored[-1][1]),
    ]


def _plot_example_episodes(
    results: Dict[str, Any],
    config: TiDEConfig,
    phase_name: str,
    output_dir: str,
    dataset_name: str,
    context_hours_to_show: int = 4,
) -> None:
    """Plot 6 representative nocturnal episodes: 2 best, 2 median, 2 worst.

    Each plot shows context BG (last N hours before midnight), actual BG over
    the 6h forecast horizon, predicted BG, a vertical forecast-start marker
    at midnight, hypo/hyper threshold lines, and per-episode RMSE.
    """
    per_episode = results.get("per_episode", [])
    if not per_episode:
        logger.info("  No episodes to plot")
        return

    forecast_dir = Path(output_dir) / "forecasts" / phase_name
    forecast_dir.mkdir(parents=True, exist_ok=True)

    episodes_to_plot = _select_representative_episodes(per_episode)
    steps_per_hour = 60 / config.interval_mins

    for i, (tag, ep) in enumerate(episodes_to_plot):
        pred = np.array(ep["pred"])
        target = np.array(ep["target_bg"])
        patient_id = ep["patient_id"]
        anchor = ep["anchor"]

        fig, ax = plt.subplots(figsize=(14, 5))

        # Context BG (truncated to last N hours for readability)
        context_bg = np.array(ep.get("context_bg", []))
        context_steps_to_show = int(context_hours_to_show * steps_per_hour)
        if len(context_bg) > context_steps_to_show:
            context_bg = context_bg[-context_steps_to_show:]

        # Time axes: context is negative hours (before midnight), forecast is positive
        t_ctx = (np.arange(len(context_bg)) - len(context_bg)) / steps_per_hour
        t_forecast = np.arange(len(pred)) / steps_per_hour

        # Plot context, actual, forecast
        if len(context_bg) > 0:
            ax.plot(t_ctx, context_bg, "b-", linewidth=1.5, label="BG (context)")
        ax.plot(t_forecast, target, "g-", linewidth=2, label="BG (actual)")
        ax.plot(t_forecast, pred, "r--", linewidth=2, alpha=0.8, label="BG (forecast)")

        # Forecast start marker at midnight (t=0)
        ax.axvline(0, color="gray", linestyle=":", linewidth=1.5, label="Midnight")

        # Hypo/hyper thresholds
        ax.axhline(
            y=3.9, color="orange", linestyle="--", linewidth=1, alpha=0.5,
            label="Hypoglycemia (3.9 mM)",
        )
        ax.axhline(
            y=10.0, color="red", linestyle="--", linewidth=1, alpha=0.5,
            label="Hyperglycemia (10.0 mM)",
        )

        ax.set_xlabel("Hours relative to midnight", fontsize=12)
        ax.set_ylabel("Blood Glucose (mM)", fontsize=12)

        tag_str = f" [{tag.upper()}]" if tag else ""
        ax.set_title(
            f"[{phase_name.upper()}]{tag_str} Nocturnal Forecast - "
            f"{dataset_name} (Patient {patient_id}, {anchor})",
            fontsize=13,
        )
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)

        # RMSE annotation
        ep_rmse = float(np.sqrt(np.mean((pred - target) ** 2)))
        ax.text(
            0.98, 0.02, f"RMSE: {ep_rmse:.3f} mM",
            transform=ax.transAxes, fontsize=11,
            verticalalignment="bottom", horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        suffix = f"_{tag}" if tag else ""
        plot_path = (
            forecast_dir / f"{phase_name}_{dataset_name}{suffix}_{i:02d}.png"
        )
        plt.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close()

    logger.info(f"  Saved {len(episodes_to_plot)} forecast plots to: {forecast_dir}")
    if len(per_episode) > 6:
        logger.info("    Selection: 2 best, 2 median, 2 worst (by episode RMSE)")


# =============================================================================
# STEP FUNCTIONS
# =============================================================================


def step1_generate_holdout_configs(
    config_dir: str = "configs/data/holdout",
    output_dir: str | None = None,
    datasets: list | None = None,
) -> bool:
    """Step 1: Check holdout configs exist and copy to artifacts directory.
    """
    logger.info("=" * 80)
    logger.info("STEP 1: Generate Holdout Configurations")
    logger.info("=" * 80)

    config_path = Path(config_dir)

    if config_path.exists():
        configs = list(config_path.glob("*.yaml"))
        logger.info(f"Holdout configs already exist: {len(configs)} datasets")
        for cfg in configs:
            logger.info(f"  - {cfg.stem}")

        # Copy matching configs to output_dir/configs/
        if output_dir and datasets:
            artifacts_config_dir = Path(output_dir) / "configs"
            artifacts_config_dir.mkdir(parents=True, exist_ok=True)

            copied_count = 0
            for cfg in configs:
                if cfg.stem in datasets:
                    shutil.copy2(cfg, artifacts_config_dir / cfg.name)
                    logger.info(f"  Copied: {cfg.name}")
                    copied_count += 1

            logger.info(
                f"Copied {copied_count}/{len(datasets)} configs to: {artifacts_config_dir}"
            )

        return True
    else:
        logger.warning(f"Config directory does not exist: {config_dir}")
        logger.info(
            "  Run: python scripts/data_processing_scripts/generate_holdout_configs.py"
        )
        return False


def step2_validate_holdout_configs(datasets: list, config_dir: str) -> bool:
    """Step 2: Validate holdout configurations for all datasets.

    Includes config detail logging (holdout_type, temporal %, patient count)
    and validation summary table.
    """
    logger.info(" ")
    logger.info("=" * 80)
    logger.info("STEP 2: Validate Holdout Configurations")
    logger.info(f"Validating {len(datasets)} dataset(s)")
    logger.info("=" * 80)

    from src.data.versioning import holdout_utils

    registry = DatasetRegistry(holdout_config_dir=config_dir)

    validation_results = []
    for idx, dataset_name in enumerate(datasets):
        logger.info(f"--- Dataset {idx + 1}/{len(datasets)}: {dataset_name} ---")
        config = registry.get_holdout_config(dataset_name)
        if config is None:
            logger.error(f"No config found for {dataset_name}")
            validation_results.append(
                {
                    "dataset_name": dataset_name,
                    "config_exists": False,
                    "load_successful": False,
                    "no_data_leakage": False,
                    "train_size": 0,
                    "holdout_size": 0,
                    "errors": ["No holdout configuration found"],
                }
            )
            continue

        # Config detail logging
        logger.info(f"Config loaded: {config.holdout_type.value}")
        if config.temporal_config:
            logger.info(
                f"  Temporal holdout: {config.temporal_config.holdout_percentage * 100}%"
            )
        if config.patient_config:
            logger.info(
                f"  Holdout patients: {len(config.patient_config.holdout_patients)}"
            )

        results = holdout_utils.validate_holdout_config(
            dataset_name, registry, verbose=False
        )
        validation_results.append(results)

        if results["errors"]:
            logger.error(f"Validation failed with {len(results['errors'])} error(s)")
            for error in results["errors"]:
                logger.error(f"    - {error}")
        else:
            logger.info("All comprehensive validations passed")

    # Validation summary table
    holdout_utils.print_validation_summary(validation_results, verbose=False)

    failed_datasets = [r["dataset_name"] for r in validation_results if r["errors"]]
    if failed_datasets:
        logger.error(f"Validation failed for: {', '.join(failed_datasets)}")
        return False

    logger.info("All datasets validated successfully")
    return True


def step3_load_training_data(
    dataset_names: list, config_dir: str, output_dir: Optional[str] = None
) -> pd.DataFrame:
    """Step 3: Load and combine training data from multiple datasets.
    """
    logger.info(" ")
    logger.info("=" * 80)
    logger.info("STEP 3: Load Training Data")
    logger.info("=" * 80)

    registry = DatasetRegistry(holdout_config_dir=config_dir)

    combined_data, column_info = combine_datasets_for_training(
        dataset_names=dataset_names, registry=registry, config_dir=config_dir
    )

    # Save split metadata (which patients skipped/adjusted)
    split_metadata = registry.get_split_metadata()
    if split_metadata and output_dir:
        metadata_dir = Path(output_dir)
        metadata_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = metadata_dir / "split_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(split_metadata, f, indent=2)
        logger.info(f"Split metadata saved to: {metadata_path}")

        for ds_name, meta in split_metadata.items():
            n_skipped = len(meta.get("skipped_patients", {}))
            n_adjusted = len(meta.get("adjusted_patients", {}))
            n_filled = meta.get("nan_p_num_filled", 0)
            if n_skipped or n_adjusted or n_filled:
                logger.info(
                    f"  {ds_name}: {n_skipped} skipped, {n_adjusted} adjusted, "
                    f"{n_filled:,} NaN p_num filled"
                )

    # Column comparison table
    print_dataset_column_table(column_info, list(combined_data.columns))

    logger.info("Combined training data ready")
    logger.info(f"  Total samples: {len(combined_data):,}")
    logger.info(f"  Total columns: {len(combined_data.columns)}")
    logger.info(f"  Datasets: {', '.join(dataset_names)}")

    if "p_num" in combined_data.columns:
        n_patients = len(combined_data["p_num"].unique())
        logger.info(f"  Total patients: {n_patients}")

    return combined_data


def step5_train_model(
    config: TiDEConfig,
    combined_data: pd.DataFrame,
    dataset_names: List[str],
    holdout_data: pd.DataFrame,
    holdout_config,
    holdout_type: str,
    output_dir: str,
) -> Tuple[TiDEForecaster, Dict, Path]:
    """Step 5: Train TiDE model from scratch.

    - Evaluates via _evaluate_and_save()
    - Adds GPU info logging + model dir size
    """
    logger.info(" ")
    logger.info("=" * 80)
    logger.info("STEP 5: Train TiDE Model")
    logger.info(f"Datasets: {', '.join(dataset_names)}")
    logger.info("=" * 80)

    # TiDE config summary
    logger.info("TiDE config:")
    logger.info(f"  Context: {config.context_length} ({config.context_length * 5 / 60:.1f}h)")
    logger.info(f"  Forecast: {config.forecast_length} ({config.forecast_length * 5 / 60:.1f}h)")
    logger.info(f"  Hidden dim: {config.encoder_hidden_dim}")
    logger.info(f"  Scaling: {config.scaling}")
    logger.info(f"  LR: {config.lr}")
    logger.info(f"  Covariates: {config.covariate_cols}")
    logger.info(f"  Time limit: {config.time_limit}")
    logger.info(f"  Min segment length: {config.min_segment_length}")

    # GPU info
    try:
        import torch
        logger.info(f"  GPU available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"  GPU count: {torch.cuda.device_count()}")
            logger.info(f"  GPU name: {torch.cuda.get_device_name(0)}")
    except ImportError:
        logger.info("  GPU info: torch not available")

    # Create and train model
    model = TiDEForecaster(config)

    print(f"\n>>> Starting TiDE training on: {', '.join(dataset_names)}")
    print(f">>> Output directory: {output_dir}")
    print(f">>> Time limit: {config.time_limit}s\n")

    try:
        results = model.fit(train_data=combined_data, output_dir=output_dir)
        print("\n>>> Training completed successfully\n")
        logger.info("Training completed")
        logger.info(f"  Results keys: {list(results.keys())}")

        # Save model checkpoint
        model_path = Path(output_dir) / "model.pt"
        model_path.mkdir(parents=True, exist_ok=True)
        model.save(str(model_path))
        logger.info(f"Model saved to: {model_path}")
        _log_dir_size(model_path)

        # Evaluate after training
        for ds_name in dataset_names:
            _evaluate_and_save(
                model=model,
                holdout_data=holdout_data,
                holdout_config=holdout_config,
                holdout_type=holdout_type,
                phase_name="1_after_training",
                output_dir=output_dir,
                dataset_name=ds_name,
            )

        return model, results, model_path

    except Exception as e:
        print(f"\n>>> ERROR: Training failed: {e}\n")
        logger.error(f"Training failed: {e}")
        raise


def step6_load_checkpoint(
    model_path: Path,
    dataset_names: List[str],
    holdout_data: pd.DataFrame,
    holdout_config,
    holdout_type: str,
    output_dir: str,
) -> Optional[TiDEForecaster]:
    """Step 6: Load model from checkpoint and verify it works.
    """
    logger.info(" ")
    logger.info("=" * 80)
    logger.info("STEP 6: Load Model from Checkpoint")
    logger.info("=" * 80)

    model_path = Path(model_path)

    if not model_path.exists():
        logger.error(f"Model path not found: {model_path}")
        return None

    logger.info(f"Loading model from: {model_path}")

    try:
        model = TiDEForecaster.load(str(model_path))
        logger.info(f"Model loaded successfully from: {model_path}")

        # Evaluate after loading
        for ds_name in dataset_names:
            _evaluate_and_save(
                model=model,
                holdout_data=holdout_data,
                holdout_config=holdout_config,
                holdout_type=holdout_type,
                phase_name="2_after_loading",
                output_dir=output_dir,
                dataset_name=ds_name,
            )

        return model

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        traceback.print_exc()
        return None


def step7_resume_training(
    model: TiDEForecaster,
    combined_data: pd.DataFrame,
    dataset_names: List[str],
    holdout_data: pd.DataFrame,
    holdout_config,
    holdout_type: str,
    output_dir: str,
) -> Tuple[TiDEForecaster, Dict, Path]:
    """Step 7: Resume training on loaded model for additional time.
    """
    logger.info(" ")
    logger.info("=" * 80)
    logger.info("STEP 7: Resume Training on Loaded Model")
    logger.info(f"Datasets: {', '.join(dataset_names)}")
    logger.info("=" * 80)

    resumed_output_dir = Path(output_dir) / "resumed_training"
    resumed_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n>>> Resuming TiDE training on: {', '.join(dataset_names)}")
    print(f">>> Output directory: {resumed_output_dir}\n")

    try:
        results = model.fit(
            train_data=combined_data, output_dir=str(resumed_output_dir)
        )
        print("\n>>> Resumed training completed successfully\n")
        logger.info("Resumed training completed")
        logger.info(f"  Results keys: {list(results.keys())}")

        # Save resumed model
        model_path = resumed_output_dir / "model.pt"
        model_path.mkdir(parents=True, exist_ok=True)
        model.save(str(model_path))
        logger.info(f"Resumed model saved to: {model_path}")
        _log_dir_size(model_path)

        # Evaluate after resumed training
        for ds_name in dataset_names:
            _evaluate_and_save(
                model=model,
                holdout_data=holdout_data,
                holdout_config=holdout_config,
                holdout_type=holdout_type,
                phase_name="3_after_resumed_training",
                output_dir=output_dir,
                dataset_name=ds_name,
            )

        return model, results, model_path

    except Exception as e:
        print(f"\n>>> ERROR: Resumed training failed: {e}\n")
        logger.error(f"Resumed training failed: {e}")
        raise


# =============================================================================
# MAIN WORKFLOW
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end TiDE holdout workflow: training + nocturnal evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Workflow Steps:
  1. Check holdout configs exist
  2. Validate holdout configs
  3. Load and combine training data
  -- (Step 4 skipped: TiDE trains from scratch) --
  5. Train TiDE model from scratch
  6. Load model from checkpoint (verify save/load works)
  7. Resume training on loaded model

TiDE: Time-series Dense Encoder via AutoGluon TimeSeriesPredictor.
Trains from scratch — no pretrained weights or zero-shot capability.
        """,
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["brown_2019"],
        help="Dataset names to combine (default: brown_2019)",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="configs/data/holdout_5pct",
        help="Holdout config directory (default: configs/data/holdout_5pct)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (auto-generated timestamp if not specified)",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training steps (only load existing model and evaluate)",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default=None,
        help="Path to YAML model config (e.g., configs/models/tide/default.yaml)",
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        default=None,
        help="AutoGluon time limit in seconds (default: from YAML or 3600)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (default: from YAML or 256)",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=None,
        help="Context window in steps (default: from YAML or 512)",
    )
    parser.add_argument(
        "--forecast-length",
        type=int,
        default=None,
        help="Forecast horizon in steps (default: from YAML or 72)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate (default: from YAML or 0.000931)",
    )
    parser.add_argument(
        "--holdout-type",
        type=str,
        default="both",
        choices=["temporal", "patient", "both"],
        help="Which holdout subset to evaluate (default: both)",
    )

    args = parser.parse_args()

    # Load YAML config if provided
    yaml_config = None
    if args.model_config:
        yaml_config = load_yaml_config(args.model_config)
        logger.info(f"Loaded model config from: {args.model_config}")

    # Set output directory (auto-generate if not specified)
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M")
        args.output_dir = (
            f"./trained_models/artifacts/tide/{timestamp}_tide_holdout_workflow"
        )

    logger.info("=" * 80)
    logger.info("TiDE HOLDOUT WORKFLOW")
    logger.info("=" * 80)
    logger.info(f"Datasets: {', '.join(args.datasets)}")
    logger.info(f"Config dir: {args.config_dir}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"Model config: {args.model_config or 'None (using defaults)'}")
    logger.info(f"Holdout type: {args.holdout_type}")
    logger.info(f"Skip training: {args.skip_training}")
    logger.info("=" * 80)

    # Save run_config.json
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    run_config = {
        "datasets": args.datasets,
        "config_dir": args.config_dir,
        "holdout_type": args.holdout_type,
        "model_config_yaml": args.model_config,
        "cli_overrides": {
            "time_limit": args.time_limit,
            "batch_size": args.batch_size,
            "context_length": args.context_length,
            "forecast_length": args.forecast_length,
            "learning_rate": args.learning_rate,
        },
        "skip_training": args.skip_training,
        "timestamp": datetime.now().isoformat(),
    }
    with open(output_path / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)
    logger.info(f"Run config saved to: {output_path / 'run_config.json'}")

    # Copy model config YAML to output dir for reproducibility
    if args.model_config:
        shutil.copy2(args.model_config, output_path / "model_config.yaml")
        logger.info(f"Copied model config to: {output_path / 'model_config.yaml'}")

    try:
        # =================================================================
        # STEP 1: Check/generate holdout configs
        # =================================================================
        if not step1_generate_holdout_configs(
            args.config_dir, args.output_dir, args.datasets
        ):
            logger.error("Please generate holdout configs first")
            return

        # =================================================================
        # STEP 2: Validate configuration for all datasets
        # =================================================================
        if not step2_validate_holdout_configs(args.datasets, args.config_dir):
            logger.error("Configuration validation failed")
            return

        # =================================================================
        # STEP 3: Load and combine training data
        # =================================================================
        combined_train_data = step3_load_training_data(
            args.datasets, args.config_dir, args.output_dir
        )

        # Guard: check data is not empty
        if combined_train_data.empty:
            logger.error("No training data loaded — aborting")
            return

        # =================================================================
        # Load holdout data for evaluation (used by steps 5, 6, 7)
        # Loaded once here, reused across steps
        # =================================================================
        registry = DatasetRegistry(holdout_config_dir=args.config_dir)
        eval_dataset = args.datasets[0]
        holdout_data = registry.load_holdout_data_only(eval_dataset)
        holdout_config = registry.get_holdout_config(eval_dataset)

        logger.info(
            f"Holdout data loaded for {eval_dataset}: {holdout_data.shape} "
            f"({holdout_data['p_num'].nunique()} patients)"
        )

        # =================================================================
        # STEP 4: SKIPPED — TiDE trains from scratch (no zero-shot)
        # =================================================================
        logger.info(" ")
        logger.info("=" * 80)
        logger.info(
            "STEP 4: SKIPPED (TiDE trains from scratch, no zero-shot capability)"
        )
        logger.info("=" * 80)

        if args.skip_training:
            # =============================================================
            # SKIP TRAINING — load existing model and evaluate
            # =============================================================
            logger.info(" ")
            logger.info("=" * 80)
            logger.info("SKIPPING TRAINING STEPS (--skip-training flag set)")
            logger.info("=" * 80)

            model_path = Path(args.output_dir) / "model.pt"
            if not model_path.exists():
                logger.error(f"No existing model found at: {model_path}")
                return

            model = step6_load_checkpoint(
                model_path=model_path,
                dataset_names=[eval_dataset],
                holdout_data=holdout_data,
                holdout_config=holdout_config,
                holdout_type=args.holdout_type,
                output_dir=args.output_dir,
            )
            if model is None:
                logger.error("Failed to load existing model")
                return
        else:
            # =============================================================
            # STEP 5: Train TiDE model
            # =============================================================
            tide_config = build_tide_config(args, yaml_config)

            model, _, model_path = step5_train_model(
                config=tide_config,
                combined_data=combined_train_data,
                dataset_names=args.datasets,
                holdout_data=holdout_data,
                holdout_config=holdout_config,
                holdout_type=args.holdout_type,
                output_dir=args.output_dir,
            )

            # =============================================================
            # STEP 6: Load model from checkpoint (verify save/load)
            # =============================================================
            model = step6_load_checkpoint(
                model_path=model_path,
                dataset_names=[eval_dataset],
                holdout_data=holdout_data,
                holdout_config=holdout_config,
                holdout_type=args.holdout_type,
                output_dir=args.output_dir,
            )
            if model is None:
                logger.error("Failed to load model from checkpoint")
                return

            # =============================================================
            # STEP 7: Resume training on loaded model
            # =============================================================
            model, _, _ = step7_resume_training(
                model=model,
                combined_data=combined_train_data,
                dataset_names=args.datasets,
                holdout_data=holdout_data,
                holdout_config=holdout_config,
                holdout_type=args.holdout_type,
                output_dir=args.output_dir,
            )

        # =================================================================
        # WORKFLOW COMPLETE — artifact listing
        # =================================================================
        logger.info("\n" + "=" * 80)
        logger.info("WORKFLOW COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        logger.info(f"Output directory: {args.output_dir}")
        logger.info("Generated artifacts:")
        logger.info("  - run_config.json                        : Full run configuration")
        if not args.skip_training:
            logger.info(
                "  - predictions/1_after_training/           : Post-training nocturnal eval"
            )
            logger.info(
                "  - predictions/2_after_loading/            : Post-load nocturnal eval"
            )
            logger.info(
                "  - predictions/3_after_resumed_training/   : Post-resume nocturnal eval"
            )
            logger.info(
                "  - model.pt/                               : Trained model"
            )
            logger.info(
                "  - resumed_training/model.pt/              : Resumed training model"
            )
        else:
            logger.info(
                "  - predictions/2_after_loading/            : Post-load nocturnal eval"
            )
        logger.info(
            "  - forecasts/*/                            : Example forecast plots"
        )
        logger.info("=" * 80)

    except KeyboardInterrupt:
        logger.info("\n\nWorkflow interrupted by user")
    except Exception as e:
        logger.error(f"\n\nWorkflow failed: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
