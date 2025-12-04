# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)
# For commercial licensing, contact: [Add your contact information]

#!/usr/bin/env python3
"""
Enhanced TTM Training Runner
Handles metrics collection and saves to a JSON file for model registry integration.
"""

import argparse
import functools
import json
import sys
import threading
from pathlib import Path
from src.utils.logging_helper import info_print
import yaml

# Add the parent directory to path so we can import from src
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.training.ttm import finetune_ttm, load_config
# Import our thread-safe info_print (defined above) instead of the one from ttm


# Thread-local storage for function context
_function_context = threading.local()


def function_logger(func):
    """Decorator that prepends function name to all info_print calls within the function

    This is thread-safe and doesn't modify global state.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Set the current function name in thread-local storage
        old_function_name = getattr(_function_context, "function_name", None)
        _function_context.function_name = func.__name__

        try:
            result = func(*args, **kwargs)
            return result
        finally:
            # Restore previous function name (if any)
            if old_function_name is not None:
                _function_context.function_name = old_function_name
            else:
                if hasattr(_function_context, "function_name"):
                    delattr(_function_context, "function_name")

    return wrapper


@function_logger
def main():
    parser = argparse.ArgumentParser(
        description="TTM Fine-tuning with metrics collection"
    )
    parser.add_argument(
        "--config", type=str, required=False, help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        required=False,
        help="Directory to save run outputs and logs",
    )
    args = parser.parse_args()

    info_print("=====================================")
    info_print("=== TTM Runner Starting ===")

    # Load configuration from YAML file if provided
    if args.config:
        info_print(f"Loading configuration from: {args.config}")
        config = load_config(args.config)

        # Save config to run directory if provided
        if args.run_dir:
            run_dir = Path(args.run_dir)
            run_dir.mkdir(parents=True, exist_ok=True)
            config_copy_path = run_dir / "experiment_config.yaml"
            with open(config_copy_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
            info_print(f"Configuration saved to: {config_copy_path}")

        # Extract parameters from config
        model_config = config.get("model", {})
        data_config = config.get("data", {})
        training_config = config.get("training", {})

        info_print("=====================================")
        info_print("=== Calling finetune_ttm() ===")
        metrics = finetune_ttm(
            model_path=model_config.get(
                "path", "ibm-granite/granite-timeseries-ttm-r2"
            ),
            context_length=model_config.get("context_length", 512),
            forecast_length=model_config.get("forecast_length", 96),
            # Data Configuration
            data_source_name=data_config.get("source_name", "kaggle_brisT1D"),
            y_feature=data_config.get("y_feature", ["bg_mM"]),
            x_features=data_config.get(
                "x_features",
                [
                    "steps",
                    "cob",
                    "carb_availability",
                    "insulin_availability",
                    "iob",
                ],
            ),
            timestamp_column=data_config.get("timestamp_column", "datetime"),
            resolution_min=data_config.get("resolution_min", 5),
            data_split=tuple(data_config.get("data_split", [0.9, 0.1])),
            fewshot_percent=data_config.get("fewshot_percent", 100),
            dataloader_num_workers=training_config.get("dataloader_num_workers", 2),
            # Training Configuration
            batch_size=training_config.get("batch_size", 128),
            learning_rate=training_config.get("learning_rate", 0.001),
            num_epochs=training_config.get("num_epochs", 10),
            resume_dir=training_config.get("resume_dir"),
            use_cpu=training_config.get("use_cpu", False),
            loss=training_config.get("loss", "mse"),
        )
    else:
        # Fallback to default configuration if no YAML provided
        info_print("No configuration file provided, using default parameters")
        info_print("=====================================")
        info_print("=== Calling finetune_ttm() ===")
        metrics = finetune_ttm(
            model_path="ibm-granite/granite-timeseries-ttm-r2",
            context_length=512,
            forecast_length=96,
            # Data Configuration
            data_source_name="kaggle_brisT1D",
            y_feature=["bg_mM"],
            x_features=[
                "steps",
                "cob",
                "carb_availability",
                "insulin_availability",
                "iob",
            ],
            timestamp_column="datetime",
            resolution_min=5,
            data_split=(0.9, 0.1),
            fewshot_percent=100,
            # Training Configuration
            batch_size=128,
            learning_rate=0.001,
            num_epochs=10,
            use_cpu=False,
            loss="mse",
        )

    # Save metrics to file for model registry
    if args.run_dir and metrics:
        metrics_file = Path(args.run_dir) / "training_metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
        info_print(f"Training metrics saved to: {metrics_file}")

        # Also output metrics to stdout for SLURM script parsing
        print(f"METRICS_JSON:{json.dumps(metrics)}")


if __name__ == "__main__":
    main()
