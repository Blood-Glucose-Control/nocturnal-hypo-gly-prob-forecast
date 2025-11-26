# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)
# For commercial licensing, contact: [Add your contact information]

import argparse
import glob
import os
import sys
from pathlib import Path

import pandas as pd
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from transformers import Trainer, TrainerCallback, TrainingArguments
from transformers.integrations import INTEGRATION_TO_CALLBACK
from tsfm_public import (
    TimeSeriesPreprocessor,
    TrackingCallback,
    count_parameters,
    get_datasets,
)
from tsfm_public.toolkit.get_model import get_model
from tsfm_public.toolkit.lr_finder import optimal_lr_finder

from src.data.cache_manager import get_cache_manager
from src.data.preprocessing.split_or_combine_patients import reduce_features_multi_patient
from src.utils.os_helper import get_project_root
from src.utils.logging_helper import info_print, error_print, debug_print

CONTEXT_LENGTH = 512
PREDICTION_LENGTH = 96

# Debug configuration - set to False for production runs
# To enable debug mode, set environment variable: export TTM_DEBUG=true
DEBUG_MODE = os.getenv("TTM_DEBUG", "false").lower() == "true"

def load_config(config_path):
    """Load YAML configuration file"""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="TTM Fine-tuning with YAML configuration"
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
    return parser.parse_args()


class CustomMetricsCallback(TrainerCallback):
    """Custom callback to log additional metrics and track training progress"""

    def __init__(self):
        super().__init__()
        self.best_eval_loss = float("inf")
        self.final_train_loss = None
        self.final_eval_loss = None
        self.training_samples_per_second = None
        self.best_checkpoint = None
        self.training_complete = False

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when logging occurs"""
        if logs is not None:
            # Add timestamp
            import time

            logs["timestamp"] = time.time()

            # Add custom training info (but only add these, don't duplicate)
            logs["custom_batch_size"] = args.per_device_train_batch_size
            logs["custom_learning_rate"] = args.learning_rate

            # Track metrics for registry (but don't add extra custom metrics to avoid clutter)
            if "train_loss" in logs:
                self.final_train_loss = logs["train_loss"]

            if "eval_loss" in logs:
                self.final_eval_loss = logs["eval_loss"]
                if logs["eval_loss"] < self.best_eval_loss:
                    self.best_eval_loss = logs["eval_loss"]
                    self.best_checkpoint = f"checkpoint-{state.global_step}"

            if "train_samples_per_second" in logs:
                self.training_samples_per_second = logs["train_samples_per_second"]

        return control

    def on_train_end(self, args, state, control, **kwargs):
        """Called when training ends"""
        self.training_complete = True
        # Print to both stdout (to show up in slurm .out) and stderr (for immediate feedback)
        summary = f"\n{'=' * 60}\nTRAINING COMPLETED SUCCESSFULLY\n"
        summary += f"Final train loss: {self.final_train_loss}\n"
        summary += f"Final eval loss: {self.final_eval_loss}\n"
        summary += f"Best eval loss: {self.best_eval_loss}\n"
        summary += f"Best checkpoint: {self.best_checkpoint}\n"
        if self.training_samples_per_second:
            summary += f"Training samples/second: {self.training_samples_per_second}\n"
        summary += "=" * 60

        # Print to both stderr (immediate) and stdout (for slurm out file)
        print(summary, file=sys.stderr, flush=True)
        print(summary, file=sys.stdout, flush=True)

    def get_metrics_summary(self):
        """Get summary of tracked metrics for model registry"""
        return {
            "final_train_loss": self.final_train_loss,
            "final_eval_loss": self.final_eval_loss,
            "best_eval_loss": self.best_eval_loss,
            "best_checkpoint": self.best_checkpoint,
            "training_samples_per_second": self.training_samples_per_second,
        }


def compute_custom_metrics(eval_pred):
    """Custom metrics function for evaluation"""
    import numpy as np
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    # Debug: Check the structure of eval_pred
    debug_print(f"eval_pred type: {type(eval_pred)}")
    debug_print(
        f"eval_pred length: {len(eval_pred) if hasattr(eval_pred, '__len__') else 'N/A'}"
    )

    try:
        predictions, labels = eval_pred
        debug_print(
            f"predictions type: {type(predictions)}, shape: {getattr(predictions, 'shape', 'No shape attr')}"
        )
        debug_print(
            f"labels type: {type(labels)}, shape: {getattr(labels, 'shape', 'No shape attr')}"
        )
    except Exception as e:
        debug_print(f"Error unpacking eval_pred: {e}")
        return {"custom_error": "Failed to unpack eval_pred"}

    # Handle EvalPrediction object properly
    if hasattr(eval_pred, "predictions") and hasattr(eval_pred, "label_ids"):
        predictions = eval_pred.predictions
        labels = eval_pred.label_ids
        debug_print("Using .predictions and .label_ids attributes")

    # Handle nested structures (common with TTM)
    if isinstance(predictions, (tuple, list)):
        debug_print(f"predictions is tuple/list with length: {len(predictions)}")
        if len(predictions) > 0:
            debug_print(f"first prediction element type: {type(predictions[0])}")
            debug_print(
                f"first prediction element shape: {getattr(predictions[0], 'shape', 'No shape')}"
            )

        # For TTM, often the first element contains the actual predictions
        if len(predictions) > 0 and hasattr(predictions[0], "shape"):
            predictions = predictions[0]
        else:
            debug_print("Cannot extract predictions from tuple/list")
            return {"custom_error": "Cannot extract predictions from complex structure"}

    if isinstance(labels, (tuple, list)):
        debug_print(f"labels is tuple/list with length: {len(labels)}")
        if len(labels) > 0:
            debug_print(f"first label element type: {type(labels[0])}")
            debug_print(
                f"first label element shape: {getattr(labels[0], 'shape', 'No shape')}"
            )

        # For TTM, often the first element contains the actual labels
        if len(labels) > 0 and hasattr(labels[0], "shape"):
            labels = labels[0]
        else:
            debug_print("Cannot extract labels from tuple/list")
            return {"custom_error": "Cannot extract labels from complex structure"}

    # Convert to numpy arrays if needed (with better error handling)
    try:
        if not isinstance(predictions, np.ndarray):
            predictions = np.array(predictions)
        if not isinstance(labels, np.ndarray):
            labels = np.array(labels)

        debug_print(f"Final predictions shape: {predictions.shape}")
        debug_print(f"Final labels shape: {labels.shape}")
    except Exception as e:
        debug_print(f"Error converting to numpy arrays: {e}")
        return {"custom_error": f"Array conversion failed: {str(e)}"}

    # Handle shape mismatches and flatten if needed
    if predictions.shape != labels.shape:
        debug_print("Shape mismatch, attempting to flatten or reshape")
        debug_print(
            f"predictions shape: {predictions.shape}, labels shape: {labels.shape}"
        )

        # Try flattening both
        predictions_flat = predictions.flatten()
        labels_flat = labels.flatten()

        if len(predictions_flat) == len(labels_flat):
            predictions = predictions_flat
            labels = labels_flat
            debug_print("Successfully flattened both arrays")
        else:
            debug_print("Flattening didn't resolve shape mismatch")
            return {
                "custom_error": f"Shape mismatch: pred {predictions.shape} vs labels {labels.shape}"
            }

    try:
        # Calculate custom metrics
        mse = mean_squared_error(labels, predictions)
        mae = mean_absolute_error(labels, predictions)
        rmse = np.sqrt(mse)

        # Calculate MAPE safely
        mape = 0
        if np.any(labels != 0):
            mape = (
                np.mean(
                    np.abs((labels - predictions) / np.where(labels != 0, labels, 1))
                )
                * 100
            )

        return {
            "custom_mse": float(mse),
            "custom_mae": float(mae),
            "custom_rmse": float(rmse),
            "custom_mape": float(mape),
        }
    except Exception as e:
        debug_print(f"Error computing metrics: {e}")
        import traceback

        debug_print(f"Full traceback: {traceback.format_exc()}")
        return {"custom_error": str(e)}


def load_processed_data_from_cache(data_source_name):
    """
    Load processed CSV files directly from cache directory and concatenate them.

    Args:
        data_source_name (str): Name of the data source (e.g., "kaggle_brisT1D")

    Returns:
        dict: Dictionary with patient_id as key and DataFrame as value
    """
    # Get the cache directory path using CacheManager
    cache_manager = get_cache_manager()
    cache_dir = cache_manager.get_absolute_path_by_type(data_source_name, "processed")

    # Find all CSV files in the processed directory
    csv_files = glob.glob(os.path.join(cache_dir, "*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No processed CSV files found in {cache_dir}")

    info_print(f"Found {len(csv_files)} processed CSV files in {cache_dir}")

    data_dict = {}

    for csv_file in csv_files:
        # Extract patient ID from filename (assuming format like "patient_123.csv")
        filename = os.path.basename(csv_file)
        patient_id = filename.replace(".csv", "")

        # Load the CSV file
        try:
            df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
            data_dict[patient_id] = df
            info_print(f"Loaded {patient_id}: {len(df)} rows")
        except Exception as e:
            info_print(f"Error loading {csv_file}: {e}")
            continue

    info_print(f"Successfully loaded {len(data_dict)} patients")
    return data_dict

def _get_finetune_trainer(
    dataset_name,
    model_path,
    batch_size,
    learning_rate=None,
    context_length=CONTEXT_LENGTH,
    forecast_length=PREDICTION_LENGTH,
    fewshot_percent=5,  # If resume from a checkpoint, this will need to be the same as the previous run
    dataloader_num_workers=2,
    freeze_backbone=True,
    num_epochs=50,
    loss="mse",
    quantile=0.5,
    use_cpu=False,
    column_specifiers={},
    resolution_min=5,
    data=None,
    split_config=None,
    save_dir=None,
    resume_dir=None,
):
    """
    Internal function to get the finetune trainer. resume_dir will override save_dir if provided.
    """
    info_print("-" * 20, f"Running few-shot {fewshot_percent}%", "-" * 20, "\n")

    tsp = TimeSeriesPreprocessor(
        **column_specifiers,
        context_length=context_length,
        prediction_length=forecast_length,
        scaling=True,
        encode_categorical=False,
        scaler_type="standard",
    )

    finetune_forecast_model = get_model(
        model_path,
        context_length=context_length,
        prediction_length=forecast_length,
        freq_prefix_tuning=False,
        freq=f"{resolution_min}min",
        prefer_l1_loss=False,
        prefer_longer_context=True,
        # Can also provide TTM Config args. A param?
        loss=loss,
        quantile=quantile,
    )

    dset_train, dset_val, dset_test = get_datasets(
        tsp,
        data,
        split_config,
        fewshot_fraction=fewshot_percent / 100,
        fewshot_location="last",  # Take the last x percent of the training data
        use_frequency_token=finetune_forecast_model.config.resolution_prefix_tuning,
    )

    if freeze_backbone:
        print(
            "Number of params before freezing backbone",
            count_parameters(finetune_forecast_model),
        )

        # Freeze the backbone of the model
        for param in finetune_forecast_model.backbone.parameters():
            param.requires_grad = False

        # Count params
        print(
            "Number of params after freezing the backbone",
            count_parameters(finetune_forecast_model),
        )

    # Find optimal learning rate
    # Use with caution: Set it manually if the suggested learning rate is not suitable
    if learning_rate is None:
        learning_rate, finetune_forecast_model = optimal_lr_finder(
            finetune_forecast_model,
            dset_train,
            batch_size=batch_size,
        )
        info_print("OPTIMAL SUGGESTED LEARNING RATE =", learning_rate)
    info_print(f"Using learning rate = {learning_rate}")

    # Whether to start a new training run or resume from a checkpoint
    out_dir = os.path.join(save_dir, dataset_name)
    output_dir = os.path.join(out_dir, "output")
    if resume_dir is not None:
        output_dir = resume_dir

    finetune_forecast_args = TrainingArguments(
        output_dir=output_dir,  # If output_dir does exists, it will resume training from the lastest checkpoint with train(resume_from_checkpoint=True)
        overwrite_output_dir=False,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        do_eval=True,
        eval_strategy="steps",
        eval_steps=1000,  # Evaluate every 1000 steps (less frequent = faster training)
        fp16=True,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        dataloader_num_workers=2,
        report_to="none",
        save_strategy="steps",
        logging_strategy="steps",
        logging_steps=100,  # Log every 100 steps
        logging_first_step=True,  # Log the first step
        save_steps=2000,  # Save checkpoints every 2000 steps
        save_total_limit=100,
        logging_dir=os.path.join(
            out_dir, "logs"
        ),  # Make sure to specify a logging directory
        load_best_model_at_end=True,  # Load the best model when training ends
        metric_for_best_model="eval_loss",  # Metric to monitor for early stopping
        greater_is_better=False,  # For loss
        use_cpu=use_cpu,
        # Additional logging control
        log_level="info",  # Control log verbosity
        disable_tqdm=False,  # Keep progress bars
    )

    info_print(f"Training for {num_epochs} epochs")

    # Create the early stopping callback
    # early_stopping_callback = EarlyStoppingCallback(
    #     early_stopping_patience=10,  # Number of epochs with no improvement after which to stop
    #     early_stopping_threshold=1e-5,  # Minimum improvement required to consider as improvement
    # )
    tracking_callback = TrackingCallback()
    custom_metrics_callback = CustomMetricsCallback()

    # Optimizer and scheduler
    optimizer = AdamW(finetune_forecast_model.parameters(), lr=learning_rate)
    scheduler = ExponentialLR(optimizer, gamma=0.9999)  # TODO: Plot the learning rate.

    finetune_forecast_trainer = Trainer(
        model=finetune_forecast_model,
        args=finetune_forecast_args,
        train_dataset=dset_train,
        eval_dataset=dset_val,
        compute_metrics=compute_custom_metrics,  # Add custom metrics computation
        callbacks=[
            # TODO:TONY - Remove early stopping for now
            # early_stopping_callback,
            tracking_callback,
            custom_metrics_callback,  # Add custom metrics callback
        ],
        optimizers=(optimizer, scheduler),
    )
    finetune_forecast_trainer.remove_callback(INTEGRATION_TO_CALLBACK["codecarbon"])

    return finetune_forecast_trainer, dset_test, custom_metrics_callback


def finetune_ttm(
    # Model Configuration
    model_path,
    context_length,
    forecast_length,
    # Data Configuration
    data_source_name,
    y_feature,
    x_features,
    timestamp_column,
    resolution_min,
    data_split,
    fewshot_percent,
    # Training Configuration
    batch_size,
    learning_rate,
    num_epochs,
    resume_dir=None,
    loss="mse",
    use_cpu=False,
    dataloader_num_workers=2,
):
    """
    Fine-tunes a Time Series Transformer Model (TTM) on patient time series data.

    Args:
        # Model Configuration
        model_path (str):
            Path or identifier of the pre-trained model to fine-tune. This can be either:
            - The name of a Huggingface repository (e.g., "ibm-granite/granite-timeseries-ttm-r2")
            - A local path to a folder containing model files in a format supported by transformers.
        context_length (int):
            Number of time steps in the model's input context window.
            This can be 512, 1024 or 1536.
        forecast_length (int):
            Number of time steps to forecast.
            This can be 96, 192, 336 or 720.

        # Data Configuration
        data_source_name (str):
            Name of the data source to load patient data from. Currently supported:
            - "kaggle_brisT1D"
        y_feature (list of str):
            List of target variable(s) to predict.
        x_features (list of str):
            List of input/control/exogenous features.
            Categorical features are NOT yet supported.
        timestamp_column (str):
            Name of the column containing timestamps.
        resolution_min (int):
            Time resolution of the data in minutes.
        data_split (tuple):
            Tuple specifying train and test split proportions (e.g., (0.7, 0.2)).
        fewshot_percent (int):
            Percentage of data to use for few-shot learning (100 for full-shot).

        # Training Configuration
        batch_size (int):
            Batch size for training.
        learning_rate (float):
            Max learning rate for optimizer.
        num_epochs (int):
            Number of training epochs.
        resume_dir (str):
            ./output Directory to resume training from the lastest checkpoint.
            If not provided, a new training run will be started.
            Increase epochs to continue training from the lastest checkpoint.
            To resume training, make sure:
                1. fewshot_percent is the same as the previous run
                2. resolution_min is the same as the previous run
                3. context_length is the same as the previous run
                4. forecast_length is the same as the previous run
                5. batch_size is the same as the previous run
    Returns:
        None

    This function prepares the data, splits it into train/validation/test sets, and fine-tunes
    a transformer-based time series model on the specified patient dataset.
    """

    #### Prepare data ####
    info_print("-" * 20, "Preparing data...", "-" * 20, "\n")
    # loader = get_loader(
    #     data_source_name=data_source_name,
    #     # NOTE: The val split here is done via ttm preprocessing so ignore this param
    #     num_validation_days=20,
    #     use_cached=True,
    # )
    # data_dict = loader.processed_data
    column_specifiers = {
        "timestamp_column": timestamp_column,
        "id_columns": ["id"],
        "target_columns": y_feature,
        "control_columns": x_features,
    }

    data_dict = load_processed_data_from_cache(data_source_name)

    all_patients_df = reduce_features_multi_patient(
        data_dict, resolution_min, x_features, y_feature
    )
    # datetime has to be a column
    data = all_patients_df.reset_index()

    # Train/val split
    train, test = data_split
    split_config = {"train": train, "test": test}
    data_length = len(data)
    info_print(f"Data length: {data_length}")
    info_print(f"Split config: Train {train * 100}%, Test {test * 100}%")
    info_print(f"Data ids: {data['id'].unique()}")
    info_print("Data processing complete")

    #### Prepare model ####
    info_print("-" * 20, "Preparing model...", "-" * 20, "\n")
    info_print(f"Model path: {model_path}")

    # Where to save the checkpoint
    root_dir = get_project_root()
    model_dir = os.path.join(root_dir, "models", "ttm", data_source_name)
    # Timestamp the save_dir to differentiate between runs
    current_time = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
    new_run_saved_dir = os.path.join(model_dir, current_time)

    dataset_name = f"{resolution_min}min_patients"
    finetune_forecast_trainer, dset_test, custom_metrics_callback = (
        _get_finetune_trainer(
            model_path=model_path,
            dataset_name=dataset_name,
            num_epochs=num_epochs,
            context_length=context_length,
            forecast_length=forecast_length,
            batch_size=batch_size,
            fewshot_percent=fewshot_percent,
            learning_rate=learning_rate,
            loss=loss,
            use_cpu=use_cpu,
            column_specifiers=column_specifiers,
            resolution_min=resolution_min,
            data=data,
            dataloader_num_workers=dataloader_num_workers,
            split_config=split_config,
            save_dir=new_run_saved_dir,
            resume_dir=resume_dir,
        )
    )

    #### Fine-tune ####
    info_print("-" * 20, "Fine-tuning...", "-" * 20, "\n")
    if resume_dir is not None:
        info_print(f"Resuming training from {resume_dir}")
        finetune_forecast_trainer.train(resume_from_checkpoint=True)
    else:
        info_print(
            f"Starting new training run in output directory: {new_run_saved_dir}"
        )
        finetune_forecast_trainer.train()

    #### Evaluate ####
    info_print("-" * 20, "Evaluating...", "-" * 20, "\n")
    info_print(
        "+" * 20, f"Test MSE after few-shot {fewshot_percent}% fine-tuning", "+" * 20
    )
    finetune_forecast_trainer.model.loss = "mse"  # fixing metric to mse for evaluation

    fewshot_output = finetune_forecast_trainer.evaluate(dset_test)
    print(fewshot_output)
    print("+" * 60)

    # Return metrics for model registry
    return custom_metrics_callback.get_metrics_summary()


# TODO: Try out Pinball loss too
# TODO: Pull get_datasets out of the function and check the shape of the data
# TODO: Figure out all the params we need like freeze_backbone, optimizer, scheduler, etc.
# TODO: Add back early stopping
# TODO: Add utilities to load from a checkpoint (like the notebook)
if __name__ == "__main__":
    args = parse_arguments()

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

        finetune_ttm(
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
                ["steps", "cob", "carb_availability", "insulin_availability", "iob"],
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
        finetune_ttm(
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
