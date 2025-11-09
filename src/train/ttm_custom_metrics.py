# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)
# For commercial licensing, contact: [Add your contact information]

import glob
import os

import pandas as pd
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from transformers import Trainer, TrainerCallback, TrainingArguments
from transformers.integrations.integration_utils import INTEGRATION_TO_CALLBACK
from tsfm_public import (
    TimeSeriesPreprocessor,
    TrackingCallback,
    count_parameters,
    get_datasets,
)
from tsfm_public.toolkit.time_series_preprocessor import ScalerType
from tsfm_public.toolkit.get_model import get_model
from tsfm_public.toolkit.lr_finder import optimal_lr_finder
from tsfm_public.toolkit.time_series_preprocessor import DEFAULT_FREQUENCY_MAPPING

from src.tuning.benchmark import impute_missing_values
from src.utils.os_helper import get_project_root

CONTEXT_LENGTH = 512
PREDICTION_LENGTH = 96


class CustomMetricsCallback(TrainerCallback):
    """Custom callback to log additional metrics to trainer_state.json"""

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when logging occurs"""
        if logs is not None:
            # Add custom metrics to logs
            if "train_loss" in logs:
                logs["custom_metric_example"] = logs["train_loss"] * 0.5

            # Add timestamp
            import time

            logs["timestamp"] = time.time()

            # Add custom training info
            logs["custom_batch_size"] = args.per_device_train_batch_size
            logs["custom_learning_rate"] = args.learning_rate

            # Add patient info if available
            logs["data_source"] = "kaggle_brisT1D"
            logs["context_length"] = CONTEXT_LENGTH
            logs["prediction_length"] = PREDICTION_LENGTH

        return control  # Must return control, not logs!


def compute_custom_metrics(eval_pred):
    """Custom metrics function for evaluation"""
    import numpy as np
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    predictions, labels = eval_pred

    # Flatten arrays if needed
    if len(predictions.shape) > 2:
        predictions = predictions.reshape(-1, predictions.shape[-1])
        labels = labels.reshape(-1, labels.shape[-1])

    # Calculate custom metrics
    mse = mean_squared_error(labels, predictions)
    mae = mean_absolute_error(labels, predictions)
    rmse = np.sqrt(mse)

    # Calculate additional metrics
    mape = (
        np.mean(np.abs((labels - predictions) / np.where(labels != 0, labels, 1e-8)))
        * 100
    )

    # Calculate R² score
    ss_res = np.sum((labels - predictions) ** 2)
    ss_tot = np.sum((labels - np.mean(labels)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))

    return {
        "custom_mse": float(mse),
        "custom_mae": float(mae),
        "custom_rmse": float(rmse),
        "custom_mape": float(mape),
        "custom_r2": float(r2),
        "custom_num_predictions": int(len(predictions.flatten())),
    }


def load_processed_data_from_cache(data_source_name):
    """
    Load processed CSV files directly from cache directory and concatenate them.

    Args:
        data_source_name (str): Name of the data source (e.g., "kaggle_brisT1D")

    Returns:
        dict: Dictionary with patient_id as key and DataFrame as value
    """
    # Get the cache directory path
    root_dir = get_project_root()
    cache_dir = os.path.join(root_dir, "cache", "data", data_source_name, "processed")

    # Find all CSV files in the processed directory
    csv_files = glob.glob(os.path.join(cache_dir, "*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No processed CSV files found in {cache_dir}")

    print(f"Found {len(csv_files)} processed CSV files in {cache_dir}")

    data_dict = {}

    for csv_file in csv_files:
        # Extract patient ID from filename (assuming format like "patient_123.csv")
        filename = os.path.basename(csv_file)
        patient_id = filename.replace(".csv", "")

        # Load the CSV file
        try:
            df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
            data_dict[patient_id] = df
            print(f"Loaded {patient_id}: {len(df)} rows")
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
            continue

    print(f"Successfully loaded {len(data_dict)} patients")
    return data_dict


def reduce_features_multi_patient(patients_dict, resolution_min, x_features, y_feature):
    """
    1. Select patients with the correct resolution
    2. Remove all unnecessary columns
    3. Impute missing values
    4. Add patient id
    5. Concatenate all patients
    """
    processed_patients = []

    for patient_id, df in patients_dict.items():
        # Check if patient has the correct interval
        if (df.index[1] - df.index[0]).components.minutes == resolution_min:
            print(f"Processing patient {patient_id}...")
            # Process each patient individually
            p_df = df.iloc[:]
            p_df = p_df[x_features + y_feature]
            # Impute missing values for this patient
            p_df = impute_missing_values(p_df, columns=x_features)
            p_df = impute_missing_values(p_df, columns=y_feature)
            p_df["id"] = patient_id
            processed_patients.append(p_df)

    return pd.concat(processed_patients)


def _get_finetune_trainer(
    dataset_name,
    model_path,
    batch_size,
    learning_rate=None,
    context_length=CONTEXT_LENGTH,
    forecast_length=PREDICTION_LENGTH,
    fewshot_percent=5,  # If resume from a checkpoint, this will need to be the same as the previous run
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
    print("-" * 20, f"Running few-shot {fewshot_percent}%", "-" * 20, "\n")

    tsp = TimeSeriesPreprocessor(
        **column_specifiers,
        context_length=context_length,
        prediction_length=forecast_length,
        scaling=True,
        encode_categorical=False,
        scaler_type=ScalerType.STANDARD,
    )

    finetune_forecast_model = get_model(
        model_path,
        context_length=context_length,
        prediction_length=forecast_length,
        freq_prefix_tuning=False,
        freq=DEFAULT_FREQUENCY_MAPPING[f"{resolution_min}min"],
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
        print("OPTIMAL SUGGESTED LEARNING RATE =", learning_rate)
    print(f"Using learning rate = {learning_rate}")

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
        eval_steps=500,  # Evaluate every 500 steps
        fp16=True,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        dataloader_num_workers=8,
        report_to="none",
        save_strategy="steps",
        logging_strategy="steps",
        logging_steps=100,  # Log every 100 steps
        logging_first_step=True,  # Log the first step
        save_steps=500,  # Save checkpoints every 500 steps
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
        # Additional custom logging
        include_inputs_for_metrics=True,  # Include inputs in compute_metrics
    )

    print(f"Training for {num_epochs} epochs")

    # Create the early stopping callback
    # early_stopping_callback = EarlyStoppingCallback(
    #     early_stopping_patience=10,  # Number of epochs with no improvement after which to stop
    #     early_stopping_threshold=1e-5,  # Minimum improvement required to consider as improvement
    # )
    tracking_callback = TrackingCallback()
    custom_metrics_callback = CustomMetricsCallback()

    # Optimizer and scheduler
    optimizer = AdamW(finetune_forecast_model.parameters(), lr=learning_rate)
    scheduler = ExponentialLR(optimizer, gamma=0.9999)

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

    return finetune_forecast_trainer, dset_test


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
):
    """
    Fine-tunes a Time Series Transformer Model (TTM) on patient time series data.

    This version includes custom metrics logging to trainer_state.json.

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
    print("-" * 20, "Preparing data...", "-" * 20, "\n")
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

    # Train/val/test split
    train, test, val = data_split
    split_config = {"train": train, "test": test}
    data_length = len(data)
    print(f"Data length: {data_length}")
    print(f"Split config: Train {train * 100}%, Test {test * 100}%")
    print(f"Data ids: {data['id'].unique()}")
    print("Data processing complete")

    #### Prepare model ####
    print("-" * 20, "Preparing model...", "-" * 20, "\n")
    print(f"Model path: {model_path}")

    # Where to save the checkpoint
    root_dir = get_project_root()
    model_dir = os.path.join(root_dir, "models", "ttm", data_source_name)
    # Timestamp the save_dir to differentiate between runs
    current_time = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
    new_run_saved_dir = os.path.join(model_dir, current_time)

    dataset_name = f"{resolution_min}min_patients"
    finetune_forecast_trainer, dset_test = _get_finetune_trainer(
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
        split_config=split_config,
        save_dir=new_run_saved_dir,
        resume_dir=resume_dir,
    )

    #### Fine-tune ####
    print("-" * 20, "Fine-tuning...", "-" * 20, "\n")
    if resume_dir is not None:
        print(f"Resuming training from {resume_dir}")
        finetune_forecast_trainer.train(resume_from_checkpoint=True)
    else:
        print(f"Starting new training run in output directory: {new_run_saved_dir}")
        finetune_forecast_trainer.train()

    #### Evaluate ####
    print("-" * 20, "Evaluating...", "-" * 20, "\n")
    print("+" * 20, f"Test MSE after few-shot {fewshot_percent}% fine-tuning", "+" * 20)
    finetune_forecast_trainer.model.loss = "mse"  # fixing metric to mse for evaluation

    fewshot_output = finetune_forecast_trainer.evaluate(dset_test)
    print(fewshot_output)
    print("+" * 60)


# TODO: Try out Pinball loss too
# TODO: Pull get_datasets out of the function and check the shape of the data
# TODO: Figure out all the params we need like freeze_backbone, optimizer, scheduler, etc.
# TODO: Add back early stopping
# TODO: Add utilities to load from a checkpoint (like the notebook)
if __name__ == "__main__":
    finetune_ttm(
        model_path="ibm-granite/granite-timeseries-ttm-r2",
        context_length=512,
        forecast_length=96,
        # Data Configuration
        data_source_name="kaggle_brisT1D",
        y_feature=["bg_mM"],
        x_features=["steps", "cob", "carb_availability", "insulin_availability", "iob"],
        timestamp_column="datetime",
        resolution_min=5,
        data_split=(0.85, 0.1, 0.05),
        fewshot_percent=100,
        # Training Configuration
        batch_size=128,
        learning_rate=0.001,
        num_epochs=500,  # Increase if needed
        # resume_dir="models/ttm/kaggle_brisT1D/2025-09-30_03-19-14/5min_patients/output",
        use_cpu=False,
        loss="mse",
    )
