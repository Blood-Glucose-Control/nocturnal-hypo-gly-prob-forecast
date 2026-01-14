"""
TTM (TinyTimeMixer) model implementation using the base TSFM framework.

This module provides a concrete implementation of TTM that inherits from
the base TSFM framework, demonstrating how to integrate existing models.
"""

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import (
    TrainingArguments,
    Trainer,
)

# Import your existing TTM-related modules
from tsfm_public import (
    TimeSeriesPreprocessor,
    get_datasets,
)
from tsfm_public.toolkit.get_model import get_model
from tsfm_public.toolkit.time_series_preprocessor import ScalerType

# Local imports
from src.models.base import BaseTimeSeriesFoundationModel, TrainingStrategy
from src.models.ttm.config import TTMConfig
from src.data.diabetes_datasets.data_loader import get_loader
from src.data.models import ColumnNames
from src.data.preprocessing.split_or_combine_patients import (
    reduce_features_multi_patient,
)
from src.utils.logging_helper import info_print, debug_print, error_print


class TTMForecaster(BaseTimeSeriesFoundationModel):
    """TTM (TinyTimeMixer) forecaster implementation using the base TSFM framework.

    TinyTimeMixer is an MLP-based time series foundation model that uses mixing
    layers instead of attention mechanisms. This class integrates TTM with the
    unified base framework while preserving all existing functionality.

    Attributes:
        config: TTM-specific configuration (TTMConfig instance).
        preprocessor: TimeSeriesPreprocessor for data normalization and windowing.
        column_specifiers: Dictionary mapping data columns to their roles
            (id, timestamp, target, control, etc.).

    Note:
        TTM does NOT support LoRA fine-tuning as it lacks transformer attention
        layers. The supports_lora() method returns False.

    Example:
        >>> config = TTMConfig(model_path="ibm-granite/granite-timeseries-ttm-r2")
        >>> model = TTMForecaster(config)
        >>> model.fit(train_data="kaggle_brist1d")
        >>> predictions = model.predict(test_data)
    """

    def __init__(self, config: TTMConfig, lora_config=None, distributed_config=None):
        """Initialize the TTM forecaster.

        Args:
            config: TTM configuration object. If a non-TTMConfig is passed,
                it will be converted using essential parameters.
            lora_config: LoRA configuration (ignored for TTM as it doesn't
                support LoRA fine-tuning).
            distributed_config: Configuration for distributed training
                (DDP, DeepSpeed, or FSDP).

        Note:
            The model is initialized during construction via _initialize_model().
            The preprocessor and column_specifiers are initialized lazily during
            the first call to _prepare_data().
        """
        # Use the config as-is if it's already a TTMConfig
        # Only convert if we receive a different type
        if not isinstance(config, TTMConfig):
            # Create a basic TTMConfig from the essential parameters
            essential_params = {
                "model_path": getattr(config, "model_path", None),
                "context_length": getattr(config, "context_length", 512),
                "forecast_length": getattr(config, "forecast_length", 96),
                "learning_rate": getattr(config, "learning_rate", 1e-4),
                "batch_size": getattr(config, "batch_size", 64),
                "num_epochs": getattr(config, "num_epochs", 10),
            }
            config = TTMConfig(**essential_params)

        super().__init__(config, lora_config, distributed_config)

        # Type annotation to help linter understand config type
        self.config: TTMConfig = self.config

        # TTM-specific attributes
        self.preprocessor = None
        self.column_specifiers = None

    # Abstract method implementations
    ## Abstract implemented public methods
    def predict(
        self, data: Any, batch_size: Optional[int] = None, return_dict: bool = False
    ) -> Union[np.ndarray, Dict[str, Any]]:
        """
        Make predictions on new data.

        Args:
            data: Input data for prediction
            batch_size: Batch size for prediction (defaults to config.batch_size)
            return_dict: Whether to return additional information

        Returns:
            Predictions as numpy array or dictionary with additional info
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before making predictions")

        # Prepare data for prediction
        data_loader, _, _ = self._prepare_data(data)

        # Set model to evaluation mode
        self.model.eval()

        predictions = []
        with torch.no_grad():
            for batch in data_loader:
                # Move batch to appropriate device
                if torch.cuda.is_available() and not self.config.use_cpu:
                    batch = {
                        k: v.cuda()
                        for k, v in batch.items()
                        if isinstance(v, torch.Tensor)
                    }

                # Forward pass
                outputs = self.model(**batch)

                # Extract predictions (implementation depends on model)
                if hasattr(outputs, "prediction_logits"):
                    batch_predictions = outputs.prediction_logits
                elif hasattr(outputs, "logits"):
                    batch_predictions = outputs.logits
                else:
                    batch_predictions = outputs

                predictions.append(batch_predictions.cpu().numpy())

        # Concatenate all predictions
        predictions = np.concatenate(predictions, axis=0)

        if return_dict:
            return {
                "predictions": predictions,
                "model_config": self.config.to_dict(),
                "n_samples": len(predictions),
            }

        return predictions

    def training_backend(self) -> TrainingStrategy:
        """Return the training strategy used by TTM.

        Returns:
            TrainingStrategy: Always returns TrainingStrategy.TRANSFORMERS as
                TTM uses the HuggingFace Transformers Trainer for training.
        """
        return TrainingStrategy.TRANSFORMERS

    def supports_lora(self) -> bool:
        """Check if TTM supports LoRA fine-tuning.

        Returns:
            bool: Always returns False. TTM is an MLP-based (Mixer) architecture
                that lacks transformer attention layers required for LoRA.
        """
        return False

    ## Abstract implemented private methods
    def _initialize_model(self) -> None:
        """Initialize the TTM model architecture.

        Loads the pre-trained TTM model from the configured model_path and
        configures parameter gradients based on the training_mode:
        - 'zero_shot': All parameters frozen (no training)
        - 'fine_tune' or 'from_scratch': All parameters trainable

        Raises:
            Exception: If model initialization fails (e.g., invalid model_path,
                network issues, or incompatible configuration).
        """
        try:
            info_print(f"Initializing TTM model from {self.config.model_path}")

            # Prepare minimal parameters for TTM model initialization
            model_params = {
                "model_path": self.config.model_path,
                "context_length": self.config.context_length,
                "prediction_length": self.config.forecast_length,
                "freq": f"{self.config.resolution_min}min",
            }

            # Only add prediction_filter_length if it's not None
            if self.config.prediction_filter_length is not None:
                model_params["prediction_filter_length"] = (
                    self.config.prediction_filter_length
                )

            # Get TTM model using the existing tsfm_public toolkit
            ttm_model = get_model(**model_params)

            # Configure parameter gradients based on training strategy
            if self.config.training_mode == "zero_shot":
                info_print("Freezing all parameters for zero-shot evaluation")
                for param in ttm_model.parameters():
                    param.requires_grad = False
            else:
                # For any training scenario (fine_tune, from_scratch), enable gradients
                info_print(
                    f"Enabling gradients for all parameters ({self.config.training_mode} mode)"
                )
                for param in ttm_model.parameters():
                    param.requires_grad = True

            self.model = ttm_model
            info_print("TTM model initialized successfully")

        except Exception as e:
            error_print(f"Failed to initialize TTM model: {str(e)}")
            raise

    def _prepare_data(
        self,
        train_data: Any,
        val_data: Optional[Any] = None,
        test_data: Optional[Any] = None,
    ) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
        """Prepare data loaders for TTM training, validation, and testing.

        Handles multiple input formats and integrates with the existing data
        loading pipeline. Creates TimeSeriesPreprocessor for normalization
        and windowing on first call.

        Args:
            train_data: Training data in one of the following formats:
                - str: Data source name (e.g., "kaggle_brist1d") to load via get_loader
                - pd.DataFrame: Pre-loaded DataFrame with time series data
                - dict: Multi-patient dictionary to be converted to DataFrame
            val_data: Validation data (currently unused, split from train_data).
            test_data: Test data (currently unused, split from train_data).

        Returns:
            Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
                - train_loader: DataLoader for training data
                - val_loader: DataLoader for validation data (or None)
                - test_loader: DataLoader for test data (or None)

        Raises:
            ValueError: If train_data is not a supported type.
            Exception: If data preprocessing or dataset creation fails.
        """
        info_print("Preparing data for TTM training...")

        # If train_data is a string, assume it's a data source name
        if isinstance(train_data, str):
            data_source_name = train_data

            # Load data using your existing loader
            loader = get_loader(
                data_source_name=data_source_name,
                num_validation_days=20,  # Adjust as needed
                use_cached=True,
            )  # type: ignore
            data = loader.processed_data
            debug_print(
                f"Loaded data from source '{data_source_name}' with data.head:\n{data[ next(iter(data))].head()}"
            )
        elif isinstance(train_data, pd.DataFrame):
            data = train_data
        else:
            raise ValueError(f"Unsupported data type: {type(train_data)}")

        # Check if dataset conversion is needed
        if isinstance(data, dict):
            data = reduce_features_multi_patient(
                patients_dict=data,
                resolution_min=self.config.resolution_min,
                x_features=self.config.input_features,
                y_feature=self.config.target_features,
            )
            info_print(
                f"Converted multi-patient dict to DataFrame \n dataset now how has the following columns available: {data.columns}"
            )
            data = data.reset_index()

        # Set up column specifiers (adapt to your data structure)
        if self.column_specifiers is None:
            self.column_specifiers = self._create_column_specifiers(data)

        # Create preprocessor
        if self.preprocessor is None:
            self.preprocessor = TimeSeriesPreprocessor(
                **self.column_specifiers,
                context_length=self.config.context_length,
                prediction_length=self.config.forecast_length,
                scaling=True,
                encoder_categorical=False,
                scaler_type=ScalerType.STANDARD.value,
            )

        # Create datasets using your existing function
        try:
            dset_train, dset_val, dset_test = get_datasets(
                ts_preprocessor=self.preprocessor,
                dataset=data,
                split_config=self.config.split_config,
                fewshot_fraction=self.config.fewshot_percent / 100,
                fewshot_location="last",  # Take the last x percent of the training data
            )

            # Create data loaders
            train_loader = DataLoader(
                dset_train,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=self.config.dataloader_num_workers,
            )

            val_loader = None
            if dset_val is not None:
                val_loader = DataLoader(
                    dset_val,
                    batch_size=self.config.batch_size,
                    shuffle=False,
                    num_workers=self.config.dataloader_num_workers,
                )

            test_loader = None
            if dset_test is not None:
                test_loader = DataLoader(
                    dset_test,
                    batch_size=self.config.batch_size,
                    shuffle=False,
                    num_workers=self.config.dataloader_num_workers,
                )

            info_print("Data preparation complete:")
            info_print(f"  Train samples: {len(dset_train):,} if dset_train else 0")
            info_print(f"  Val samples: {len(dset_val):,} if dset_val else 0")
            info_print(f"  Test samples: {len(dset_test):,} if dset_test else 0")

            return train_loader, val_loader, test_loader

        except Exception as e:
            error_print(f"Failed to prepare data: {str(e)}")
            raise

    def _save_checkpoint(self, output_dir: str) -> None:
        """Save TTM model checkpoint using HuggingFace format.

        Args:
            output_dir: Directory path where model checkpoint will be saved.

        Note:
            Uses save_pretrained() for HuggingFace-compatible format that
            can be loaded with AutoModel.from_pretrained().
        """
        if self.model is not None:
            self.model.save_pretrained(output_dir)
            info_print(f"TTM model saved to {output_dir}")

    def _load_checkpoint(self, model_dir: str) -> None:
        """Load TTM model checkpoint from a directory.

        Args:
            model_dir: Directory containing saved model checkpoint in
                HuggingFace format.

        Raises:
            Exception: If loading fails (e.g., corrupted files, incompatible
                model version, or missing files).
        """
        try:
            # TTM models can be loaded using the transformers library
            from transformers import AutoModel

            self.model = AutoModel.from_pretrained(model_dir)
            info_print(f"TTM model checkpoint loaded from {model_dir}")

        except Exception as e:
            error_print(f"Failed to load model checkpoint: {str(e)}")
            raise

    def _train_model(
        self,
        train_data: Any,
        val_data: Optional[Any] = None,
        test_data: Optional[Any] = None,
        output_dir: str = "./output",
        **kwargs,
    ) -> Dict[str, Any]:
        """Execute TTM training using the HuggingFace Trainer.

        Implements the model-specific training loop using Transformers Trainer
        with configured callbacks, metrics, and distributed training support.

        Args:
            train_data: Training data (see _prepare_data for supported formats).
            val_data: Validation data (currently unused, split from train_data).
            test_data: Test data (currently unused, split from train_data).
            output_dir: Directory for saving model checkpoints and logs.
            **kwargs: Additional arguments:
                - resume_from_checkpoint: Path to checkpoint to resume from.

        Returns:
            Dict[str, Any]: Dictionary containing:
                - train_metrics: Metrics from training (loss, runtime, etc.)
                - test_metrics: Metrics from test evaluation (if test data provided)
        """
        # Configure tqdm to update less frequently (every 30 seconds instead of constantly)
        # This reduces log file bloat while still showing progress
        import os

        os.environ["TQDM_MININTERVAL"] = "30"  # Update progress bar every 30 seconds

        # Prepare data loaders
        train_loader, val_loader, test_loader = self._prepare_data(
            train_data, val_data, test_data
        )

        # Create training arguments
        training_args = self._create_training_arguments(output_dir)

        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_loader.dataset if train_loader else None,
            eval_dataset=val_loader.dataset if val_loader else None,
            compute_metrics=self._compute_trainer_metrics,
            callbacks=self._get_callbacks(),
        )

        # Train the model
        resume_from_checkpoint = kwargs.get("resume_from_checkpoint", None)
        if resume_from_checkpoint:
            info_print(f"Resuming training from {resume_from_checkpoint}")
            train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        else:
            train_result = trainer.train()

        # Save the model
        trainer.save_model()

        # Evaluate on test set if provided
        test_metrics = {}
        if test_loader is not None:
            test_metrics = trainer.evaluate(eval_dataset=test_loader.dataset)
            info_print(f"Test metrics: {test_metrics}")

        return {
            "train_metrics": train_result.metrics,
            "test_metrics": test_metrics,
        }

    # TTM-specific public methods
    def predict_zero_shot(
        self,
        data: Any,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """Make zero-shot predictions without fine-tuning.

        Temporarily sets the fit strategy to 'zero_shot' and generates
        predictions using the pre-trained model weights.

        Args:
            data: Input data for prediction (see _prepare_data for formats).
            batch_size: Batch size for prediction. If None, uses config default.

        Returns:
            np.ndarray: Model predictions with shape (n_samples, forecast_length).

        Note:
            This method temporarily overrides is_fitted check. The original
            training_mode is restored after prediction.
        """
        info_print("Making zero-shot predictions with TTM")

        # Temporarily override fit strategy
        original_strategy = self.config.training_mode
        self.config.training_mode = "zero_shot"

        try:
            predictions = self.predict(data, batch_size)
            return predictions
        finally:
            # Restore original strategy
            self.config.training_mode = original_strategy

    def get_ttm_specific_info(self) -> Dict[str, Any]:
        """Get TTM-specific model information.

        Extends the base get_model_info() with TTM-specific details.

        Returns:
            Dict[str, Any]: Dictionary containing base model info plus
                'ttm_specific' key with:
                - model_path: HuggingFace model path
                - context_length: Input sequence length
                - forecast_length: Prediction horizon
                - num_input_channels: Number of input features
                - num_output_channels: Number of output features
                - freeze_backbone: Whether backbone is frozen
                - scaler_type: Data normalization method
        """
        base_info = self.get_model_info()

        ttm_info = {
            "model_path": self.config.model_path,
            "context_length": self.config.context_length,
            "forecast_length": self.config.forecast_length,
            "num_input_channels": self.config.num_input_channels,
            "num_output_channels": self.config.num_output_channels,
            "freeze_backbone": self.config.freeze_backbone,
            "scaler_type": self.config.scaler_type,
        }

        base_info.update({"ttm_specific": ttm_info})
        return base_info

    # TTM-specific private methods
    def _create_column_specifiers(self, data: pd.DataFrame) -> Dict[str, List[str]]:
        """Create column specifiers based on available data columns.

        Maps data columns to the roles expected by TimeSeriesPreprocessor:
        id, timestamp, target, control, observable, conditional, and
        static categorical columns.

        Args:
            data: DataFrame containing the time series data.

        Returns:
            Dict[str, List[str]]: Dictionary mapping column roles to lists of
                column names. Only includes columns that exist in the data.

        Note:
            Default mappings use ColumnNames enum values. Modify this method
            if your data uses different column naming conventions.
        """
        # Default mappings - adapt these to your data structure
        column_specifiers = {
            "id_columns": [ColumnNames.P_NUM.value],
            "timestamp_column": ColumnNames.DATETIME.value,
            "target_columns": [ColumnNames.BG.value],
            "observable_columns": [],
            "control_columns": [
                ColumnNames.STEPS.value,
                ColumnNames.COB.value,
                ColumnNames.CARB_AVAILABILITY.value,
                ColumnNames.INSULIN_AVAILABILITY.value,
                ColumnNames.IOB.value,
            ],
            "conditional_columns": [],
            "static_categorical_columns": [],
        }

        # Filter to only include columns that exist in the data
        available_columns = set(data.columns)

        for key, columns in column_specifiers.items():
            if isinstance(columns, list):
                column_specifiers[key] = [
                    col for col in columns if col in available_columns
                ]

        return column_specifiers

    def _get_distributed_training_args(self) -> Dict[str, Any]:
        """Get distributed training arguments for HuggingFace TrainingArguments.

        Builds a dictionary of distributed training parameters based on the
        configured strategy (DDP, DeepSpeed, or FSDP).

        Returns:
            Dict[str, Any]: Dictionary of distributed training arguments to be
                merged into TrainingArguments. Returns empty dict if distributed
                training is not enabled.
        """
        if not self.distributed_config.enabled:
            return {}

        args = {}

        if self.distributed_config.strategy == "ddp":
            # For DDP, TrainingArguments automatically handles distributed training
            # when torch.distributed is initialized. We just need to ensure
            # proper configuration is passed.
            args["ddp_backend"] = self.distributed_config.backend

            # DDP performance optimizations
            args["ddp_find_unused_parameters"] = (
                self.distributed_config.find_unused_parameters
            )
            args["ddp_bucket_cap_mb"] = (
                25  # Default bucket size for gradient communication
            )

            # TrainingArguments will auto-detect distributed environment

        elif self.distributed_config.strategy == "deepspeed":
            args["deepspeed"] = self.distributed_config.deepspeed_config

        elif self.distributed_config.strategy == "fsdp":
            args.update(self.distributed_config.fsdp_config or {})

        return args

    def _compute_trainer_metrics(self, eval_pred) -> Dict[str, float]:
        """Compute evaluation metrics for the HuggingFace Trainer.

        This method is passed to Trainer's compute_metrics parameter and handles
        the EvalPrediction object format used by the Transformers library.

        Args:
            eval_pred: EvalPrediction object from HuggingFace Trainer containing:
                - predictions: Model predictions (may be nested)
                - label_ids: Ground truth labels (may be nested)

        Returns:
            Dict[str, float]: Dictionary containing computed metrics:
                - mse: Mean Squared Error
                - rmse: Root Mean Squared Error
                - mae: Mean Absolute Error
                - mape: Mean Absolute Percentage Error

        Note:
            This is separate from BaseTimeSeriesFoundationModel._compute_metrics() because the
            Trainer passes EvalPrediction objects rather than raw arrays.
        """
        try:
            # Extract predictions and labels (handle TTM's output format)
            if hasattr(eval_pred, "predictions"):
                predictions = eval_pred.predictions
                labels = eval_pred.label_ids
            else:
                predictions, labels = eval_pred

            # Handle nested structures (common with TTM)
            if isinstance(predictions, (tuple, list)) and len(predictions) > 0:
                if hasattr(predictions[0], "shape"):
                    predictions = predictions[0]

            if isinstance(labels, (tuple, list)) and len(labels) > 0:
                if hasattr(labels[0], "shape"):
                    labels = labels[0]

            # Convert to numpy arrays
            if not isinstance(predictions, np.ndarray):
                predictions = np.array(predictions)
            if not isinstance(labels, np.ndarray):
                labels = np.array(labels)

            # Compute metrics
            mse = np.mean((predictions - labels) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(predictions - labels))

            # MAPE with handling for zero values
            mask = labels != 0
            mape = (
                np.mean(np.abs((predictions[mask] - labels[mask]) / labels[mask]) * 100)
                if np.any(mask)
                else 0.0
            )

            metrics = {
                "mse": float(mse),
                "rmse": float(rmse),
                "mae": float(mae),
                "mape": float(mape),
            }

            debug_print(f"Computed metrics: {metrics}")
            return metrics

        except Exception as e:
            error_print(f"Error computing metrics: {str(e)}")
            return {"custom_error": str(e)}

    def _get_callbacks(self) -> List:
        """Get training callbacks for the HuggingFace Trainer.

        Creates callbacks based on the model configuration, including
        early stopping if patience > 0.

        Returns:
            List[TrainerCallback]: List of callback instances to pass to Trainer.
        """

        callbacks = []

        # Early stopping only works if evaluation is enabled
        # Since we use eval_strategy="no" for speed, skip early stopping
        # if self.config.early_stopping_patience > 0:
        #     callbacks.append(
        #         EarlyStoppingCallback(
        #             early_stopping_patience=self.config.early_stopping_patience,
        #             early_stopping_threshold=0.0,
        #         )
        #     )

        return callbacks

    def _create_training_arguments(self, output_dir: str) -> TrainingArguments:
        """Create HuggingFace TrainingArguments for TTM training.

        Builds TrainingArguments from the model configuration, including
        distributed training settings if enabled.

        Args:
            output_dir: Directory path for saving checkpoints, logs, and
                the final trained model.

        Returns:
            TrainingArguments: Configured training arguments for the
                HuggingFace Trainer.
        """
        # Base training arguments
        base_args = {
            "output_dir": output_dir,
            "learning_rate": self.config.learning_rate,
            "num_train_epochs": self.config.num_epochs,
            "per_device_train_batch_size": self.config.batch_size,
            "per_device_eval_batch_size": self.config.batch_size,
            "warmup_steps": self.config.warmup_steps,
            "weight_decay": self.config.weight_decay,
            "logging_dir": self.config.logging_dir
            if self.config.logging_dir
            else os.path.join(output_dir, "logs"),
            "logging_steps": 1000,  # Log every 1000 steps (reduces log verbosity significantly)
            "eval_strategy": "no",  # Disable evaluation during training for speed
            "save_strategy": "epoch",  # Only save at end of epoch
            "eval_steps": None,
            "save_steps": self.config.save_steps,
            "metric_for_best_model": self.config.metric_for_best_model,
            "greater_is_better": self.config.greater_is_better,
            "load_best_model_at_end": False,  # Disabled since eval is off
            "fp16": self.config.fp16,
            "dataloader_num_workers": self.config.dataloader_num_workers,
            "use_cpu": self.config.use_cpu,
            "report_to": "none",  # Disable wandb/tensorboard by default
            "disable_tqdm": False,  # Keep progress bar enabled
            "logging_first_step": True,
            "logging_nan_inf_filter": False,
        }

        # Add distributed training arguments
        distributed_args = self._get_distributed_training_args()
        base_args.update(distributed_args)

        return TrainingArguments(**base_args)


# Factory function for creating TTM models
def create_ttm_model(
    model_path: str = "ibm-granite/granite-timeseries-ttm-r2",
    context_length: int = 512,
    forecast_length: int = 96,
    **kwargs,
) -> TTMForecaster:
    """
    Factory function to create a TTM model with sensible defaults.

    Args:
        model_path: Path to TTM model
        context_length: Input sequence length
        forecast_length: Prediction horizon
        **kwargs: Additional configuration parameters

    Returns:
        Configured TTM forecaster
    """
    config = TTMConfig(
        model_path=model_path,
        context_length=context_length,
        forecast_length=forecast_length,
        **kwargs,
    )

    return TTMForecaster(config)
