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
    EarlyStoppingCallback,
)

# Import your existing TTM-related modules
from tsfm_public import (
    TimeSeriesPreprocessor,
    get_datasets,
)
from tsfm_public.toolkit.get_model import get_model
from tsfm_public.toolkit.time_series_preprocessor import ScalerType

# Local imports
from src.models.base import BaseTSFM, TrainingStrategy
from src.models.ttm.config import TTMConfig
from src.data.diabetes_datasets.data_loader import get_loader
from src.data.models import ColumnNames
from src.data.preprocessing.split_or_combine_patients import (
    reduce_features_multi_patient,
)
from src.utils.logging_helper import info_print, debug_print, error_print


class TTMForecaster(BaseTSFM):
    """
    TTM (TinyTimeMixer) forecaster implementation using the base TSFM framework.

    This class demonstrates how to integrate an existing model (TTM) into the
    unified base framework while preserving all existing functionality.
    """

    def __init__(self, config: TTMConfig, lora_config=None, distributed_config=None):
        """Initialize TTM forecaster."""
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

    def _initialize_model(self) -> None:
        """Initialize the TTM model architecture."""
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
            if self.config.fit_strategy == "zero_shot":
                info_print("Freezing all parameters for zero-shot evaluation")
                for param in ttm_model.parameters():
                    param.requires_grad = False
            else:
                # For any training scenario (fine_tune, from_scratch), enable gradients
                info_print(
                    f"Enabling gradients for all parameters ({self.config.fit_strategy} mode)"
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
        """
        Prepare data loaders for TTM training.

        This method integrates with your existing data loading pipeline.
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
            )
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
            info_print(f"  Train samples: {len(dset_train) if dset_train else 0}")
            info_print(f"  Val samples: {len(dset_val) if dset_val else 0}")
            info_print(f"  Test samples: {len(dset_test) if dset_test else 0}")

            return train_loader, val_loader, test_loader

        except Exception as e:
            error_print(f"Failed to prepare data: {str(e)}")
            error_print(f"Failed to prepare data: {str(e)}")
            raise

    def _create_column_specifiers(self, data: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Create column specifiers based on data structure.

        This method maps your data columns to the expected format.
        Adapt this based on your actual column naming conventions.
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
        """Get distributed training arguments for TrainingArguments."""
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

    def _compute_metrics(self, eval_pred) -> Dict[str, float]:
        """
        Compute evaluation metrics for TTM.

        This uses your existing custom metrics computation logic.
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

    def get_training_strategy(self) -> TrainingStrategy:
        """TTM uses Transformers library for training."""
        return TrainingStrategy.TRANSFORMERS

    def supports_lora(self) -> bool:
        """TTM is MLP-based (Mixer architecture) and does NOT support LoRA fine-tuning."""
        return False

    def _get_callbacks(self) -> List:
        """Get training callbacks for TTM."""

        callbacks = []

        # Early stopping
        if self.config.early_stopping_patience > 0:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=self.config.early_stopping_patience,
                    early_stopping_threshold=0.0,
                )
            )

        return callbacks

    def _create_training_arguments(self, output_dir: str) -> TrainingArguments:
        """Create TTM-specific training arguments with distributed support."""
        # Base training arguments
        base_args = {
            "output_dir": output_dir,
            "learning_rate": self.config.learning_rate,
            "num_train_epochs": self.config.num_epochs,
            "per_device_train_batch_size": self.config.batch_size,
            "per_device_eval_batch_size": self.config.batch_size,
            "warmup_steps": self.config.warmup_steps,
            "weight_decay": self.config.weight_decay,
            "logging_dir": os.path.join(output_dir, "logs"),
            "logging_steps": self.config.logging_steps,
            "eval_strategy": self.config.eval_strategy,
            "eval_steps": self.config.eval_steps,
            "save_steps": self.config.save_steps,
            "metric_for_best_model": self.config.metric_for_best_model,
            "greater_is_better": self.config.greater_is_better,
            "load_best_model_at_end": True,
            "fp16": self.config.fp16,
            "dataloader_num_workers": self.config.dataloader_num_workers,
            "use_cpu": self.config.use_cpu,
            "report_to": "none",  # Disable wandb/tensorboard by default
        }

        # Add distributed training arguments
        distributed_args = self._get_distributed_training_args()
        base_args.update(distributed_args)

        return TrainingArguments(**base_args)

    def _train_model(
        self,
        train_data: Any,
        val_data: Optional[Any] = None,
        test_data: Optional[Any] = None,
        output_dir: str = "./output",
        **kwargs,
    ) -> Dict[str, Any]:
        """
        TTM-specific training implementation using Transformers Trainer.
        """
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
            compute_metrics=self._compute_metrics,
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

    def _save_model_weights(self, output_dir: str) -> None:
        """Save TTM model using Transformers format."""
        if self.model is not None:
            self.model.save_pretrained(output_dir)
            info_print(f"TTM model saved to {output_dir}")

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

    def _load_model_weights(self, model_dir: str) -> None:
        """Load TTM model weights from directory."""
        try:
            # TTM models can be loaded using the transformers library
            from transformers import AutoModel

            self.model = AutoModel.from_pretrained(model_dir)
            info_print(f"TTM model weights loaded from {model_dir}")

        except Exception as e:
            error_print(f"Failed to load model weights: {str(e)}")
            raise

    def predict_zero_shot(
        self,
        data: Any,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """
        Make zero-shot predictions (no fine-tuning).

        This method allows using pre-trained TTM without fine-tuning.
        """
        info_print("Making zero-shot predictions with TTM")

        # Temporarily override fit strategy
        original_strategy = self.config.fit_strategy
        self.config.fit_strategy = "zero_shot"

        try:
            predictions = self.predict(data, batch_size)
            return predictions
        finally:
            # Restore original strategy
            self.config.fit_strategy = original_strategy

    def get_ttm_specific_info(self) -> Dict[str, Any]:
        """Get TTM-specific model information."""
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
