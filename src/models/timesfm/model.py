"""
TimesFM model implementation using the base TSFM framework.

This module provides a concrete implementation of TimesFM that inherits from
the base TSFM framework, demonstrating how to integrate foundation models.
TimesFM is a pretrained decoder-only foundation model from Google Research.
"""

import os
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (
    TrainingArguments,
)

# Local imports
from src.models.base import BaseTimeSeriesFoundationModel
from src.models.timesfm.config import TimesFMConfig
from src.utils.logging_helper import info_print, error_print


class TimesFMForecaster(BaseTimeSeriesFoundationModel):
    """TimesFM forecaster implementation using the base TSFM framework.

    TimesFM is a pretrained time series foundation model developed by Google Research
    that uses a decoder-only transformer architecture. While the base model doesn't
    support fine-tuning, this implementation provides a consistent interface for
    potential adapter-based fine-tuning (e.g., AdaPTS).

    Attributes:
        config: TimesFM-specific configuration (TimesFMConfig instance).
        preprocessor: Optional preprocessor for data normalization.
        column_specifiers: Dictionary mapping data columns to their roles.

    Note:
        TimesFM is primarily designed for zero-shot forecasting, but adapter-based
        methods like AdaPTS could enable fine-tuning capabilities.

    Example:
        >>> config = TimesFMConfig(checkpoint_path="google/timesfm-1.0-200m")
        >>> model = TimesFMForecaster(config)
        >>> predictions = model.predict(test_data)
    """

    def __init__(
        self, config: TimesFMConfig, lora_config=None, distributed_config=None
    ):
        """Initialize the TimesFM forecaster.

        Args:
            config: TimesFM configuration object. If a non-TimesFMConfig is passed,
                it will be converted using essential parameters.
            lora_config: LoRA configuration for parameter-efficient fine-tuning
                (reserved for future adapter implementations).
            distributed_config: Configuration for distributed training
                (DDP, DeepSpeed, or FSDP).

        Note:
            The model is initialized during construction via _initialize_model().
        """
        # Use the config as-is if it's already a TimesFMConfig
        if not isinstance(config, TimesFMConfig):
            # Create a basic TimesFMConfig from the essential parameters
            essential_params = {
                "checkpoint_path": getattr(config, "checkpoint_path", None),
                "context_length": getattr(config, "context_length", 512),
                "horizon_length": getattr(config, "horizon_length", 128),
                "learning_rate": getattr(config, "learning_rate", 1e-4),
                "batch_size": getattr(config, "batch_size", 32),
                "num_epochs": getattr(config, "num_epochs", 10),
            }
            config = TimesFMConfig(**essential_params)

        super().__init__(config, lora_config, distributed_config)

        # Type annotation to help linter understand config type
        self.config: TimesFMConfig = self.config

        # TimesFM-specific attributes
        self.preprocessor = None
        self.column_specifiers = None
        self._is_loaded = False

    # Abstract method implementations
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
        data_loader, _, _ = self._prepare_training_data(data, None, None)

        # Set model to evaluation mode
        if hasattr(self.model, "eval"):
            self.model.eval()

        predictions = []
        with torch.no_grad():
            for batch in data_loader:
                # Process batch through TimesFM
                if isinstance(batch, dict):
                    # Extract time series data
                    if "input_values" in batch:
                        inputs = batch["input_values"]
                    elif "inputs" in batch:
                        inputs = batch["inputs"]
                    else:
                        raise ValueError("Cannot find input data in batch")
                else:
                    inputs = batch

                # Convert to numpy for TimesFM
                inputs_np = inputs.cpu().numpy()

                # Process each sample in batch
                batch_predictions = []
                for i in range(inputs_np.shape[0]):
                    ts = inputs_np[i]  # (seq_len,) or (seq_len, n_features)

                    # TimesFM expects 1D input, process each feature
                    if ts.ndim == 2:
                        # Multi-feature: process each feature separately
                        feature_preds = []
                        for feat_idx in range(ts.shape[1]):
                            forecast = self.model.forecast(
                                inputs=ts[:, feat_idx], freq=None
                            )
                            feature_preds.append(forecast[0])  # Point forecast
                        pred = np.stack(feature_preds, axis=-1)
                    else:
                        # Single feature
                        forecast = self.model.forecast(inputs=ts, freq=None)
                        pred = forecast[0]  # Point forecast

                    batch_predictions.append(pred)

                batch_predictions = np.stack(batch_predictions, axis=0)
                predictions.append(batch_predictions)

        # Concatenate all predictions
        predictions = np.concatenate(predictions, axis=0)

        if return_dict:
            return {
                "predictions": predictions,
                "model_type": "timesfm",
                "config": self.config.to_dict(),
            }

        return predictions

    def supports_lora(self) -> bool:
        """Check if TimesFM supports LoRA fine-tuning.

        Returns:
            False by default, but could be True with adapter frameworks like AdaPTS.
        """
        # While TimesFM doesn't natively support fine-tuning,
        # adapter methods could enable this in the future
        return False

    def _initialize_model(self) -> Any:
        """Initialize the TimesFM model.

        Returns:
            Initialized TimesFM model instance.

        Raises:
            ImportError: If timesfm package is not installed.
            ValueError: If model fails to load.
        """
        if self._is_loaded:
            return self.model

        info_print("Initializing TimesFM model")

        try:
            # Import timesfm library
            import timesfm

            info_print(f"Loading TimesFM with backend: {self.config.backend}")

            # Initialize TimesFM model
            model = timesfm.TimesFm(
                context_len=self.config.context_length,
                horizon_len=self.config.horizon_length,
                input_patch_len=self.config.input_patch_len,
                output_patch_len=self.config.output_patch_len,
                num_layers=self.config.num_layers,
                model_dims=self.config.model_dims,
                backend=self.config.backend,
            )

            # Load checkpoint
            if self.config.checkpoint_path:
                info_print(f"Loading checkpoint from: {self.config.checkpoint_path}")
                model.load_from_checkpoint(repo_id=self.config.checkpoint_path)
            else:
                # Use default checkpoint from Hugging Face
                info_print("Loading default TimesFM checkpoint from Hugging Face")
                model.load_from_checkpoint(repo_id="google/timesfm-1.0-200m")

            self._is_loaded = True
            info_print("TimesFM model loaded successfully")

            return model

        except ImportError:
            error_print(
                "timesfm package is required but not installed. "
                "Install it with: pip install timesfm"
            )
            raise
        except Exception as e:
            error_print(f"Failed to initialize TimesFM model: {str(e)}")
            raise

    def _prepare_training_data
        self,
        train_data: Any,
        val_data: Optional[Any] = None,
        test_data: Optional[Any] = None,
    ) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
        """Prepare data for training or inference.

        Args:
            train_data: Training data (can be dataset name, DataFrame, or DataLoader)
            val_data: Validation data
            test_data: Test data

        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # TODO: Implement data preparation logic for TimesFM
        # This should handle various input formats and create appropriate DataLoaders

        raise NotImplementedError(
            "Data preparation for TimesFM not yet implemented. "
            "Please implement data preprocessing in _prepare_training_data()."
        )

    def _get_training_args(self) -> TrainingArguments:
        """Get HuggingFace TrainingArguments for TimesFM.

        Note: TimesFM doesn't support native fine-tuning, but this is provided
        for potential adapter-based training (e.g., AdaPTS).

        Returns:
            TrainingArguments instance configured for TimesFM training.
        """
        return TrainingArguments(
            output_dir=self.config.output_dir or "./timesfm_output",
            per_device_train_batch_size=self.config.per_core_batch_size,
            per_device_eval_batch_size=self.config.per_core_batch_size,
            num_train_epochs=self.config.num_epochs,
            learning_rate=self.config.learning_rate,
            save_strategy="epoch",
            eval_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            logging_dir=os.path.join(
                self.config.output_dir or "./timesfm_output", "logs"
            ),
            logging_steps=10,
            save_total_limit=3,
            fp16=torch.cuda.is_available() and not self.config.use_cpu,
        )


def create_timesfm_model(
    checkpoint_path: Optional[str] = None,
    context_length: int = 512,
    horizon_length: int = 128,
    backend: str = "cpu",
    **kwargs,
) -> TimesFMForecaster:
    """Factory function to create a TimesFM model with sensible defaults.

    Args:
        checkpoint_path: Path to TimesFM checkpoint (default: google/timesfm-1.0-200m).
        context_length: Input sequence length.
        horizon_length: Output prediction horizon.
        backend: Backend to use ('cpu', 'gpu', 'tpu').
        **kwargs: Additional configuration parameters.

    Returns:
        Initialized TimesFMForecaster instance.

    Example:
        >>> model = create_timesfm_model(
        ...     checkpoint_path="google/timesfm-1.0-200m",
        ...     context_length=512,
        ...     horizon_length=128,
        ...     backend="gpu"
        ... )
    """
    config = TimesFMConfig(
        checkpoint_path=checkpoint_path,
        context_length=context_length,
        horizon_length=horizon_length,
        backend=backend,
        **kwargs,
    )

    return TimesFMForecaster(config)
