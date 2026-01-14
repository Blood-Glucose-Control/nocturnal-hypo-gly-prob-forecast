"""
Moment model implementation using the base TSFM framework.

This module provides a concrete implementation of Moment that inherits from
the base TSFM framework, demonstrating how to integrate foundation models.
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
from src.models.moment.config import MomentConfig
from src.utils.logging_helper import info_print, error_print


class MomentForecaster(BaseTimeSeriesFoundationModel):
    """Moment forecaster implementation using the base TSFM framework.

    Moment is a time series foundation model developed by Carnegie Mellon University
    that uses transformer-based architecture with masked reconstruction pre-training.
    This class integrates Moment with the unified base framework.

    Attributes:
        config: Moment-specific configuration (MomentConfig instance).
        preprocessor: Optional preprocessor for data normalization.
        column_specifiers: Dictionary mapping data columns to their roles.

    Note:
        Moment DOES support LoRA fine-tuning as it is transformer-based.

    Example:
        >>> config = MomentConfig(model_path="AutonLab/MOMENT-1-large")
        >>> model = MomentForecaster(config)
        >>> model.fit(train_data="kaggle_brist1d")
        >>> predictions = model.predict(test_data)
    """

    def __init__(self, config: MomentConfig, lora_config=None, distributed_config=None):
        """Initialize the Moment forecaster.

        Args:
            config: Moment configuration object. If a non-MomentConfig is passed,
                it will be converted using essential parameters.
            lora_config: LoRA configuration for parameter-efficient fine-tuning.
            distributed_config: Configuration for distributed training
                (DDP, DeepSpeed, or FSDP).

        Note:
            The model is initialized during construction via _initialize_model().
        """
        # Use the config as-is if it's already a MomentConfig
        if not isinstance(config, MomentConfig):
            # Create a basic MomentConfig from the essential parameters
            essential_params = {
                "model_path": getattr(config, "model_path", "AutonLab/MOMENT-1-large"),
                "context_length": getattr(config, "context_length", 512),
                "forecast_length": getattr(config, "forecast_length", 96),
                "learning_rate": getattr(config, "learning_rate", 1e-4),
                "batch_size": getattr(config, "batch_size", 32),
                "num_epochs": getattr(config, "num_epochs", 10),
            }
            config = MomentConfig(**essential_params)

        super().__init__(config, lora_config, distributed_config)

        # Type annotation to help linter understand config type
        self.config: MomentConfig = self.config

        # Moment-specific attributes
        self.preprocessor = None
        self.column_specifiers = None

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

                # Extract predictions
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
                "model_type": "moment",
                "config": self.config.to_dict(),
            }

        return predictions

    def supports_lora(self) -> bool:
        """Check if Moment supports LoRA fine-tuning.

        Returns:
            True, as Moment is transformer-based and supports LoRA.
        """
        return True

    def _initialize_model(self) -> Any:
        """Initialize the Moment model.

        Returns:
            Initialized Moment model instance.

        Raises:
            ValueError: If model_path is not specified or model fails to load.
        """
        if not self.config.model_path:
            raise ValueError("model_path must be specified in config")

        info_print(f"Initializing Moment model from {self.config.model_path}")

        try:
            # TODO: Implement actual Moment model loading
            # This will depend on how Moment models are distributed
            # For now, this is a placeholder

            # Example structure (to be implemented):
            # from momentfm import MOMENTPipeline
            # model = MOMENTPipeline.from_pretrained(
            #     self.config.model_path,
            #     context_length=self.config.context_length,
            #     forecast_length=self.config.forecast_length,
            # )

            raise NotImplementedError(
                "Moment model loading not yet implemented. "
                "Please implement model initialization in _initialize_model()."
            )

        except Exception as e:
            error_print(f"Failed to initialize Moment model: {str(e)}")
            raise

    def _prepare_data(
        self, data: Any, split: Optional[str] = None
    ) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
        """Prepare data for training or inference.

        Args:
            data: Input data (can be dataset name, DataFrame, or DataLoader)
            split: Optional split to use ('train', 'val', 'test')

        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # TODO: Implement data preparation logic
        # This should handle various input formats and create appropriate DataLoaders

        raise NotImplementedError(
            "Data preparation for Moment not yet implemented. "
            "Please implement data preprocessing in _prepare_data()."
        )

    def _get_training_args(self) -> TrainingArguments:
        """Get HuggingFace TrainingArguments for Moment.

        Returns:
            TrainingArguments instance configured for Moment training.
        """
        return TrainingArguments(
            output_dir=self.config.output_dir or "./moment_output",
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            num_train_epochs=self.config.num_epochs,
            learning_rate=self.config.learning_rate,
            save_strategy="epoch",
            evaluation_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            logging_dir=os.path.join(
                self.config.output_dir or "./moment_output", "logs"
            ),
            logging_steps=10,
            save_total_limit=3,
            fp16=torch.cuda.is_available() and not self.config.use_cpu,
        )


def create_moment_model(
    model_path: str = "AutonLab/MOMENT-1-large",
    context_length: int = 512,
    forecast_length: int = 96,
    use_lora: bool = False,
    **kwargs,
) -> MomentForecaster:
    """Factory function to create a Moment model with sensible defaults.

    Args:
        model_path: HuggingFace model identifier or local path.
        context_length: Input sequence length.
        forecast_length: Output prediction horizon.
        use_lora: Whether to use LoRA for parameter-efficient fine-tuning.
        **kwargs: Additional configuration parameters.

    Returns:
        Initialized MomentForecaster instance.

    Example:
        >>> model = create_moment_model(
        ...     model_path="AutonLab/MOMENT-1-large",
        ...     context_length=512,
        ...     forecast_length=96,
        ...     use_lora=True
        ... )
    """
    config = MomentConfig(
        model_path=model_path,
        context_length=context_length,
        forecast_length=forecast_length,
        use_lora=use_lora,
        **kwargs,
    )

    return MomentForecaster(config)
