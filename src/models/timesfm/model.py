"""
TimesFM model implementation using the base TSFM framework.

This module provides a concrete implementation of TimesFM that inherits from
the base TSFM framework, demonstrating how to integrate foundation models.
TimesFM is a pretrained decoder-only foundation model from Google Research.
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

# Local imports
from src.models.base import BaseTSFM, TrainingStrategy
from src.models.timesfm.config import TimesFMConfig
from src.utils.logging_helper import info_print, debug_print, error_print


class TimesFMForecaster(BaseTSFM):
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

    def __init__(self, config: TimesFMConfig, lora_config=None, distributed_config=None):
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
        data_loader, _, _ = self._prepare_data(data)

        # Set model to evaluation mode
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
                    
        _prepare_data(
        self, data: Any, split: Optional[str] = None
    ) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
        """Prepare data for training or inference.

        Args:
            data: Input data (can be dataset name, DataFrame, or DataLoader)
            split: Optional split to use ('train', 'val', 'test')

        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # TODO: Implement data preparation logic for TimesFM
        # This should handle various input formats and create appropriate DataLoaders
        
        raise NotImplementedError(
            "Data preparation for TimesFM not yet implemented. "
            "Please implement data preprocessing in _prepare_data()."
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
            evaluation_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            logging_dir=os.path.join(
                self.config.output_dir or "./timesfm_output", "logs"
            ),
            logging_steps=10,
            save_total_limit=3,
            fp16=torch.cuda.is_available() and self.config.device != "cpu",
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

    return TimesFMForecaster(config) # quantiles shape: (num_quantiles, horizon_length)
                    feature_quantiles.append(quantiles)
                
                # Stack batch quantiles
                feature_quantiles = np.stack(feature_quantiles, axis=0)
                quantile_predictions.append(feature_quantiles)
            
            # Stack feature quantiles
            quantile_predictions = np.stack(quantile_predictions, axis=-1)
            # Shape: (batch_size, num_quantiles, horizon_length, n_features)
            
            results['quantiles'] = torch.from_numpy(quantile_predictions).to(x.device).float()
        
        return results
    
    def fit(
        self,
        train_data: Any,
        val_data: Optional[Any] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Fit the model (no-op for pretrained TimesFM).
        
        TimesFM is a pretrained model and does not require fine-tuning.
        This method is provided for interface compatibility.
        
        Args:
            train_data: Training data (not used)
            val_data: Validation data (not used)
            **kwargs: Additional arguments
            
        Returns:
            Empty training history dictionary
        """
        logger.warning(
            "TimesFM is a pretrained model and does not support fine-tuning. "
            "The fit() method is a no-op."
        )
        
        if not self._is_loaded:
            self._load_model()
        
        return {"history": {}}
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint.
        
        Note: TimesFM is a pretrained model, so we only save configuration.
        
        Args:
            path: Path to save checkpoint
        """
        logger.info(f"Saving TimesFM config to {path}")
        # Save config only, as model is pretrained
        torch.save({
            'config': self.config,
            'model_name': self.config.model_name,
        }, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint.
        
        Args:
            path: Path to load checkpoint from
        """
        logger.info(f"Loading TimesFM checkpoint from {path}")
        checkpoint = torch.load(path, map_location='cpu')
        self.config = checkpoint['config']
        
        # Reload model with new config
        self._is_loaded = False
        self._load_model()
    
    @property
    def device(self) -> torch.device:
        """Get the device the model is on."""
        return torch.device(self.config.device)
    
    def to(self, device: torch.device):
        """Move model to specified device.
        
        Args:
            device: Target device
        """
        self.config.device = str(device)
        # Note: TimesFM handles device placement internally
        return self
    
    def train(self, mode: bool = True):
        """Set model to training mode (no-op for TimesFM)."""
        # TimesFM is inference-only
        return self
    
    def eval(self):
        """Set model to evaluation mode (no-op for TimesFM)."""
        # TimesFM is always in eval mode
        return self
