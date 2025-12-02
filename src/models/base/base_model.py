# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)
# For commercial licensing, contact:

"""
Base model framework for Time Series Foundation Models (TSFMs).

This module provides the abstract base classes and utilities for implementing
different time series foundation models in a unified framework.
"""

import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from enum import Enum

# Local imports - adapt these to your existing structure
from src.utils.logging_helper import info_print, error_print


class TrainingStrategy(Enum):
    """Training strategy options for different model architectures."""

    TRANSFORMERS = "transformers"  # Uses transformers.Trainer
    PYTORCH = "pytorch"  # Custom PyTorch training loop
    CUSTOM = "custom"  # Model-specific training implementation


@dataclass
class ModelConfig:
    """Configuration class for model architecture and training parameters."""

    # Model architecture
    model_type: str = "base"
    model_path: Optional[str] = None  # Path to pre-trained model
    context_length: int = 512
    forecast_length: int = 96
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    dropout: float = 0.1

    # Model behavior
    fit_strategy: str = "fine_tune"  # "zero_shot", "fine_tune", "from_scratch"
    freeze_backbone: bool = False

    # Training configuration
    learning_rate: float = 1e-4
    batch_size: int = 64
    num_epochs: int = 10
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    gradient_clip_val: float = 1.0

    # Evaluation
    eval_strategy: str = "steps"
    eval_steps: int = 1000
    save_steps: int = 2000
    logging_steps: int = 100
    early_stopping_patience: int = 10
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False

    # Hardware/Performance
    fp16: bool = True
    dataloader_num_workers: int = 2
    use_cpu: bool = False

    # Training strategy
    training_strategy: TrainingStrategy = TrainingStrategy.TRANSFORMERS

    # Loss function
    loss_function: str = "mse"  # "mse", "mae", "huber", "pinball"

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        result = {}
        for k, v in self.__dict__.items():
            # Handle enum values by converting to their string representation
            if hasattr(v, "value"):  # Enum objects have a 'value' attribute
                result[k] = v.value
            else:
                result[k] = v
        return result

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ModelConfig":
        """Create config from dictionary."""
        return cls(**config_dict)


@dataclass
class LoRAConfig:
    """Configuration for LoRA (Low-Rank Adaptation) fine-tuning."""

    enabled: bool = False
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    bias: str = "none"  # "none", "all", "lora_only"

    # Architecture compatibility
    auto_detect_modules: bool = True  # Automatically detect target modules


@dataclass
class DistributedConfig:
    """Configuration for distributed training."""

    enabled: bool = False
    strategy: str = "ddp"  # "ddp", "deepspeed", "fsdp"
    world_size: int = 1
    local_rank: int = 0
    backend: str = "nccl"

    # DDP specific optimizations
    find_unused_parameters: bool = (
        False  # Set to False for better performance with static models like TTM
    )
    gradient_as_bucket_view: bool = True  # Enable for memory efficiency

    # DeepSpeed specific
    deepspeed_config: Optional[Dict[str, Any]] = None

    # FSDP specific
    fsdp_config: Optional[Dict[str, Any]] = None


class BaseTSFM(ABC):
    """
    Abstract base class for all Time Series Foundation Models.

    This class provides the common interface and functionality that all
    time series foundation models should implement, including:
    - Model initialization and configuration
    - Training pipeline management
    - Distributed training setup
    - LoRA integration for memory-efficient fine-tuning
    - Model saving/loading with metadata
    - Evaluation and metrics computation
    """

    def __init__(
        self,
        config: ModelConfig,
        lora_config: Optional[LoRAConfig] = None,
        distributed_config: Optional[DistributedConfig] = None,
    ):
        """
        Initialize the base TSFM.

        Args:
            config: Model configuration
            lora_config: LoRA configuration for efficient fine-tuning
            distributed_config: Distributed training configuration
        """
        self.config = config
        self.lora_config = lora_config or LoRAConfig()
        self.distributed_config = distributed_config or DistributedConfig()

        # Model and training components
        self.model: Optional[torch.nn.Module] = None
        self.tokenizer = None  # For models that need tokenization

        # Training state
        self.is_fitted = False
        self.training_history = {}
        self.best_metrics = {}

        # Distributed training
        self._distributed_setup_done = False

        # Logging
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize model
        self._initialize_model()

    @abstractmethod
    def _initialize_model(self) -> None:
        """Initialize the specific model architecture."""
        pass

    @abstractmethod
    def _prepare_data(
        self,
        train_data: Any,
        val_data: Optional[Any] = None,
        test_data: Optional[Any] = None,
    ) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
        """
        Prepare data loaders for training, validation, and testing.

        Args:
            train_data: Training dataset
            val_data: Validation dataset (optional)
            test_data: Test dataset (optional)

        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        pass

    @abstractmethod
    def get_training_strategy(self) -> TrainingStrategy:
        """
        Return the training strategy this model uses.

        Returns:
            TrainingStrategy: The training approach for this model
        """
        pass

    @abstractmethod
    def supports_lora(self) -> bool:
        """
        Check if this model architecture supports LoRA fine-tuning.

        Returns:
            bool: True if the model supports LoRA, False otherwise
        """
        pass

    def setup_distributed(self) -> None:
        """Set up distributed training if configured."""
        if not self.distributed_config.enabled or self._distributed_setup_done:
            return

        if self.distributed_config.strategy == "ddp":
            self._setup_ddp()
        elif self.distributed_config.strategy == "deepspeed":
            self._setup_deepspeed()
        elif self.distributed_config.strategy == "fsdp":
            self._setup_fsdp()
        else:
            raise ValueError(
                f"Unknown distributed strategy: {self.distributed_config.strategy}"
            )

        self._distributed_setup_done = True

    def _setup_ddp(self) -> None:
        """Set up PyTorch Distributed Data Parallel.
        DDP requires explicit setup before the Trainer, unlike DeepSpeed and FSDP.
        """
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend=self.distributed_config.backend,
                world_size=self.distributed_config.world_size,
                rank=self.distributed_config.local_rank,
            )

        if self.model is not None and torch.cuda.is_available():
            device = torch.device(f"cuda:{self.distributed_config.local_rank}")
            # Move model to the appropriate device
            self.model = self.model.to(device)
            # Wrap with DDP
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.distributed_config.local_rank],
                output_device=self.distributed_config.local_rank,
            )

    def _setup_deepspeed(self) -> None:
        """Set up DeepSpeed for large model training."""
        # DeepSpeed configuration is handled entirely by the Trainer through TrainingArguments
        # We just validate that the config exists if needed
        if self.distributed_config.deepspeed_config is None:
            info_print(
                "DeepSpeed enabled but no config provided - using Trainer defaults"
            )
        else:
            info_print(
                "DeepSpeed will be configured in TrainingArguments with provided config"
            )

    def _setup_fsdp(self) -> None:
        """Set up Fully Sharded Data Parallel."""
        info_print("FSDP will be configured in TrainingArguments")

    def enable_lora(self) -> None:
        """Enable LoRA for memory-efficient fine-tuning."""
        if not self.lora_config.enabled or self.model is None:
            return

        # Check if this model supports LoRA
        if not self.supports_lora():
            info_print(
                f"LoRA is not supported for {self.__class__.__name__} architecture"
            )
            info_print("LoRA requires transformer-based models with attention layers")
            return

        try:
            from peft import LoraConfig, get_peft_model, TaskType
        except ImportError:
            error_print("PEFT is not installed. Please install with: pip install peft")
            return

        # Create LoRA config
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,  # Adjust based on your task
            inference_mode=False,
            r=self.lora_config.rank,
            lora_alpha=self.lora_config.alpha,
            lora_dropout=self.lora_config.dropout,
            target_modules=self.lora_config.target_modules,
            bias=self.lora_config.bias,
        )

        # Auto-detect target modules if enabled
        if self.lora_config.auto_detect_modules:
            detected_modules = self._detect_lora_target_modules()
            if detected_modules:
                peft_config.target_modules = detected_modules
                info_print(f"Auto-detected LoRA target modules: {detected_modules}")
            else:
                info_print(
                    f"Using configured target modules: {self.lora_config.target_modules}"
                )

        # Apply LoRA to model
        self.model = get_peft_model(self.model, peft_config)
        info_print(f"LoRA enabled with rank {self.lora_config.rank}")
        info_print(f"Target modules: {peft_config.target_modules}")

        # Print trainable parameters
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        total_params = sum(p.numel() for p in self.model.parameters())
        info_print(f"Trainable parameters: {trainable_params:,}")
        info_print(f"Total parameters: {total_params:,}")
        info_print(f"Trainable %: {100 * trainable_params / total_params:.2f}%")

    def _detect_lora_target_modules(self) -> List[str]:
        """
        Automatically detect suitable target modules for LoRA.

        Returns:
            List[str]: List of module names suitable for LoRA adaptation
        """
        if self.model is None:
            return []

        target_modules = []

        # Common transformer module patterns
        transformer_patterns = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",  # Attention projections
            "gate_proj",
            "up_proj",
            "down_proj",  # Feed-forward layers
            "query",
            "key",
            "value",
            "output",  # Alternative naming
            "dense",
            "linear",  # Generic linear layers
        ]

        # Scan model modules
        for name, module in self.model.named_modules():
            module_name = name.split(".")[-1]  # Get the last part of the name

            # Check if it's a linear layer and matches patterns
            if hasattr(module, "weight") and hasattr(module, "bias"):
                if any(
                    pattern in module_name.lower() for pattern in transformer_patterns
                ):
                    if module_name not in target_modules:
                        target_modules.append(module_name)

        # Remove duplicates and sort
        target_modules = sorted(list(set(target_modules)))

        return target_modules

    def fit(
        self,
        train_data: Any,
        val_data: Optional[Any] = None,
        test_data: Optional[Any] = None,
        output_dir: str = "./output",
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Fit the model to training data.

        This is a simple wrapper that just calls the model's specific
        training implementation. Each model handles its own training strategy.
        """
        info_print(f"Starting training for {self.__class__.__name__}")
        info_print(f"Training strategy: {self.get_training_strategy().value}")

        # Setup distributed training if configured
        self.setup_distributed()

        # Enable LoRA if configured and supported
        self.enable_lora()

        # Let each model handle its own training
        return self._train_model(train_data, val_data, test_data, output_dir, **kwargs)

    @abstractmethod
    def _train_model(
        self,
        train_data: Any,
        val_data: Optional[Any],
        test_data: Optional[Any],
        output_dir: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Model-specific training implementation.

        Args:
            train_data: Training dataset
            val_data: Validation dataset (optional)
            test_data: Test dataset (optional)
            output_dir: Directory to save outputs
            **kwargs: Additional arguments
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _train_model() method"
        )

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

    def evaluate(self, test_data: Any) -> Dict[str, float]:
        """Evaluate the model. Models can override this entirely."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement evaluate() method"
        )

    def save_model(
        self, output_dir: str, save_config: bool = True, save_metadata: bool = True
    ) -> None:
        """
        Save the model and associated metadata.
        Models can override this entirely or call super() for common metadata saving.
        """
        os.makedirs(output_dir, exist_ok=True)

        # Save configuration
        if save_config:
            config_path = os.path.join(output_dir, "config.json")
            with open(config_path, "w") as f:
                json.dump(self.config.to_dict(), f, indent=2)

        # Save metadata
        if save_metadata:
            metadata = {
                "model_type": self.__class__.__name__,
                "is_fitted": self.is_fitted,
                "training_history": self.training_history,
                "best_metrics": self.best_metrics,
                "config": self.config.to_dict(),
                "lora_config": self.lora_config.__dict__,
                "distributed_config": self.distributed_config.__dict__,
                "training_strategy": self.get_training_strategy().value,
            }

            metadata_path = os.path.join(output_dir, "metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

        # Models should override this to save actual model weights
        raise NotImplementedError(
            f"{self.__class__.__name__} must override save_model() to save model weights"
        )

    @classmethod
    def load_model(
        cls, model_dir: str, config: Optional[ModelConfig] = None
    ) -> "BaseTSFM":
        """
        Load a saved model.

        Args:
            model_dir: Directory containing the saved model
            config: Optional config override

        Returns:
            Loaded model instance
        """
        # Load config if not provided
        if config is None:
            config_path = os.path.join(model_dir, "config.json")
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config_dict = json.load(f)
                config = ModelConfig.from_dict(config_dict)
            else:
                raise ValueError(f"No config found at {config_path}")

        # Create instance
        instance = cls(config)

        # Load the actual model weights
        # This is model-specific and should be implemented in subclasses
        instance._load_model_weights(model_dir)

        # Load metadata
        metadata_path = os.path.join(model_dir, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            instance.training_history = metadata.get("training_history", {})
            instance.best_metrics = metadata.get("best_metrics", {})
            instance.is_fitted = metadata.get("is_fitted", False)

        info_print(f"Model loaded from {model_dir}")
        return instance

    @abstractmethod
    def _load_model_weights(self, model_dir: str) -> None:
        """Load model weights from directory. Each model implements this."""
        pass

    def _get_early_stopping_config(self) -> Dict[str, Any]:
        """Get early stopping configuration for models that support it."""
        return {
            "patience": self.config.early_stopping_patience,
            "threshold": 0.0,
            "metric": self.config.metric_for_best_model,
            "greater_is_better": self.config.greater_is_better,
        }

    def _save_training_metadata(self, output_dir: str, metrics: Dict[str, Any]) -> None:
        """Save comprehensive training metadata.
        1) Supports the experiment tracking system
        2) Enables the model registry to have rich metadata about each trained model
        3) Facilitates reproducibility by capturing the exact state when training occurred
        4) Helps with debugging by recording all relevant configuration and environment details
        """
        metadata_file = os.path.join(output_dir, "training_metadata.json")

        # Add additional metadata
        metadata = {
            "model_type": self.__class__.__name__,
            "timestamp": pd.Timestamp.now().isoformat(),
            "metrics": metrics,
            "config": self.config.to_dict(),
            "lora_enabled": self.lora_config.enabled,
            "distributed_enabled": self.distributed_config.enabled,
        }

        # Add git information if available
        try:
            import git

            repo = git.Repo(search_parent_directories=True)
            metadata["git_commit"] = repo.head.commit.hexsha
            metadata["git_branch"] = repo.active_branch.name
            metadata["git_dirty"] = repo.is_dirty()
        except (ImportError, Exception):
            # Git info not critical - continue without it
            # This catches ImportError (GitPython not installed) and any git-related errors
            pass

        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        info_print(f"Training metadata saved to {metadata_file}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        info = {
            "model_type": self.__class__.__name__,
            "config": self.config.to_dict(),
            "is_fitted": self.is_fitted,
            "lora_enabled": self.lora_config.enabled,
            "distributed_enabled": self.distributed_config.enabled,
            "training_strategy": self.get_training_strategy().value,
        }

        if self.model is not None:
            # Count parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )

            info.update(
                {
                    "total_parameters": total_params,
                    "trainable_parameters": trainable_params,
                    "trainable_percentage": 100 * trainable_params / total_params
                    if total_params > 0
                    else 0,
                }
            )

        return info

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(config={self.config.model_type}, fitted={self.is_fitted})"


# Utility functions for model management


def create_model_from_config(config_path: str) -> BaseTSFM:
    """
    Factory function to create a model from a configuration file.

    Args:
        config_path: Path to configuration file

    Returns:
        Model instance
    """
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    model_type = config_dict.pop("model_type", "base")

    # Import and create the appropriate model class
    if model_type == "ttm":
        from src.models.ttm import TTMForecaster, TTMConfig

        config = TTMConfig(**config_dict)
        return TTMForecaster(config)
    elif model_type == "chronos":
        from src.models.chronos.model import ChronosForecaster

        config = ModelConfig.from_dict(config_dict)
        return ChronosForecaster(config)
    elif model_type == "tsmixer":
        from src.models.tsmixer import TSMixerForecaster, TSMixerConfig

        config = TSMixerConfig(**config_dict)
        return TSMixerForecaster(config)
    # Add other model types as needed
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def compare_models(
    models: List[BaseTSFM], test_data: Any, metrics: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple models on the same test dataset.

    Args:
        models: List of fitted models to compare
        test_data: Test dataset
        metrics: List of metrics to compute

    Returns:
        Dictionary mapping model names to their metrics
    """
    results = {}

    for model in models:
        if not model.is_fitted:
            error_print(f"Model {model.__class__.__name__} is not fitted, skipping")
            continue

        # Prepare test data
        _, _, test_loader = model._prepare_data(None, None, test_data)

        # Evaluate model
        model_metrics = model.evaluate(test_loader)
        results[model.__class__.__name__] = model_metrics

    return results
