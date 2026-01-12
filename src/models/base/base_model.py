# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)
# For commercial licensing, contact: cjrisi/christopher AT uwaterloo/gluroo DOT ca/com

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
    """Configuration class for model architecture and training parameters.

    This dataclass holds all configuration parameters needed to initialize,
    train, and evaluate a time series foundation model.

    Attributes:
        model_type: Identifier for the model type (e.g., "ttm", "chronos").
        model_path: Path to pre-trained model weights or HuggingFace model ID.
        context_length: Number of historical time steps used as input.
        forecast_length: Number of future time steps to predict.
        d_model: Dimension of the model's hidden representations.
        n_heads: Number of attention heads (for transformer-based models).
        n_layers: Number of transformer/encoder layers.
        dropout: Dropout probability for regularization.
        fit_strategy: Training approach - "zero_shot", "fine_tune", or "from_scratch".
        freeze_backbone: Whether to freeze pre-trained weights during fine-tuning.
        learning_rate: Learning rate for the optimizer.
        batch_size: Number of samples per training batch.
        num_epochs: Number of training epochs.
        warmup_steps: Number of warmup steps for learning rate scheduler.
        weight_decay: L2 regularization coefficient.
        gradient_clip_val: Maximum gradient norm for clipping.
        eval_strategy: When to evaluate - "steps" or "epoch".
        eval_steps: Number of steps between evaluations (if eval_strategy="steps").
        save_steps: Number of steps between checkpoint saves.
        logging_steps: Number of steps between logging updates.
        early_stopping_patience: Epochs without improvement before stopping.
        metric_for_best_model: Metric to monitor for best model selection.
        greater_is_better: Whether higher metric values are better.
        fp16: Whether to use mixed precision (FP16) training.
        dataloader_num_workers: Number of worker processes for data loading.
        use_cpu: Force CPU usage even if GPU is available.
        training_strategy: The training framework to use (Transformers, PyTorch, etc.).
        loss_function: Loss function for training - "mse", "mae", "huber", or "pinball".
    """

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
        """Convert configuration to a dictionary.

        Converts all configuration attributes to a dictionary format suitable
        for JSON serialization. Enum values are converted to their string
        representations.

        Returns:
            Dict[str, Any]: Dictionary containing all configuration parameters.
        """
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
        """Create a configuration instance from a dictionary.

        Args:
            config_dict: Dictionary containing configuration parameters.
                Keys should match the attribute names of ModelConfig.

        Returns:
            ModelConfig: New configuration instance with values from the dictionary.

        Raises:
            TypeError: If config_dict contains keys that are not valid attributes.
        """
        return cls(**config_dict)


@dataclass
class LoRAConfig:
    """Configuration for LoRA (Low-Rank Adaptation) fine-tuning.

    LoRA enables memory-efficient fine-tuning by adding trainable low-rank
    decomposition matrices to transformer layers while keeping the original
    weights frozen.

    Attributes:
        enabled: Whether to enable LoRA fine-tuning.
        rank: Rank of the low-rank decomposition matrices. Lower values use
            less memory but may reduce model capacity.
        alpha: Scaling factor for LoRA updates. Higher values increase the
            influence of LoRA adaptations.
        dropout: Dropout probability applied to LoRA layers.
        target_modules: List of module names to apply LoRA to (e.g., ["q_proj", "v_proj"]).
        bias: How to handle bias terms - "none", "all", or "lora_only".
        auto_detect_modules: Whether to automatically detect suitable target modules
            based on the model architecture.
    """

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
    """Configuration for distributed training across multiple GPUs or nodes.

    Supports various distributed training strategies including DDP (Distributed
    Data Parallel), DeepSpeed, and FSDP (Fully Sharded Data Parallel).

    Attributes:
        enabled: Whether to enable distributed training.
        strategy: Distributed training strategy - "ddp", "deepspeed", or "fsdp".
        world_size: Total number of processes participating in training.
        local_rank: Rank of this process on the local node (0-indexed).
        backend: Communication backend - "nccl" (GPU) or "gloo" (CPU).
        find_unused_parameters: Whether DDP should find unused parameters. Set to
            False for better performance with static computation graphs.
        gradient_as_bucket_view: Enable memory-efficient gradient bucketing in DDP.
        deepspeed_config: DeepSpeed configuration dictionary for ZeRO optimization,
            mixed precision, and other DeepSpeed-specific settings.
        fsdp_config: FSDP configuration dictionary for sharding policy,
            CPU offloading, and other FSDP-specific settings.
    """

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

    # Abstract methods that child classes must implement
    ## Abstract public API methods
    @abstractmethod
    def predict(
        self, data: Any, batch_size: Optional[int] = None, return_dict: bool = False
    ) -> Union[np.ndarray, Dict[str, Any]]:
        """
        Make predictions on new data.

        Each model must implement this method to handle its specific
        prediction logic and output format.

        Args:
            data: Input data for prediction
            batch_size: Batch size for prediction (defaults to config.batch_size)
            return_dict: Whether to return additional information

        Returns:
            Predictions as numpy array or dictionary with additional info
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

    ## Abstract Protected Methods
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
    def _save_model_weights(self, output_dir: str) -> None:
        """Save model weights to directory. Each model implements this."""
        pass

    @abstractmethod
    def _load_model_weights(self, model_dir: str) -> None:
        """Load model weights from directory. Each model implements this."""
        pass

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

    # Public API (fit, predict, evaluate, save_model, load_model, get_model_info...)
    def fit(
        self,
        train_data: Any,
        val_data: Optional[Any] = None,
        test_data: Optional[Any] = None,
        output_dir: str = "./output",
        **kwargs,
    ) -> Dict[str, Any]:
        """Fit the model to training data.

        This method orchestrates the complete training pipeline including:
        - Setting up distributed training (if configured)
        - Enabling LoRA adapters (if configured)
        - Calling the model-specific training implementation
        - Saving training metadata
        - Cleaning up distributed resources

        Args:
            train_data: Training dataset. Format depends on the specific model
                implementation (e.g., DataFrame, Dataset, or data source name).
            val_data: Validation dataset for monitoring training progress.
            test_data: Test dataset for final evaluation after training.
            output_dir: Directory path where model checkpoints, logs, and
                metadata will be saved.
            **kwargs: Additional keyword arguments passed to the model-specific
                training implementation (e.g., resume_from_checkpoint).

        Returns:
            Dict[str, Any]: Dictionary containing training metrics, including
                'train_metrics' and optionally 'test_metrics'.

        Raises:
            Exception: If training fails. Distributed cleanup is guaranteed
                to run even if training raises an exception.
        """
        info_print(f"Starting training for {self.__class__.__name__}")
        info_print(f"Training strategy: {self.get_training_strategy().value}")

        # Setup distributed training if configured
        self._setup_distributed()

        # Enable LoRA if configured and supported
        self._enable_lora()

        try:
            # Let each model handle its own training
            metrics = self._train_model(
                train_data, val_data, test_data, output_dir, **kwargs
            )

            # Post-training state updates
            self.is_fitted = True
            self.training_history = metrics.get("train_metrics", metrics)
            self._save_training_metadata(output_dir, metrics)

            info_print("🏁 Training complete!")
            return metrics

        finally:
            # Common cleanup that must happen in distributed training scenarios
            # This can causes serious issues causes GPUs to be locked if not run.
            self._cleanup_distributed()

    def evaluate(
        self, test_data: Any, return_predictions: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate the model on test data.

        This base implementation calls predict() and computes standard metrics.
        Child classes can override for model-specific evaluation (e.g., using Trainer).

        Args:
            test_data: Test dataset (format depends on model's _prepare_data)
            return_predictions: Whether to include predictions in output

        Returns:
            Dictionary containing metrics and optionally predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")

        # Get predictions using the child's predict implementation
        predictions = self.predict(test_data)

        # Extract ground truth from test data
        # Child classes may need to override if their data format differs
        y_true = self._extract_ground_truth(test_data)

        # Compute metrics using protected method
        metrics = self._compute_metrics(predictions, y_true)

        if return_predictions:
            return {
                "metrics": metrics,
                "predictions": predictions,
                "ground_truth": y_true,
            }

        return metrics

    def save_model(
        self, output_dir: str, save_config: bool = True, save_metadata: bool = True
    ) -> None:
        """Save the model and associated metadata to disk.

        Saves configuration and training metadata to JSON files. Child classes
        must override this method to implement actual model weight saving.

        Args:
            output_dir: Directory path where the model will be saved.
            save_config: Whether to save the model configuration to config.json.
            save_metadata: Whether to save training metadata to metadata.json.

        Raises:
            NotImplementedError: Always raised by base class. Child classes
                must override to save model weights.

        Note:
            Child classes should call super().save_model() first to save
            configuration and metadata, then save model-specific weights.
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

        self._save_model_weights(output_dir)  # <-- This line instead of raising

        info_print(f"Model saved to {output_dir}")

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

    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the model.

        Returns:
            Dict[str, Any]: Dictionary containing:
                - model_type: Name of the model class.
                - config: Full model configuration as dictionary.
                - is_fitted: Whether the model has been trained.
                - lora_enabled: Whether LoRA is enabled.
                - distributed_enabled: Whether distributed training is enabled.
                - training_strategy: The training framework being used.
                - total_parameters: Total number of model parameters (if model exists).
                - trainable_parameters: Number of trainable parameters (if model exists).
                - trainable_percentage: Percentage of parameters that are trainable.
        """
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

    # Protected Helpers
    ## Distributed Training Setup
    def _setup_distributed(self) -> None:
        """Set up distributed training environment if configured.

        Initializes the appropriate distributed training backend based on the
        configured strategy (DDP, DeepSpeed, or FSDP). This method is idempotent
        and will skip setup if already initialized.

        Raises:
            ValueError: If an unknown distributed strategy is specified.
        """
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

    def _cleanup_distributed(self) -> None:
        """Clean up distributed training resources properly."""
        if (
            self.distributed_config.enabled
            and self.distributed_config.strategy == "ddp"
            and torch.distributed.is_initialized()
        ):
            info_print("🧹 Cleaning up distributed training resources...")
            try:
                # Synchronize all processes before cleanup
                torch.distributed.barrier()
                # Properly destroy the process group
                torch.distributed.destroy_process_group()
                info_print(
                    f"✅ Distributed training cleanup complete for {self.distributed_config.local_rank}"
                )
            except Exception as e:
                # Don't crash if cleanup fails, but warn about it
                info_print(f"⚠️  Warning: Distributed cleanup failed: {e}")

    def _setup_ddp(self) -> None:
        """Set up PyTorch Distributed Data Parallel (DDP).

        Initializes the distributed process group for DDP training. Requires
        MASTER_ADDR environment variable to be set. This method is typically
        called via _setup_distributed() rather than directly.

        Raises:
            ValueError: If MASTER_ADDR environment variable is not set.
        """
        if torch.distributed.is_initialized():
            return  # Already initialized

        # Check if we have required environment variables
        master_addr = os.environ.get("MASTER_ADDR")
        master_port = os.environ.get("MASTER_PORT", "29500")

        if not master_addr:
            error_print(
                "MASTER_ADDR environment variable not set for distributed training!"
            )
            error_print("Run with: torchrun --nproc_per_node=N --nnodes=1 script.py")
            raise ValueError(
                "Distributed training requires MASTER_ADDR environment variable"
            )

        info_print(f"Initializing DDP with MASTER_ADDR={master_addr}:{master_port}")

        # Set the device for this process
        device_id = None
        if torch.cuda.is_available() and not self.config.use_cpu:
            device_id = self.distributed_config.local_rank
            torch.cuda.set_device(device_id)
            info_print(f"Set CUDA device to GPU {device_id}")

        torch.distributed.init_process_group(
            backend=self.distributed_config.backend,
            world_size=self.distributed_config.world_size,
            rank=self.distributed_config.local_rank,
            device_id=device_id,
        )
        info_print(
            f"✅ Initialized DDP process group: rank {self.distributed_config.local_rank}/{self.distributed_config.world_size}"
        )

        # NOTE: Don't wrap model with DDP here for Transformers-based models
        # The Trainer handles that automatically

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

    ## LoRA Integration
    def _enable_lora(self) -> None:
        """Enable LoRA (Low-Rank Adaptation) for memory-efficient fine-tuning.

        Applies LoRA adapters to the model's target modules, freezing the original
        weights and adding trainable low-rank matrices. This significantly reduces
        memory requirements for fine-tuning large models.

        The method will skip LoRA setup if:
        - LoRA is not enabled in the configuration
        - The model is None
        - The model architecture doesn't support LoRA

        Note:
            Requires the PEFT library to be installed.
        """
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
            "o_proj",  # Attention projections (LLaMA-style)
            "wQKV",
            "wO",  # Toto attention layers
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

    ## Metrics Computation
    def _compute_metrics(
        self, y_pred: np.ndarray, y_true: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute standard evaluation metrics.

        This is a protected method providing common metrics computation.
        Child classes can override to add model-specific metrics.

        Args:
            y_pred: Predicted values
            y_true: Ground truth values

        Returns:
            Dictionary of metric names to values
        """
        # Ensure numpy arrays
        if not isinstance(y_pred, np.ndarray):
            y_pred = np.array(y_pred)
        if not isinstance(y_true, np.ndarray):
            y_true = np.array(y_true)

        # Flatten if needed for comparison
        y_pred = y_pred.flatten()
        y_true = y_true.flatten()

        # Core metrics
        mse = float(np.mean((y_pred - y_true) ** 2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(y_pred - y_true)))

        # MAPE with zero handling
        mask = y_true != 0
        if np.any(mask):
            mape = float(
                np.mean(np.abs((y_pred[mask] - y_true[mask]) / y_true[mask])) * 100
            )
        else:
            mape = 0.0

        return {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "mape": mape,
        }

    def _extract_ground_truth(self, test_data: Any) -> np.ndarray:
        """
        Extract ground truth labels from test data.

        This is a protected method that child classes should override
        if their data format differs from the default expectation.

        Args:
            test_data: Test dataset

        Returns:
            Ground truth values as numpy array
        """
        # Default implementation - child classes should override
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _extract_ground_truth() "
            "or override evaluate() entirely"
        )

    ## Training Metadata
    def _get_early_stopping_config(self) -> Dict[str, Any]:
        """Get early stopping configuration for models that support it.

        Returns:
            Dict[str, Any]: Dictionary containing early stopping parameters:
                - patience: Number of epochs without improvement before stopping.
                - threshold: Minimum change to qualify as an improvement.
                - metric: The metric to monitor for improvements.
                - greater_is_better: Whether higher metric values are better.
        """
        return {
            "patience": self.config.early_stopping_patience,
            "threshold": 0.0,
            "metric": self.config.metric_for_best_model,
            "greater_is_better": self.config.greater_is_better,
        }

    def _save_training_metadata(self, output_dir: str, metrics: Dict[str, Any]) -> None:
        """Save comprehensive training metadata to a JSON file.

        Captures detailed information about the training run to support:
        - Experiment tracking and comparison
        - Model registry with rich metadata
        - Reproducibility via configuration and git state capture
        - Debugging with complete environment details

        Args:
            output_dir: Directory where training_metadata.json will be saved.
            metrics: Dictionary of training and evaluation metrics to record.

        Note:
            Git information (commit, branch, dirty state) is captured if
            GitPython is installed and the code is in a git repository.
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

    # Dunder methods
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(config={self.config.model_type}, fitted={self.is_fitted})"

    def __del__(self):
        """Ensure cleanup happens even if not called explicitly."""
        try:
            self._cleanup_distributed()
        except Exception:
            # Don't raise exceptions in destructor
            pass


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
    elif model_type == "toto":
        from src.models.toto import TotoForecaster, TotoConfig

        config = TotoConfig(**config_dict)
        return TotoForecaster(config)
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
