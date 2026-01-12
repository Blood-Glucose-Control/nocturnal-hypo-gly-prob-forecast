"""
Toto model implementation using the base TSFM framework.

This module provides a concrete implementation of Toto that inherits from
the base TSFM framework for fine-tuning on blood glucose forecasting.
"""

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback

from toto.model.toto import Toto
from toto.inference.forecaster import TotoForecaster as TotoOfficialForecaster
from toto.data.util.dataset import MaskedTimeseries

from src.models.base import BaseTSFM, TrainingStrategy
from src.models.toto.config import TotoConfig
from src.utils.logging_helper import info_print, error_print


class TotoDataset(Dataset):
    """PyTorch Dataset for Toto model training.

    Converts time series data into the format expected by Toto:
    - inputs: (batch, variates, timesteps)
    - input_padding_mask: (batch, variates, timesteps) boolean mask
    - id_mask: (batch, variates, timesteps) float mask for spacewise attention

    Attributes:
        data: Input data tensor of shape (num_samples, num_variates, context_length)
        targets: Target data tensor of shape (num_samples, num_variates, forecast_length)
        padding_mask: Boolean mask for padding positions
    """

    def __init__(
        self,
        data: torch.Tensor,
        targets: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ):
        """Initialize the dataset.

        Args:
            data: Input tensor of shape (num_samples, num_variates, context_length)
            targets: Target tensor of shape (num_samples, num_variates, forecast_length)
            padding_mask: Optional boolean mask for padding positions
        """
        self.data = data
        self.targets = targets
        self.padding_mask = padding_mask if padding_mask is not None else torch.ones_like(data, dtype=torch.bool)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {
            "inputs": self.data[idx],
            "targets": self.targets[idx],
            "input_padding_mask": self.padding_mask[idx],
            # id_mask is 1.0 for all variates (no masking for space-wise attention)
            "id_mask": torch.ones_like(self.data[idx]),
        }
        return item


class TotoForTrainer(torch.nn.Module):
    """Wrapper that makes Toto compatible with HuggingFace Trainer.

    HuggingFace Trainer expects model.forward() to return a dict with 'loss'.
    Toto returns TotoOutput(distribution, loc, scale). This wrapper:
    1. Concatenates context + targets to form full sequence (teacher forcing)
    2. Runs Toto forward pass with causal attention (each position sees only past)
    3. Computes NLL loss on the forecast portion of the predicted distribution
    4. Returns {'loss': loss, 'logits': mean_predictions}

    Teacher forcing is the correct approach for Toto because:
    - Toto is a decoder-only model trained with causal next-patch prediction
    - Causal attention ensures each forecast step only sees context + previous steps
    - At inference, use autoregressive generation (Toto's official forecaster API)

    Attributes:
        toto: The underlying Toto model (may be PEFT-wrapped for LoRA)
        forecast_length: Number of timesteps in the forecast horizon
        is_peft_model: Whether the model is wrapped with PEFT (LoRA)
    """

    def __init__(self, toto_model, forecast_length: int):
        """Initialize the wrapper.

        Args:
            toto_model: The Toto model instance (raw Toto or PEFT-wrapped)
            forecast_length: Length of forecast horizon for loss computation
        """
        super().__init__()
        self.toto = toto_model
        self.forecast_length = forecast_length

        # Check if this is a PEFT model (LoRA)
        self.is_peft_model = hasattr(toto_model, 'base_model')

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        input_padding_mask: torch.Tensor,
        id_mask: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass that returns loss for Trainer.

        Args:
            inputs: Context tensor (batch, variates, context_length)
            targets: Target tensor (batch, variates, forecast_length)
            input_padding_mask: Mask for padding (batch, variates, context_length)
            id_mask: Mask for spacewise attention (batch, variates, context_length)

        Returns:
            Dict with 'loss' (scalar) and 'logits' (mean predictions)
        """
        # Teacher forcing: concatenate context + targets for training
        # Toto uses causal attention, so each position only sees previous positions
        # This matches how Toto was pretrained (next-patch prediction)
        full_input = torch.cat([inputs, targets], dim=2)
        full_padding_mask = torch.cat([
            input_padding_mask,
            torch.ones_like(targets, dtype=torch.bool)
        ], dim=2)
        full_id_mask = torch.cat([
            id_mask,
            torch.ones_like(targets)
        ], dim=2)

        # Check for NaN/Inf in inputs (helps debug data issues)
        if torch.isnan(full_input).any() or torch.isinf(full_input).any():
            raise ValueError(
                f"NaN or Inf detected in input data. "
                f"NaN count: {torch.isnan(full_input).sum()}, "
                f"Inf count: {torch.isinf(full_input).sum()}"
            )

        # Forward through Toto backbone (handles both raw and PEFT-wrapped models)
        if self.is_peft_model:
            # Extract underlying Toto from PEFT wrapper
            if hasattr(self.toto, 'get_base_model'):
                underlying_toto = self.toto.get_base_model()
            elif hasattr(self.toto, 'base_model') and hasattr(self.toto.base_model, 'model'):
                underlying_toto = self.toto.base_model.model
            else:
                underlying_toto = self.toto
            toto_backbone = underlying_toto.model
        else:
            toto_backbone = self.toto.model

        output = toto_backbone(
            inputs=full_input,
            input_padding_mask=full_padding_mask,
            id_mask=full_id_mask,
        )

        # Compute NLL loss on the forecast portion only
        # output.distribution covers all timesteps, we only penalize forecast
        log_prob = output.distribution.log_prob(full_input)
        forecast_log_prob = log_prob[:, :, -self.forecast_length:]
        loss = -forecast_log_prob.mean()

        # Check for NaN in loss (indicates numerical instability)
        if torch.isnan(loss) or torch.isinf(loss):
            raise ValueError(
                f"NaN or Inf loss detected. This usually indicates:\n"
                f"1. Learning rate too high\n"
                f"2. Numerical instability (try disabling fp16)\n"
                f"3. Data preprocessing issues\n"
                f"Loss value: {loss.item()}"
            )

        # Return mean predictions as logits (for potential metric computation)
        logits = output.distribution.mean[:, :, -self.forecast_length:]

        return {"loss": loss, "logits": logits}


class TotoForecaster(BaseTSFM):
    """Toto forecaster implementation using the base TSFM framework.

    Toto (Timeseries-Optimized Transformer for Observability) is a transformer-based
    model for multivariate time series forecasting. It uses patch embeddings, rotary
    positional encodings, and alternating time-wise/space-wise attention.

    Key features:
    - Probabilistic outputs (Student-t distribution)
    - Supports LoRA fine-tuning (transformer-based architecture)
    - Uses NLL loss for training

    Attributes:
        config: Toto-specific configuration (TotoConfig instance).
        model: The Toto model instance.

    Example:
        >>> config = TotoConfig(
        ...     model_path="Datadog/Toto-Open-Base-1.0",
        ...     context_length=1024,
        ...     forecast_length=72,
        ... )
        >>> model = TotoForecaster(config)
        >>> model.fit(train_data)
        >>> predictions = model.predict(test_data)
    """

    def __init__(self, config: TotoConfig, lora_config=None, distributed_config=None):
        """Initialize the Toto forecaster.

        Args:
            config: Toto configuration object.
            lora_config: LoRA configuration for memory-efficient fine-tuning.
            distributed_config: Configuration for distributed training.
        """
        if not isinstance(config, TotoConfig):
            essential_params = {
                "model_path": getattr(config, "model_path", "Datadog/Toto-Open-Base-1.0"),
                "context_length": getattr(config, "context_length", 1024),
                "forecast_length": getattr(config, "forecast_length", 72),
                "learning_rate": getattr(config, "learning_rate", 1e-5),
                "batch_size": getattr(config, "batch_size", 32),
                "num_epochs": getattr(config, "num_epochs", 10),
            }
            config = TotoConfig(**essential_params)

        super().__init__(config, lora_config, distributed_config)
        self.config: TotoConfig = self.config

    def _get_toto_backbone(self):
        """Extract the Toto backbone model, handling both raw and PEFT-wrapped models."""
        if hasattr(self.model, 'base_model'):
            # PEFT-wrapped model
            if hasattr(self.model, 'get_base_model'):
                underlying_toto = self.model.get_base_model()
            else:
                underlying_toto = self.model.base_model.model
            return underlying_toto.model
        elif hasattr(self.model, 'model'):
            return self.model.model
        else:
            raise ValueError(f"Cannot extract Toto backbone from model of type {type(self.model)}")

    # Abstract method implementations
    def predict(
        self,
        data: Any,
        batch_size: Optional[int] = None,
        return_dict: bool = False,
        num_samples: int = 100,
    ) -> Union[np.ndarray, Dict[str, Any]]:
        """Make predictions on new data using Toto's official autoregressive inference.

        Uses Toto's official TotoForecaster API which performs autoregressive
        generation - matching how Toto was pretrained. This is critical for
        proper forecasting performance.

        Args:
            data: Input data - can be:
                - pd.DataFrame with time series data
                - torch.Tensor of shape (batch, variates, timesteps)
                - TotoDataset instance
            batch_size: Batch size for prediction (defaults to config.batch_size)
            return_dict: Whether to return additional information (quantiles, samples)
            num_samples: Number of samples for probabilistic forecasting (default: 100)

        Returns:
            If return_dict=False: numpy array of predictions (median forecasts)
            If return_dict=True: dict with predictions, quantiles, and metadata
        """
        if self.model is None:
            raise ValueError("Model must be initialized before making predictions")

        batch_size = batch_size or self.config.batch_size
        self.model.eval()
        device = next(self.model.parameters()).device

        # Create official Toto forecaster for autoregressive inference
        toto_backbone = self._get_toto_backbone()
        official_forecaster = TotoOfficialForecaster(toto_backbone)

        # Prepare data
        if isinstance(data, pd.DataFrame):
            data_loader, _, _ = self._prepare_data(data)
        elif isinstance(data, torch.Tensor):
            dataset = TotoDataset(
                data=data,
                targets=torch.zeros(data.shape[0], data.shape[1], self.config.forecast_length),
            )
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        elif isinstance(data, DataLoader):
            data_loader = data
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

        all_predictions = []
        all_q10 = []
        all_q90 = []

        with torch.no_grad():
            for batch in data_loader:
                inputs = batch["inputs"].to(device)
                padding_mask = batch["input_padding_mask"].to(device)
                id_mask = batch["id_mask"].to(device)

                # Create MaskedTimeseries for official API
                # Note: timestamp handling is simplified - using dummy timestamps
                batch_size_actual = inputs.shape[0]
                context_len = inputs.shape[2]

                # Create dummy timestamps (not critical for forecasting)
                dummy_ts = torch.arange(context_len, dtype=torch.float32).unsqueeze(0)
                dummy_ts = dummy_ts.expand(batch_size_actual, -1).to(device) * 300  # 5-min intervals

                masked_inputs = MaskedTimeseries(
                    series=inputs,
                    padding_mask=padding_mask,
                    id_mask=id_mask,
                    timestamp_seconds=dummy_ts,
                    time_interval_seconds=torch.tensor([300.0], dtype=torch.float32).to(device),
                )

                # Use official autoregressive forecasting
                forecast = official_forecaster.forecast(
                    masked_inputs,
                    prediction_length=self.config.forecast_length,
                    num_samples=num_samples,
                    samples_per_batch=min(50, num_samples),
                )

                # Extract median as point prediction
                median_pred = forecast.median.cpu().numpy()  # (batch, variates, forecast_len)
                all_predictions.append(median_pred)

                # Also get quantiles for uncertainty
                all_q10.append(forecast.quantile(0.1).cpu().numpy())
                all_q90.append(forecast.quantile(0.9).cpu().numpy())

        predictions = np.concatenate(all_predictions, axis=0)

        if return_dict:
            return {
                "predictions": predictions,
                "q10": np.concatenate(all_q10, axis=0),
                "q90": np.concatenate(all_q90, axis=0),
                "model_config": self.config.to_dict(),
                "n_samples": len(predictions),
            }

        return predictions

    def get_training_strategy(self) -> TrainingStrategy:
        """Return the training strategy used by Toto.

        Returns:
            TrainingStrategy: TRANSFORMERS (Toto uses custom PyTorch training
                with HuggingFace-style components).
        """
        return TrainingStrategy.TRANSFORMERS

    def supports_lora(self) -> bool:
        """Check if Toto supports LoRA fine-tuning.

        Returns:
            bool: True - Toto is transformer-based and supports LoRA.
        """
        return True

    def _initialize_model(self) -> None:
        """Initialize the Toto model from pretrained weights.

        Loads the model from HuggingFace and configures gradients based on
        fit_strategy.
        """
        try:
            info_print(f"Initializing Toto model from {self.config.model_path}")

            # Load pretrained Toto model
            self.model = Toto.from_pretrained(self.config.model_path)

            # Configure gradients
            if self.config.fit_strategy == "zero_shot":
                info_print("Freezing all parameters for zero-shot evaluation")
                for param in self.model.parameters():
                    param.requires_grad = False
            else:
                info_print(f"Enabling gradients ({self.config.fit_strategy} mode)")
                for param in self.model.parameters():
                    param.requires_grad = True

            # Move to appropriate device
            if torch.cuda.is_available() and not self.config.use_cpu:
                self.model = self.model.cuda()
                info_print("Moved model to CUDA")

            info_print("Toto model initialized successfully")

        except Exception as e:
            error_print(f"Failed to initialize Toto model: {str(e)}")
            raise

    def _prepare_data(
        self,
        train_data: Any,
        val_data: Optional[Any] = None,
        test_data: Optional[Any] = None,
    ) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
        """Prepare data loaders for Toto training.

        Converts DataFrame to Toto's expected format: (batch, variates, timesteps).

        Args:
            train_data: Training data as DataFrame or data source name
            val_data: Validation data (optional)
            test_data: Test data (optional)

        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        info_print("Preparing data for Toto training...")

        def df_to_dataset(df: pd.DataFrame) -> TotoDataset:
            """Convert DataFrame to TotoDataset."""
            # Extract features and create windows
            features = self.config.input_features
            context_len = self.config.context_length
            forecast_len = self.config.forecast_length
            total_len = context_len + forecast_len

            # Get the feature columns
            feature_data = df[features].values  # (timesteps, num_features)

            # Create sliding windows
            num_windows = len(feature_data) - total_len + 1
            if num_windows <= 0:
                raise ValueError(
                    f"Data length ({len(feature_data)}) is too short for "
                    f"context_length ({context_len}) + forecast_length ({forecast_len})"
                )

            # Create windows: (num_windows, total_len, num_features)
            windows = np.array([
                feature_data[i:i + total_len]
                for i in range(num_windows)
            ])

            # Split into context and forecast
            context = windows[:, :context_len, :]  # (num_windows, context_len, num_features)
            forecast = windows[:, context_len:, :]  # (num_windows, forecast_len, num_features)

            # Transpose to Toto format: (batch, variates, timesteps)
            context = np.transpose(context, (0, 2, 1))
            forecast = np.transpose(forecast, (0, 2, 1))

            # Convert to tensors
            context_tensor = torch.tensor(context, dtype=torch.float32)
            forecast_tensor = torch.tensor(forecast, dtype=torch.float32)

            # Create padding mask (True where data is valid, i.e., not NaN)
            padding_mask = ~torch.isnan(context_tensor)

            # Replace NaNs and clip extreme values for numerical stability
            # Blood glucose typically ranges 2-25 mM, clip to reasonable bounds
            context_tensor = torch.nan_to_num(context_tensor, nan=0.0, posinf=50.0, neginf=0.0)
            forecast_tensor = torch.nan_to_num(forecast_tensor, nan=0.0, posinf=50.0, neginf=0.0)

            # Clip to prevent extreme outliers that could cause numerical instability
            context_tensor = torch.clamp(context_tensor, min=0.0, max=50.0)
            forecast_tensor = torch.clamp(forecast_tensor, min=0.0, max=50.0)

            return TotoDataset(
                data=context_tensor,
                targets=forecast_tensor,
                padding_mask=padding_mask,
            )

        # Handle string data source names
        if isinstance(train_data, str):
            from src.data.diabetes_datasets.data_loader import get_loader
            loader = get_loader(
                data_source_name=train_data,
                num_validation_days=20,
                use_cached=True,
            )
            # Get training data from the loader's train split
            combined_train = []
            for patient_id, patient_data in loader.train_data.items():
                combined_train.append(patient_data)
            train_data = pd.concat(combined_train, ignore_index=True)

            # Get validation data from the loader's validation split
            if loader.validation_data and len(loader.validation_data) > 0:
                combined_val = []
                for patient_id, patient_data in loader.validation_data.items():
                    combined_val.append(patient_data)
                val_data = pd.concat(combined_val, ignore_index=True)

        # Create datasets
        train_dataset = df_to_dataset(train_data)

        val_dataset = None
        if val_data is not None:
            if isinstance(val_data, str):
                from src.data.diabetes_datasets.data_loader import get_loader
                loader = get_loader(data_source_name=val_data, use_cached=True)
                combined_data = []
                for patient_id, patient_data in loader.processed_data.items():
                    combined_data.append(patient_data)
                val_data = pd.concat(combined_data, ignore_index=True)
            val_dataset = df_to_dataset(val_data)

        test_dataset = None
        if test_data is not None:
            if isinstance(test_data, str):
                from src.data.diabetes_datasets.data_loader import get_loader
                loader = get_loader(data_source_name=test_data, use_cached=True)
                combined_data = []
                for patient_id, patient_data in loader.processed_data.items():
                    combined_data.append(patient_data)
                test_data = pd.concat(combined_data, ignore_index=True)
            test_dataset = df_to_dataset(test_data)

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.dataloader_num_workers,
        )

        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.dataloader_num_workers,
            )

        test_loader = None
        if test_dataset is not None:
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.dataloader_num_workers,
            )

        info_print("Data preparation complete:")
        info_print(f"  Train samples: {len(train_dataset)}")
        info_print(f"  Val samples: {len(val_dataset) if val_dataset else 0}")
        info_print(f"  Test samples: {len(test_dataset) if test_dataset else 0}")

        return train_loader, val_loader, test_loader

    def _save_model_weights(self, output_dir: str) -> None:
        """Save Toto model weights.

        Args:
            output_dir: Directory to save model weights.
        """
        if self.model is not None:
            model_path = os.path.join(output_dir, "toto_model")
            os.makedirs(model_path, exist_ok=True)
            self.model.save_pretrained(model_path)
            info_print(f"Toto model saved to {model_path}")

    def _load_model_weights(self, model_dir: str) -> None:
        """Load Toto model weights.

        Args:
            model_dir: Directory containing saved model weights.
        """
        try:
            model_path = os.path.join(model_dir, "toto_model")
            if os.path.exists(model_path):
                self.model = Toto.from_pretrained(model_path)
            else:
                # Try loading directly from model_dir
                self.model = Toto.from_pretrained(model_dir)

            if torch.cuda.is_available() and not self.config.use_cpu:
                self.model = self.model.cuda()

            info_print(f"Toto model weights loaded from {model_dir}")

        except Exception as e:
            error_print(f"Failed to load model weights: {str(e)}")
            raise

    def _train_model(
        self,
        train_data: Any,
        val_data: Optional[Any] = None,
        test_data: Optional[Any] = None,
        output_dir: str = "./output",
        **kwargs,
    ) -> Dict[str, Any]:
        """Execute Toto training using HuggingFace Trainer.

        Uses the Trainer class for:
        - Automatic distributed training support
        - Checkpointing and logging
        - Early stopping
        - Mixed precision (fp16)

        Args:
            train_data: Training data
            val_data: Validation data (optional)
            test_data: Test data (optional)
            output_dir: Directory for saving checkpoints
            **kwargs: Additional arguments (e.g., resume_from_checkpoint)

        Returns:
            Dictionary with training and evaluation metrics.
        """
        # Prepare datasets (not loaders - Trainer handles batching)
        train_loader, val_loader, test_loader = self._prepare_data(
            train_data, val_data, test_data
        )

        # Wrap Toto model for Trainer compatibility
        # TotoForTrainer computes NLL loss internally and returns {'loss': ..., 'logits': ...}
        trainer_model = TotoForTrainer(self.model, self.config.forecast_length)

        # Create training arguments
        has_val_data = val_loader is not None
        training_args = self._create_training_arguments(output_dir, has_val_data=has_val_data)

        # Create Trainer
        # Note: We don't specify data_collator - the default collator handles
        # dict-of-tensors correctly (stacks each tensor in the batch)
        trainer = Trainer(
            model=trainer_model,
            args=training_args,
            train_dataset=train_loader.dataset,
            eval_dataset=val_loader.dataset if val_loader else None,
            callbacks=self._get_callbacks(has_val_data=has_val_data),
        )

        # Log whether LoRA is being used
        is_lora = trainer_model.is_peft_model
        if is_lora:
            info_print("LoRA ENABLED: Training adapter layers only")
            trainable = sum(p.numel() for p in trainer_model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in trainer_model.parameters())
            info_print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
        else:
            info_print("Full fine-tuning: All parameters trainable")

        info_print("Starting training with HuggingFace Trainer")
        info_print(f"  Train samples: {len(train_loader.dataset)}")
        info_print(f"  Batch size: {self.config.batch_size}")
        info_print(f"  Epochs: {self.config.num_epochs}")
        info_print(f"  Learning rate: {self.config.learning_rate}")
        info_print(f"  LoRA: {is_lora}")

        # Train the model
        resume_from_checkpoint = kwargs.get("resume_from_checkpoint", None)
        if resume_from_checkpoint:
            info_print(f"Resuming from checkpoint: {resume_from_checkpoint}")
            train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        else:
            train_result = trainer.train()

        # Save the final model
        final_model_dir = os.path.join(output_dir, "final_model")
        trainer.save_model(final_model_dir)

        # If using LoRA/PEFT, also save in PEFT format for easy loading
        if trainer_model.is_peft_model:
            peft_save_dir = os.path.join(output_dir, "peft_adapter")
            info_print(f"Saving PEFT adapter to {peft_save_dir}")
            trainer_model.toto.save_pretrained(peft_save_dir)
            info_print("PEFT adapter saved! Load with: PeftModel.from_pretrained(base_model, peft_save_dir)")

        # Update self.model with trained weights
        # The wrapper's toto attribute has the trained weights
        self.model = trainer_model.toto

        # Test evaluation
        test_metrics = {}
        if test_loader is not None:
            info_print("Evaluating on test set...")
            test_metrics = self._evaluate_loader(test_loader)
            info_print(f"Test metrics: {test_metrics}")

        info_print("Training complete!")

        return {
            "train_metrics": train_result.metrics,
            "test_metrics": test_metrics,
        }

    def _collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate samples into batches for Toto.

        Called by Trainer to combine individual samples into a batch.

        Args:
            batch: List of samples from TotoDataset.__getitem__()

        Returns:
            Dictionary with stacked tensors for inputs, targets, masks
        """
        return {
            "inputs": torch.stack([b["inputs"] for b in batch]),
            "targets": torch.stack([b["targets"] for b in batch]),
            "input_padding_mask": torch.stack([b["input_padding_mask"] for b in batch]),
            "id_mask": torch.stack([b["id_mask"] for b in batch]),
        }

    def _create_training_arguments(self, output_dir: str, has_val_data: bool = False) -> TrainingArguments:
        """Create HuggingFace TrainingArguments for Toto training.

        Configures all training hyperparameters including:
        - Learning rate and scheduling
        - Batch size and gradient accumulation
        - Evaluation and logging frequency
        - Checkpointing and early stopping
        - Mixed precision (fp16)

        Args:
            output_dir: Directory for saving checkpoints and logs
            has_val_data: Whether validation data is available

        Returns:
            Configured TrainingArguments instance
        """
        # Determine evaluation strategy based on validation data availability
        eval_strategy = self.config.eval_strategy if has_val_data else "no"

        # Get LR scheduler type from config (default to cosine for fine-tuning)
        lr_scheduler_type = getattr(self.config, 'lr_scheduler_type', 'cosine')

        # Base training arguments
        base_args = {
            "output_dir": output_dir,
            "learning_rate": self.config.learning_rate,
            "num_train_epochs": self.config.num_epochs,
            "per_device_train_batch_size": self.config.batch_size,
            "per_device_eval_batch_size": self.config.batch_size,
            "warmup_steps": self.config.warmup_steps,
            "warmup_ratio": getattr(self.config, 'warmup_ratio', 0.0),  # Alternative to warmup_steps
            "lr_scheduler_type": lr_scheduler_type,  # cosine, linear, constant, etc.
            "weight_decay": self.config.weight_decay,
            "max_grad_norm": self.config.gradient_clip_val,
            "logging_dir": os.path.join(output_dir, "logs"),
            "logging_steps": self.config.logging_steps,
            "eval_strategy": eval_strategy,
            "eval_steps": self.config.eval_steps if has_val_data else None,
            "save_steps": self.config.save_steps,
            "save_total_limit": 3,  # Keep only last 3 checkpoints
            "load_best_model_at_end": has_val_data,  # Only load best model if we have validation data
            "fp16": self.config.fp16 and torch.cuda.is_available(),
            "dataloader_num_workers": self.config.dataloader_num_workers,
            "remove_unused_columns": False,  # Keep all columns for Toto
            "report_to": "none",  # Disable wandb/tensorboard by default
            # Tell Trainer which keys are labels (needed for evaluation loss computation)
            "label_names": ["targets", "input_padding_mask", "id_mask"],
            # Explicitly set metric_for_best_model to None to disable it
            # The Trainer will automatically track loss without needing this
            "metric_for_best_model": None,
            "greater_is_better": None,
        }

        # Add distributed training arguments if configured
        distributed_args = self._get_distributed_training_args()
        base_args.update(distributed_args)

        return TrainingArguments(**base_args)

    def _get_distributed_training_args(self) -> Dict[str, Any]:
        """Get distributed training arguments for TrainingArguments.

        Configures DDP, DeepSpeed, or FSDP based on distributed_config.

        Returns:
            Dictionary of distributed training arguments
        """
        if not self.distributed_config.enabled:
            return {}

        args = {}

        if self.distributed_config.strategy == "ddp":
            args["ddp_backend"] = self.distributed_config.backend
            args["ddp_find_unused_parameters"] = self.distributed_config.find_unused_parameters
            args["ddp_bucket_cap_mb"] = 25

        elif self.distributed_config.strategy == "deepspeed":
            if self.distributed_config.deepspeed_config:
                args["deepspeed"] = self.distributed_config.deepspeed_config

        elif self.distributed_config.strategy == "fsdp":
            if self.distributed_config.fsdp_config:
                args.update(self.distributed_config.fsdp_config)

        return args

    def _get_callbacks(self, has_val_data: bool = False) -> List:
        """Get training callbacks for Trainer.

        Args:
            has_val_data: Whether validation data is available

        Returns:
            List of callback instances (e.g., EarlyStoppingCallback)
        """
        callbacks = []

        # Early stopping if patience > 0 and we have validation data
        # Early stopping requires validation metrics to monitor
        if self.config.early_stopping_patience > 0 and has_val_data:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=self.config.early_stopping_patience,
                    early_stopping_threshold=0.0,
                )
            )

        return callbacks

    def _evaluate_loader(self, data_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model on a data loader.

        Args:
            data_loader: DataLoader to evaluate on.

        Returns:
            Dictionary of metrics.
        """
        self.model.eval()
        device = next(self.model.parameters()).device

        # Use official Toto forecaster for autoregressive inference
        toto_backbone = self._get_toto_backbone()
        official_forecaster = TotoOfficialForecaster(toto_backbone)

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in data_loader:
                inputs = batch["inputs"].to(device)
                targets = batch["targets"].to(device)
                padding_mask = batch["input_padding_mask"].to(device)
                id_mask = batch["id_mask"].to(device)

                # Create MaskedTimeseries for official API
                batch_size_actual = inputs.shape[0]
                context_len = inputs.shape[2]

                # Create dummy timestamps (5-min intervals)
                dummy_ts = torch.arange(context_len, dtype=torch.float32).unsqueeze(0)
                dummy_ts = dummy_ts.expand(batch_size_actual, -1).to(device) * 300

                masked_inputs = MaskedTimeseries(
                    series=inputs,
                    padding_mask=padding_mask,
                    id_mask=id_mask,
                    timestamp_seconds=dummy_ts,
                    time_interval_seconds=torch.tensor([300.0], dtype=torch.float32).to(device),
                )

                # Use official autoregressive forecasting (fewer samples for speed)
                forecast = official_forecaster.forecast(
                    masked_inputs,
                    prediction_length=self.config.forecast_length,
                    num_samples=20,  # Fewer samples for faster evaluation
                    samples_per_batch=20,
                )

                # Use median as point prediction
                forecast_pred = forecast.median.cpu().numpy()

                all_preds.append(forecast_pred)
                all_targets.append(targets.cpu().numpy())

        preds = np.concatenate(all_preds, axis=0).flatten()
        targets = np.concatenate(all_targets, axis=0).flatten()

        return self._compute_metrics(preds, targets)
