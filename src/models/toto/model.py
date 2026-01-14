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
    """

    def __init__(
        self,
        data: torch.Tensor,
        targets: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ):
        self.data = data
        self.targets = targets
        self.padding_mask = (
            padding_mask
            if padding_mask is not None
            else torch.ones_like(data, dtype=torch.bool)
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "inputs": self.data[idx],
            "targets": self.targets[idx],
            "input_padding_mask": self.padding_mask[idx],
            "id_mask": torch.ones_like(self.data[idx]),
        }


class TotoForTrainer(torch.nn.Module):
    """Wrapper that makes Toto compatible with HuggingFace Trainer.

    HuggingFace Trainer expects model.forward() to return a dict with 'loss'.
    This wrapper:
    1. Concatenates context + targets for teacher forcing
    2. Runs Toto forward pass with causal attention
    3. Computes NLL loss on the forecast portion
    4. Returns {'loss': loss, 'logits': mean_predictions}
    """

    def __init__(
        self,
        toto_model,
        forecast_length: int,
        mse_weight: float = 0.1,
        target_variate_idx: int = 0,
    ):
        super().__init__()
        self.toto = toto_model
        self.forecast_length = forecast_length
        self.is_peft_model = hasattr(toto_model, "base_model")
        self.mse_weight = mse_weight  # Weight for MSE in composite loss
        self.target_variate_idx = target_variate_idx  # Index of target variate (0 = BG)

    def _get_toto_backbone(self):
        """Extract the underlying Toto backbone, handling PEFT-wrapped models."""
        if self.is_peft_model:
            if hasattr(self.toto, "get_base_model"):
                underlying_toto = self.toto.get_base_model()
            elif hasattr(self.toto, "base_model") and hasattr(
                self.toto.base_model, "model"
            ):
                underlying_toto = self.toto.base_model.model
            else:
                underlying_toto = self.toto
            return underlying_toto.model
        else:
            return self.toto.model

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        input_padding_mask: torch.Tensor,
        id_mask: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass that returns loss for Trainer."""
        # Teacher forcing: concatenate context + targets
        full_input = torch.cat([inputs, targets], dim=2)
        full_padding_mask = torch.cat(
            [input_padding_mask, torch.ones_like(targets, dtype=torch.bool)], dim=2
        )
        full_id_mask = torch.cat([id_mask, torch.ones_like(targets)], dim=2)

        # Check for NaN/Inf in inputs
        if torch.isnan(full_input).any() or torch.isinf(full_input).any():
            raise ValueError(
                f"NaN or Inf detected in input data. "
                f"NaN count: {torch.isnan(full_input).sum()}, "
                f"Inf count: {torch.isinf(full_input).sum()}"
            )

        # Forward through Toto backbone
        toto_backbone = self._get_toto_backbone()
        output = toto_backbone(
            inputs=full_input,
            input_padding_mask=full_padding_mask,
            id_mask=full_id_mask,
        )

        # Compute NLL loss on the forecast portion only
        # CRITICAL: Do NOT normalize inputs before log_prob - Toto handles this internally
        log_prob = output.distribution.log_prob(full_input)
        forecast_log_prob = log_prob[:, :, -self.forecast_length :]

        # IMPORTANT: Only compute loss on target variate (e.g., BG at index 0)
        # For multivariate input → univariate output, we don't want to train on exogenous features
        target_idx = self.target_variate_idx
        target_log_prob = forecast_log_prob[
            :, target_idx, :
        ]  # Shape: (batch, forecast_len)
        nll_loss = -target_log_prob.mean()

        # Compute MSE loss to align training with evaluation metric (RMSE)
        # This helps prevent the model from optimizing NLL at the expense of point predictions
        pred_mean = output.distribution.mean[
            :, target_idx, -self.forecast_length :
        ]  # Only target variate
        target_values = targets[:, target_idx, :]  # Only target variate
        mse_loss = torch.nn.functional.mse_loss(pred_mean, target_values)

        # Composite loss: NLL for probabilistic calibration + MSE for point accuracy
        # mse_weight controls the balance (0.1 = 10% MSE, 90% NLL)
        mse_weight = getattr(self, "mse_weight", 0.1)
        loss = nll_loss + mse_weight * mse_loss

        # Check for NaN in loss
        if torch.isnan(loss) or torch.isinf(loss):
            raise ValueError(
                f"NaN or Inf loss detected. This usually indicates:\n"
                f"1. Learning rate too high\n"
                f"2. Numerical instability (try disabling fp16)\n"
                f"3. Data preprocessing issues\n"
                f"Loss value: {loss.item()}"
            )

        # Return mean predictions as logits (only target variate)
        logits = output.distribution.mean[:, target_idx, -self.forecast_length :]

        return {"loss": loss, "logits": logits}


class TotoForecaster(BaseTSFM):
    """Toto forecaster implementation using the base TSFM framework.

    Toto is a transformer-based model for multivariate time series forecasting
    with probabilistic outputs (Student-t distribution) and LoRA fine-tuning support.
    """

    def __init__(self, config: TotoConfig, lora_config=None, distributed_config=None):
        if not isinstance(config, TotoConfig):
            essential_params = {
                "model_path": getattr(
                    config, "model_path", "Datadog/Toto-Open-Base-1.0"
                ),
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
        if hasattr(self.model, "base_model"):
            if hasattr(self.model, "get_base_model"):
                underlying_toto = self.model.get_base_model()
            else:
                underlying_toto = self.model.base_model.model
            return underlying_toto.model
        elif hasattr(self.model, "model"):
            return self.model.model
        else:
            raise ValueError(
                f"Cannot extract Toto backbone from model of type {type(self.model)}"
            )

    def _run_inference(
        self,
        data_loader: DataLoader,
        num_samples: int = 100,
        return_quantiles: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Run autoregressive inference using Toto's official forecaster."""
        self.model.eval()
        device = next(self.model.parameters()).device

        toto_backbone = self._get_toto_backbone()
        official_forecaster = TotoOfficialForecaster(toto_backbone)

        all_predictions = []
        all_q10 = []
        all_q90 = []
        all_targets = []

        with torch.no_grad():
            for batch in data_loader:
                inputs = batch["inputs"].to(device)
                padding_mask = batch["input_padding_mask"].to(device)
                id_mask = batch["id_mask"].to(device)

                batch_size_actual = inputs.shape[0]
                context_len = inputs.shape[2]
                dummy_ts = torch.arange(context_len, dtype=torch.float32).unsqueeze(0)
                dummy_ts = dummy_ts.expand(batch_size_actual, -1).to(device) * 300

                masked_inputs = MaskedTimeseries(
                    series=inputs,
                    padding_mask=padding_mask,
                    id_mask=id_mask,
                    timestamp_seconds=dummy_ts,
                    time_interval_seconds=torch.tensor([300.0], dtype=torch.float32).to(
                        device
                    ),
                )

                forecast = official_forecaster.forecast(
                    masked_inputs,
                    prediction_length=self.config.forecast_length,
                    num_samples=num_samples,
                    samples_per_batch=min(50, num_samples),
                )

                all_predictions.append(forecast.median.cpu().numpy())

                if return_quantiles:
                    all_q10.append(forecast.quantile(0.1).cpu().numpy())
                    all_q90.append(forecast.quantile(0.9).cpu().numpy())

                if "targets" in batch:
                    all_targets.append(batch["targets"].cpu().numpy())

        result = {"predictions": np.concatenate(all_predictions, axis=0)}

        if return_quantiles:
            result["q10"] = np.concatenate(all_q10, axis=0)
            result["q90"] = np.concatenate(all_q90, axis=0)

        if all_targets:
            result["targets"] = np.concatenate(all_targets, axis=0)

        return result

    def predict(
        self,
        data: Any,
        batch_size: Optional[int] = None,
        return_dict: bool = False,
        num_samples: int = 100,
    ) -> Union[np.ndarray, Dict[str, Any]]:
        """Make predictions using Toto's official autoregressive inference."""
        if self.model is None:
            raise ValueError("Model must be initialized before making predictions")

        batch_size = batch_size or self.config.batch_size

        # Prepare data loader
        if isinstance(data, pd.DataFrame):
            data_loader, _, _ = self._prepare_data(data)
        elif isinstance(data, torch.Tensor):
            dataset = TotoDataset(
                data=data,
                targets=torch.zeros(
                    data.shape[0], data.shape[1], self.config.forecast_length
                ),
            )
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        elif isinstance(data, DataLoader):
            data_loader = data
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

        result = self._run_inference(
            data_loader, num_samples=num_samples, return_quantiles=return_dict
        )

        if return_dict:
            return {
                "predictions": result["predictions"],
                "q10": result["q10"],
                "q90": result["q90"],
                "model_config": self.config.to_dict(),
                "n_samples": len(result["predictions"]),
            }

        return result["predictions"]

    def get_training_strategy(self) -> TrainingStrategy:
        return TrainingStrategy.TRANSFORMERS

    def supports_lora(self) -> bool:
        return True

    def _initialize_model(self) -> None:
        """Initialize the Toto model from pretrained weights."""
        try:
            info_print(f"Initializing Toto model from {self.config.model_path}")
            self.model = Toto.from_pretrained(self.config.model_path)

            if self.config.fit_strategy == "zero_shot":
                info_print("Freezing all parameters for zero-shot evaluation")
                for param in self.model.parameters():
                    param.requires_grad = False
            else:
                info_print(f"Enabling gradients ({self.config.fit_strategy} mode)")
                for param in self.model.parameters():
                    param.requires_grad = True

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
        """Prepare data loaders for Toto training."""
        info_print("Preparing data for Toto training...")

        def df_to_dataset(df: pd.DataFrame) -> TotoDataset:
            """Convert DataFrame to TotoDataset."""
            features = self.config.input_features
            context_len = self.config.context_length
            forecast_len = self.config.forecast_length
            total_len = context_len + forecast_len

            feature_data = df[features].values
            num_windows = len(feature_data) - total_len + 1
            if num_windows <= 0:
                raise ValueError(
                    f"Data length ({len(feature_data)}) is too short for "
                    f"context_length ({context_len}) + forecast_length ({forecast_len})"
                )

            windows = np.array(
                [feature_data[i : i + total_len] for i in range(num_windows)]
            )
            context = np.transpose(windows[:, :context_len, :], (0, 2, 1))
            forecast = np.transpose(windows[:, context_len:, :], (0, 2, 1))

            context_tensor = torch.tensor(context, dtype=torch.float32)
            forecast_tensor = torch.tensor(forecast, dtype=torch.float32)
            padding_mask = ~torch.isnan(context_tensor)

            # Per-feature bounds for proper normalization
            # Toto handles internal normalization, but we clamp extreme outliers
            feature_bounds = {
                "bg_mM": (0.0, 50.0),  # Blood glucose: 0-50 mM (physiological range)
                "iob": (0.0, 30.0),  # Insulin on board: 0-30 units
                "cob": (0.0, 200.0),  # Carbs on board: 0-200 grams
                "steps": (0.0, 10000.0),  # Steps: 0-10000 per 5-min interval
                "cals": (0.0, 1000.0),  # Calories: 0-1000 per interval
                "hr_bpm": (0.0, 250.0),  # Heart rate: 0-250 bpm
                "activity": (0.0, 100.0),  # Activity level: 0-100
            }

            # Apply per-feature bounds
            for i, feat in enumerate(features):
                min_val, max_val = feature_bounds.get(
                    feat, (0.0, 1e6)
                )  # Default wide range
                context_tensor[:, i, :] = torch.nan_to_num(
                    context_tensor[:, i, :], nan=0.0, posinf=max_val, neginf=0.0
                )
                forecast_tensor[:, i, :] = torch.nan_to_num(
                    forecast_tensor[:, i, :], nan=0.0, posinf=max_val, neginf=0.0
                )
                context_tensor[:, i, :] = torch.clamp(
                    context_tensor[:, i, :], min=min_val, max=max_val
                )
                forecast_tensor[:, i, :] = torch.clamp(
                    forecast_tensor[:, i, :], min=min_val, max=max_val
                )

            return TotoDataset(
                data=context_tensor, targets=forecast_tensor, padding_mask=padding_mask
            )

        # Load data from source name or use DataFrame directly
        if isinstance(train_data, str):
            from src.data.diabetes_datasets.data_loader import get_loader

            loader = get_loader(
                data_source_name=train_data, num_validation_days=20, use_cached=True
            )
            train_df = pd.concat(list(loader.train_data.values()), ignore_index=True)
            val_df = (
                pd.concat(list(loader.validation_data.values()), ignore_index=True)
                if loader.validation_data
                else None
            )
        else:
            train_df = train_data
            val_df = val_data

        test_df = test_data

        train_dataset = df_to_dataset(train_df)
        val_dataset = df_to_dataset(val_df) if val_df is not None else None
        test_dataset = df_to_dataset(test_df) if test_df is not None else None

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
        """Save Toto model weights."""
        if self.model is not None:
            model_path = os.path.join(output_dir, "toto_model")
            os.makedirs(model_path, exist_ok=True)
            self.model.save_pretrained(model_path)
            info_print(f"Toto model saved to {model_path}")

    def _load_model_weights(self, model_dir: str) -> None:
        """Load Toto model weights."""
        try:
            model_path = os.path.join(model_dir, "toto_model")
            if os.path.exists(model_path):
                self.model = Toto.from_pretrained(model_path)
            else:
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
        """Execute Toto training using HuggingFace Trainer."""
        train_loader, val_loader, test_loader = self._prepare_data(
            train_data, val_data, test_data
        )

        mse_weight = getattr(self.config, "mse_weight", 0.1)
        # Get target variate index (0 = first feature = BG by default)
        target_feature = getattr(self.config, "target_feature", "bg_mM")
        input_features = getattr(self.config, "input_features", ["bg_mM"])
        target_variate_idx = (
            input_features.index(target_feature)
            if target_feature in input_features
            else 0
        )
        trainer_model = TotoForTrainer(
            self.model,
            self.config.forecast_length,
            mse_weight=mse_weight,
            target_variate_idx=target_variate_idx,
        )

        has_val_data = val_loader is not None
        training_args = self._create_training_arguments(
            output_dir, has_val_data=has_val_data
        )

        trainer = Trainer(
            model=trainer_model,
            args=training_args,
            train_dataset=train_loader.dataset,
            eval_dataset=val_loader.dataset if val_loader else None,
            callbacks=self._get_callbacks(has_val_data=has_val_data),
        )

        is_lora = trainer_model.is_peft_model
        if is_lora:
            info_print("LoRA ENABLED: Training adapter layers only")
            trainable = sum(
                p.numel() for p in trainer_model.parameters() if p.requires_grad
            )
            total = sum(p.numel() for p in trainer_model.parameters())
            info_print(
                f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)"
            )
        else:
            info_print("Full fine-tuning: All parameters trainable")

        info_print("Starting training with HuggingFace Trainer")
        info_print(f"  Train samples: {len(train_loader.dataset)}")
        info_print(f"  Batch size: {self.config.batch_size}")
        info_print(f"  Epochs: {self.config.num_epochs}")
        info_print(f"  Learning rate: {self.config.learning_rate}")
        info_print(f"  LoRA: {is_lora}")

        resume_from_checkpoint = kwargs.get("resume_from_checkpoint", None)
        if resume_from_checkpoint:
            info_print(f"Resuming from checkpoint: {resume_from_checkpoint}")
            train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        else:
            train_result = trainer.train()

        # Save the final model
        final_model_dir = os.path.join(output_dir, "final_model")
        trainer.save_model(final_model_dir)

        # If using LoRA/PEFT, also save in PEFT format
        if trainer_model.is_peft_model:
            peft_save_dir = os.path.join(output_dir, "peft_adapter")
            info_print(f"Saving PEFT adapter to {peft_save_dir}")
            trainer_model.toto.save_pretrained(peft_save_dir)

        self.model = trainer_model.toto

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

    def _create_training_arguments(
        self, output_dir: str, has_val_data: bool = False
    ) -> TrainingArguments:
        """Create HuggingFace TrainingArguments for Toto training."""
        eval_strategy = self.config.eval_strategy if has_val_data else "no"

        base_args = {
            "output_dir": output_dir,
            "learning_rate": self.config.learning_rate,
            "num_train_epochs": self.config.num_epochs,
            "per_device_train_batch_size": self.config.batch_size,
            "per_device_eval_batch_size": self.config.batch_size,
            "warmup_steps": self.config.warmup_steps,
            "weight_decay": self.config.weight_decay,
            "max_grad_norm": self.config.gradient_clip_val,
            "logging_dir": os.path.join(output_dir, "logs"),
            "logging_steps": self.config.logging_steps,
            "eval_strategy": eval_strategy,
            "eval_steps": self.config.eval_steps if has_val_data else None,
            "save_steps": self.config.save_steps,
            "save_total_limit": 3,
            "load_best_model_at_end": has_val_data,
            "fp16": self.config.fp16 and torch.cuda.is_available(),
            "dataloader_num_workers": self.config.dataloader_num_workers,
            "remove_unused_columns": False,
            "report_to": "none",
            "label_names": ["targets", "input_padding_mask", "id_mask"],
            "metric_for_best_model": None,
            "greater_is_better": None,
        }

        distributed_args = self._get_distributed_training_args()
        base_args.update(distributed_args)

        return TrainingArguments(**base_args)

    def _get_distributed_training_args(self) -> Dict[str, Any]:
        """Get distributed training arguments."""
        if not self.distributed_config.enabled:
            return {}

        args = {}

        if self.distributed_config.strategy == "ddp":
            args["ddp_backend"] = self.distributed_config.backend
            args["ddp_find_unused_parameters"] = (
                self.distributed_config.find_unused_parameters
            )
            args["ddp_bucket_cap_mb"] = 25

        elif self.distributed_config.strategy == "deepspeed":
            if self.distributed_config.deepspeed_config:
                args["deepspeed"] = self.distributed_config.deepspeed_config

        elif self.distributed_config.strategy == "fsdp":
            if self.distributed_config.fsdp_config:
                args.update(self.distributed_config.fsdp_config)

        return args

    def _get_callbacks(self, has_val_data: bool = False) -> List:
        """Get training callbacks."""
        callbacks = []

        if self.config.early_stopping_patience > 0 and has_val_data:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=self.config.early_stopping_patience,
                    early_stopping_threshold=0.0,
                )
            )

        return callbacks

    def _evaluate_loader(self, data_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model on a data loader."""
        result = self._run_inference(data_loader, num_samples=20)
        preds = result["predictions"].flatten()
        targets = result["targets"].flatten()
        return self._compute_metrics(preds, targets)
