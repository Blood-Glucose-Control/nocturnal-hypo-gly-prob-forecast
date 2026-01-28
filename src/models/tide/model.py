"""
TiDE (Time series dense encoder) model implementation using the base TSFM framework.

This module provides a concrete implementation of TiDE that inherits from
the base TSFM framework, demonstrating how to integrate existing models.
"""

import os
import logging
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import (
    TrainingArguments,
    Trainer,
)
import torch
import torch.nn as nn
# Local imports
from src.models.base import BaseTimeSeriesFoundationModel, TrainingBackend
from src.models.tide.config import TiDEConfig
from src.data.models import ColumnNames
from src.data.preprocessing.split_or_combine_patients import (
    reduce_features_multi_patient,
)
from src.utils.logging_helper import info_print, debug_print, error_print

class TiDEWindowedDataset(Dataset):
    """Dataset for TiDE model with proper windowing for time series prediction"""
    
    def __init__(self, data_loader_instance, lookback_steps=480, horizon_steps=120, 
                 dataset_type='train', stride=60):
        """
        Args:
            data_loader_instance: Instance of BrisT1DDataLoader
            lookback_steps: Number of timesteps for lookback (480 = 8 hours if 1 min resolution)
            horizon_steps: Number of timesteps for prediction (120 = 2 hours)
            dataset_type: 'train', 'validation', or 'test'
            stride: Step size for sliding window (60 = 1 hour)
        """
        self.lookback = lookback_steps
        self.horizon = horizon_steps
        self.stride = stride
        
        # Get the appropriate data
        if dataset_type == 'train' and data_loader_instance.train_data:
            self.data_dict = data_loader_instance.train_data
        elif dataset_type == 'validation' and data_loader_instance.validation_data:
            self.data_dict = data_loader_instance.validation_data
        else:
            self.data_dict = data_loader_instance.processed_data
        
        # Priority features for glucose prediction
        self.feature_cols = ['bg_mM', 'food_g', 'dose_units', 'iob', 'cob']
        self.target_col = 'bg_mM'  # What we're predicting
        
        # Validate columns exist
        first_value = next(iter(self.data_dict.values()))
        
        # Handle nested dict structure (get first DataFrame from nested dict)
        if isinstance(first_value, dict):
            first_df = next(iter(first_value.values()))
        else:
            first_df = first_value
        
        self.feature_cols = [col for col in self.feature_cols if col in first_df.columns]
        
        # Prepare valid windows
        self.windows = []
        self.prepare_windows()
        
        print(f"Created TiDE dataset:")
        print(f"  - Lookback: {lookback_steps} steps")
        print(f"  - Horizon: {horizon_steps} steps")
        print(f"  - Features: {self.feature_cols}")
        print(f"  - Total windows: {len(self.windows)}")
    
    def prepare_windows(self):
        """Prepare all valid windows from all patients, handling nested dict structures"""
        for patient_id, patient_data in self.data_dict.items():
            # Handle nested dictionary structure (e.g., test data: {patient_id: {session_id: df}})
            if isinstance(patient_data, dict):
                for session_id, patient_df in patient_data.items():
                    self._process_patient_df(patient_df, f"{patient_id}_{session_id}")
            # Handle flat dictionary structure (e.g., train data: {patient_id: df})
            elif isinstance(patient_data, pd.DataFrame):
                self._process_patient_df(patient_data, patient_id)

    def _process_patient_df(self, patient_df, patient_key):
        """Process a single patient dataframe and create windows"""
        n_samples = len(patient_df)
        min_length = self.lookback + self.horizon
        if n_samples >= min_length:
            windows_count = 0
            for i in range(self.lookback, n_samples - self.horizon + 1, self.stride):
                self.windows.append((patient_key, i))
                windows_count += 1
            if windows_count > 0:
                print(f"  Patient {patient_key}: {len(patient_df)} samples -> {windows_count} windows")
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        patient_id, end_lookback_idx = self.windows[idx]
        patient_df = self.data_dict[patient_id]
        
        # Get lookback window (input sequence)
        lookback_start = end_lookback_idx - self.lookback
        lookback_data = patient_df.iloc[lookback_start:end_lookback_idx]
        
        # Get prediction horizon (target sequence)
        horizon_data = patient_df.iloc[end_lookback_idx:end_lookback_idx + self.horizon]
        
        # Extract features
        seq_x = lookback_data[self.feature_cols].values.astype(np.float32)
        
        # For target, typically just predict glucose (bg_mM)
        # You can modify this based on what TiDE expects
        seq_y = horizon_data[[self.target_col]].values.astype(np.float32)
        
        # Handle NaN values
        seq_x = np.nan_to_num(seq_x, nan=0.0)
        seq_y = np.nan_to_num(seq_y, nan=0.0)
        
        # Convert to tensors
        seq_x = torch.tensor(seq_x, dtype=torch.float32)
        seq_y = torch.tensor(seq_y, dtype=torch.float32)
        
        # Time features (can be enhanced with hour of day, day of week, etc.)
        seq_x_mark = torch.arange(self.lookback, dtype=torch.float32).reshape(-1, 1)
        seq_y_mark = torch.arange(self.horizon, dtype=torch.float32).reshape(-1, 1)
        
        return seq_x, seq_y, seq_x_mark, seq_y_mark


class TiDEModel(BaseTimeSeriesFoundationModel):
    def __init__(
        self,
        config: TiDEConfig,
    ):
        """Initialize TiDE forecaster with configuration."""
        super().__init__(config)
        
        # Override config type for type hints
        self.config: TiDEConfig = config
        
        # TiDE-specific attributes
        self.criterion = self._get_loss_function()
        self.optimizer = None
        self.scheduler = None
        
    @property
    def training_backend(self) -> TrainingBackend:
        """TiDE uses PyTorch native training."""
        return TrainingBackend.PYTORCH

    @property
    def supports_lora(self) -> bool:
        """TiDE is MLP-based, not transformer-based, so it doesn't support LoRA."""
        return False
    
    def _initialize_model(self) -> None:
        """Initialize the TiDE model architecture."""
        # Determine input shapes based on config
        # These will be set properly when we see the actual data
        self.sizes = {
            'lookback': (self.config.seq_len, 1),  # Will be updated with actual feature count
            'attr': (1, 0),  # Static attributes - can be extended
            'dynCov': (self.config.seq_len + self.config.forecast_length, 1)  # Dynamic covariates
        }
        
        # Model will be fully initialized in _prepare_training_data
        # once we know the actual feature dimensions
        self.model = None

    def _create_model_from_sizes(self, sizes: Dict[str, Tuple]) -> TiDEModel:
        """Create TiDE model with proper sizes."""
        # Create a simple config object for TiDE
        ## get sizes and args
        model = TiDEModel(sizes, args)
        
        # Move to appropriate device
        device = self._get_device()
        model = model.to(device)
        
        return model

    def _get_device(self) -> torch.device:
        """Get the appropriate device for training/inference."""
        if self.config.use_cpu:
            return torch.device('cpu')
        elif torch.cuda.is_available():
            if self.distributed_config.enabled:
                return torch.device(f'cuda:{self.distributed_config.local_rank}')
            return torch.device('cuda')
        return torch.device('cpu')
    
    def _get_loss_function(self) -> nn.Module:
        """Get the configured loss function."""
        loss_functions = {
            'mse': nn.MSELoss(),
            'mae': nn.L1Loss(),
            'huber': nn.SmoothL1Loss(),
        }
        
        loss_fn = loss_functions.get(self.config.loss_function)
        if loss_fn is None:
            raise ValueError(f"Unsupported loss function: {self.config.loss_function}")
        
        return loss_fn
    
    def _prepare_training_data(
        self,
        train_data: Any,
    ) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
        """Prepare data loaders for training, validation, and testing.
        
        Args:
            train_data: Either a TiDEWindowedDataset or a BrisT1DDataLoader instance
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # If train_data is already a TiDEWindowedDataset, use it directly
        if isinstance(train_data, TiDEWindowedDataset):
            train_dataset = train_data
            
            # Create validation and test datasets if the data loader has them
            val_dataset = None
            test_dataset = None
            
            if hasattr(train_data, 'data_loader_instance'):
                data_loader = train_data.data_loader_instance
                
                if data_loader.validation_data:
                    val_dataset = TiDEWindowedDataset(
                        data_loader,
                        lookback_steps=self.config.seq_len,
                        horizon_steps=self.config.forecast_length,
                        dataset_type='validation',
                        stride=self.config.seq_len // 2  # 50% overlap for validation
                    )
                
                if data_loader.test_data:
                    test_dataset = TiDEWindowedDataset(
                        data_loader,
                        lookback_steps=self.config.seq_len,
                        horizon_steps=self.config.forecast_length,
                        dataset_type='test',
                        stride=self.config.seq_len // 2
                    )
        else:
            # Assume it's a BrisT1DDataLoader instance
            train_dataset = TiDEWindowedDataset(
                train_data,
                lookback_steps=self.config.seq_len,
                horizon_steps=self.config.forecast_length,
                dataset_type='train',
                stride=self.config.seq_len // 2
            )
            
            val_dataset = TiDEWindowedDataset(
                train_data,
                lookback_steps=self.config.seq_len,
                horizon_steps=self.config.forecast_length,
                dataset_type='validation',
                stride=self.config.seq_len // 2
            ) if train_data.validation_data else None
            
            test_dataset = TiDEWindowedDataset(
                train_data,
                lookback_steps=self.config.seq_len,
                horizon_steps=self.config.forecast_length,
                dataset_type='test',
                stride=self.config.seq_len // 2
            ) if train_data.test_data else None
        
        # Initialize model now that we know the feature dimensions
        if self.model is None:
            # Get a sample to determine feature dimensions
            sample_x, sample_y, _, _ = train_dataset[0]
            num_features = sample_x.shape[1]
            
            self.sizes['lookback'] = (self.config.seq_len, num_features)
            self.sizes['dynCov'] = (self.config.seq_len + self.config.forecast_length, num_features)
            
            self.model = self._create_model_from_sizes(self.sizes)
            info_print(f"Initialized TiDE model with {num_features} features")
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.dataloader_num_workers,
            pin_memory=torch.cuda.is_available() and not self.config.use_cpu
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.dataloader_num_workers,
            pin_memory=torch.cuda.is_available() and not self.config.use_cpu
        ) if val_dataset else None
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.dataloader_num_workers,
            pin_memory=torch.cuda.is_available() and not self.config.use_cpu
        ) if test_dataset else None
        
        return train_loader, val_loader, test_loader
    
    def _train_model(
        self,
        train_data: Any,
        output_dir: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """Train the TiDE model using native PyTorch training loop."""
        import os
        from tqdm import tqdm
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare data
        train_loader, val_loader, test_loader = self._prepare_training_data(train_data)
        
        # Setup optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Warmup + linear decay scheduler
        total_steps = len(train_loader) * self.config.num_epochs
        
        def lr_lambda(current_step):
            if current_step < self.config.warmup_steps:
                return float(current_step) / float(max(1, self.config.warmup_steps))
            return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - self.config.warmup_steps)))
        
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
        # Training state
        device = self._get_device()
        best_val_loss = float('inf')
        patience_counter = 0
        training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
        info_print(f"Starting TiDE training for {self.config.num_epochs} epochs")
        info_print(f"Device: {device}")
        info_print(f"Training batches: {len(train_loader)}")
        if val_loader:
            info_print(f"Validation batches: {len(val_loader)}")
        
        # Training loop
        for epoch in range(self.config.num_epochs):
            # Training phase
            self.model.train()
            train_losses = []
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")
            for batch_idx, (seq_x, seq_y, seq_x_mark, seq_y_mark) in enumerate(pbar):
                # Move to device
                seq_x = seq_x.to(device)
                seq_y = seq_y.to(device)
                seq_x_mark = seq_x_mark.to(device)
                seq_y_mark = seq_y_mark.to(device)
                
                # Forward pass
                self.optimizer.zero_grad()
                
                # TiDE expects: batch_x (static), batch_y (lookback), batch_x_mark, batch_y_mark
                # For now, use zeros for static attributes
                batch_x = torch.zeros(seq_x.shape[0], *self.sizes['attr']).to(device)
                
                predictions, targets = self.model(batch_x, seq_x, seq_x_mark, seq_y_mark)
                
                # Compute loss
                loss = self.criterion(predictions, targets)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.config.gradient_clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip_val
                    )
                
                self.optimizer.step()
                self.scheduler.step()
                
                train_losses.append(loss.item())
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            avg_train_loss = np.mean(train_losses)
            training_history['train_loss'].append(avg_train_loss)
            training_history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            # Validation phase
            if val_loader:
                val_loss = self._validate(val_loader, device)
                training_history['val_loss'].append(val_loss)
                
                info_print(
                    f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, "
                    f"val_loss={val_loss:.4f}, lr={self.optimizer.param_groups[0]['lr']:.6f}"
                )
                
                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    self.save(os.path.join(output_dir, "best_model"))
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.early_stopping_patience:
                        info_print(f"Early stopping triggered after {epoch+1} epochs")
                        break
            else:
                info_print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}")
        
        # Test evaluation if available
        test_metrics = {}
        if test_loader:
            test_metrics = self.evaluate(test_loader, return_predictions=False)
            info_print(f"Test metrics: {test_metrics}")
        
        return {
            'training_history': training_history,
            'train_metrics': {'final_train_loss': avg_train_loss},
            'test_metrics': test_metrics
        }

    def _validate(self, val_loader: DataLoader, device: torch.device) -> float:
        """Run validation and return average loss."""
        self.model.eval()
        val_losses = []
        
        with torch.no_grad():
            for seq_x, seq_y, seq_x_mark, seq_y_mark in val_loader:
                seq_x = seq_x.to(device)
                seq_y = seq_y.to(device)
                seq_x_mark = seq_x_mark.to(device)
                seq_y_mark = seq_y_mark.to(device)
                
                batch_x = torch.zeros(seq_x.shape[0], *self.sizes['attr']).to(device)
                predictions, targets = self.model(batch_x, seq_x, seq_x_mark, seq_y_mark)
                
                loss = self.criterion(predictions, targets)
                val_losses.append(loss.item())
        
        return np.mean(val_losses)

    def predict(self, data: Any, batch_size: Optional[int] = None) -> np.ndarray:
        """Make predictions on new data.
        
        Args:
            data: Either a DataLoader or TiDEWindowedDataset
            batch_size: Batch size for prediction
            
        Returns:
            Predictions as numpy array of shape (n_samples, pred_len, 1)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        self.model.eval()
        device = self._get_device()
        batch_size = batch_size or self.config.batch_size
        
        # Create data loader if needed
        if not isinstance(data, DataLoader):
            data_loader = DataLoader(
                data,
                batch_size=batch_size,
                shuffle=False,
                num_workers=self.config.dataloader_num_workers
            )
        else:
            data_loader = data
        
        predictions = []
        
        with torch.no_grad():
            for seq_x, seq_y, seq_x_mark, seq_y_mark in data_loader:
                seq_x = seq_x.to(device)
                seq_x_mark = seq_x_mark.to(device)
                seq_y_mark = seq_y_mark.to(device)
                
                batch_x = torch.zeros(seq_x.shape[0], *self.sizes['attr']).to(device)
                preds, _ = self.model(batch_x, seq_x, seq_x_mark, seq_y_mark)
                
                predictions.append(preds.cpu().numpy())
        
        return np.concatenate(predictions, axis=0)

    def _extract_ground_truth(self, test_data: Any) -> np.ndarray:
        """Extract ground truth from test data."""
        if isinstance(test_data, DataLoader):
            targets = []
            for _, seq_y, _, _ in test_data:
                # Extract the target portion
                targets.append(seq_y[:, -self.config.forecast_length:, :].numpy())
            return np.concatenate(targets, axis=0)
        else:
            # Assume it's a dataset
            targets = []
            for i in range(len(test_data)):
                _, seq_y, _, _ = test_data[i]
                targets.append(seq_y[-self.config.forecast_length:, :].numpy())
            return np.array(targets)

    def _save_checkpoint(self, output_dir: str) -> None:
        """Save TiDE model checkpoint."""
        checkpoint_path = Path(output_dir) / "model.pt"
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'sizes': self.sizes,
        }
        
        if self.optimizer is not None:
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        info_print(f"Model checkpoint saved to {checkpoint_path}")

    def _load_checkpoint(self, model_dir: str) -> None:
        """Load TiDE model checkpoint."""
        checkpoint_path = Path(model_dir) / "model.pt"
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self._get_device())
        
        # Restore sizes and initialize model
        self.sizes = checkpoint['sizes']
        self.model = self._create_model_from_sizes(self.sizes)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Optionally restore optimizer and scheduler
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.is_fitted = True
        info_print(f"Model checkpoint loaded from {checkpoint_path}")


    # Helper function for convenience
    def create_tide_model(config: Optional[TiDEConfig] = None) -> TiDEForecaster:
        """Create a TiDE forecaster with optional custom configuration."""
        if config is None:
            config = TiDEConfig()
        return TiDEForecaster(config)
