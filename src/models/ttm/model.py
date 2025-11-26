"""
TTM (TinyTimeMixer) model implementation using the base TSFM framework.

This module provides a concrete implementation of TTM that inherits from
the base TSFM framework, demonstrating how to integrate existing models.
"""

import os
import json
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import (
    TrainingArguments,
    EarlyStoppingCallback,
)

# Import your existing TTM-related modules
from tsfm_public import (
    TimeSeriesPreprocessor,
    TrackingCallback,
    get_datasets,
)
from tsfm_public.toolkit.get_model import get_model
from tsfm_public.toolkit.time_series_preprocessor import ScalerType

# Local imports
from src.models.base import BaseTSFM, ModelConfig
from src.data.diabetes_datasets.data_loader import get_loader
from src.data.models import ColumnNames
from src.data.preprocessing.split_or_combine_patients import reduce_features_multi_patient
from src.utils.logging_helper import info_print, debug_print, error_print


class TTMConfig(ModelConfig):
    """Extended configuration class for TTM-specific parameters."""
    
    def __init__(self, **kwargs):
        # Extract TTM-specific parameters before calling parent
        ttm_specific_params = {
            "scaler_type", 
            "imputation_strategy",
            "num_input_channels",
            "num_output_channels",
            "prediction_filter_length",
            "resolution_min",
        }
        
        # Filter out TTM-specific params from kwargs for parent class
        base_kwargs = {k: v for k, v in kwargs.items() if k not in ttm_specific_params}
        
        # Call parent with filtered kwargs
        super().__init__(**base_kwargs)
        
        # Set TTM-specific parameters
        self.model_type = "ttm"
        # model_path is now handled by the parent class
        
        # Data preprocessing
        self.scaler_type = kwargs.get("scaler_type", "standard")
        self.imputation_strategy = kwargs.get("imputation_strategy", "mean")
        self.input_features = kwargs.get("input_features", [
            "cob", "carb_availability", 
            "insulin_availability", "iob", "steps"])
        self.target_features = kwargs.get("target_features", ["bg_mM"])
        self.split_config = kwargs.get("split_config", {
            "train": 0.9,
            "val": 0.05,
            "test": 0.05
        })
        self.fewshot_percent = kwargs.get("fewshot_percent", 5)
        
        # TTM architecture specifics
        self.num_input_channels = kwargs.get("num_input_channels", 1)
        self.num_output_channels = kwargs.get("num_output_channels", 1)
        self.resolution_min = kwargs.get("resolution_min", 100)
        
        # Training specifics for TTM
        self.freeze_backbone = kwargs.get("freeze_backbone", True)
        self.prediction_filter_length = kwargs.get("prediction_filter_length", None)


class TTMForecaster(BaseTSFM):
    """
    TTM (TinyTimeMixer) forecaster implementation using the base TSFM framework.
    
    This class demonstrates how to integrate an existing model (TTM) into the
    unified base framework while preserving all existing functionality.
    """
    
    def __init__(
        self,
        config: TTMConfig,
        lora_config=None,
        distributed_config=None
    ):
        """Initialize TTM forecaster."""
        # Ensure we're using TTMConfig
        if not isinstance(config, TTMConfig):
            config = TTMConfig(**config.to_dict())
        
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
            # The freeze_backbone and other parameters might need to be applied after loading
            model_params = {
                "model_path": self.config.model_path,
                "context_length": self.config.context_length,
                "prediction_length": self.config.forecast_length,
                "freq":f"{self.config.resolution_min}min",
            }
            
            # Only add prediction_filter_length if it's not None
            if self.config.prediction_filter_length is not None:
                model_params["prediction_filter_length"] = self.config.prediction_filter_length
            
            # Get TTM model using the existing tsfm_public toolkit
            ttm_model = get_model(**model_params)
            
            # Apply freeze_backbone after model loading if needed
            if self.config.freeze_backbone:
                debug_print("Freezing backbone parameters...")
                for param in ttm_model.parameters():
                    param.requires_grad = False
                # Unfreeze the prediction head (usually the last layer)
                if hasattr(ttm_model, 'prediction_head'):
                    for param in ttm_model.prediction_head.parameters():
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
        test_data: Optional[Any] = None
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
            info_print(f"Converted multi-patient dict to DataFrame \n dataset now how has the following columns available: {data.columns}")
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
            
            info_print(f"Data preparation complete:")
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
                ColumnNames.STEPS.value, ColumnNames.COB.value, ColumnNames.CARB_AVAILABILITY.value, 
                ColumnNames.INSULIN_AVAILABILITY.value, ColumnNames.IOB.value
            ],
            "conditional_columns": [],
            "static_categorical_columns": [],
        }
        
        # Filter to only include columns that exist in the data
        available_columns = set(data.columns)
        
        for key, columns in column_specifiers.items():
            if isinstance(columns, list):
                column_specifiers[key] = [col for col in columns if col in available_columns]
        
        return column_specifiers
    
    def _create_training_arguments(self, output_dir: str) -> TrainingArguments:
        """Create TTM-specific training arguments."""
        return TrainingArguments(
            output_dir=output_dir,
            learning_rate=self.config.learning_rate,
            num_train_epochs=self.config.num_epochs,
            
            # Evaluation
            do_eval=True,
            eval_strategy=self.config.eval_strategy,
            eval_steps=self.config.eval_steps,
            
            # Batch size
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            
            # Performance
            fp16=self.config.fp16,
            dataloader_num_workers=self.config.dataloader_num_workers,
            
            # Logging and saving
            report_to="none",
            save_strategy="steps",
            logging_strategy="steps",
            logging_steps=self.config.logging_steps,
            logging_first_step=True,
            save_steps=self.config.save_steps,
            save_total_limit=100,
            logging_dir=os.path.join(output_dir, "logs"),
            
            # Model selection
            load_best_model_at_end=True,
            metric_for_best_model=self.config.metric_for_best_model,
            greater_is_better=self.config.greater_is_better,
            
            # Hardware
            use_cpu=self.config.use_cpu,
            
            # Distributed training
            **self._get_distributed_training_args()
        )
    
    def _get_distributed_training_args(self) -> Dict[str, Any]:
        """Get distributed training arguments for TrainingArguments."""
        if not self.distributed_config.enabled:
            return {}
        
        args = {}
        
        if self.distributed_config.strategy == "deepspeed":
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
            if hasattr(eval_pred, 'predictions'):
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
            mape = np.mean(np.abs((predictions[mask] - labels[mask]) / labels[mask]) * 100) if np.any(mask) else 0.0
            
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
    **kwargs
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
        **kwargs
    )
    
    return TTMForecaster(config)
