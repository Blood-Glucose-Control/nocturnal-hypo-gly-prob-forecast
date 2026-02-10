"""
TTM (TinyTimeMixer) model implementation using the base TSFM framework.

This module provides a concrete implementation of TTM that inherits from
the base TSFM framework, demonstrating how to integrate existing models.
"""

import os
import logging
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from transformers import (
    TrainingArguments,
    Trainer,
)

# Import your existing TTM-related modules
from tsfm_public import (
    TimeSeriesPreprocessor,
    get_datasets,
)
from tsfm_public.toolkit.get_model import get_model
from tsfm_public.toolkit.time_series_preprocessor import ScalerType

# Local imports
from src.models.base import BaseTimeSeriesFoundationModel, TrainingBackend
from src.models.ttm.config import TTMConfig
from src.data.models import ColumnNames
from src.data.preprocessing.split_or_combine_patients import (
    reduce_features_multi_patient,
)
from src.utils.logging_helper import info_print, debug_print, error_print

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ColumnSpecifiers(TypedDict, total=False):
    """Type definition for TimeSeriesPreprocessor column configuration.

    Attributes:
        id_columns: Columns identifying unique time series (e.g., patient_id).
        timestamp_column: Single column name for timestamps.
        target_columns: Columns to forecast.
        observable_columns: Known in past, unknown in future.
        control_columns: Known in both past and future.
        conditional_columns: Conditional features.
        static_categorical_columns: Static categorical features.
    """

    id_columns: List[str]
    timestamp_column: str
    target_columns: List[str]
    observable_columns: List[str]
    control_columns: List[str]
    conditional_columns: List[str]
    static_categorical_columns: List[str]


class TTMForecaster(BaseTimeSeriesFoundationModel):
    """TTM (TinyTimeMixer) forecaster implementation.

    TinyTimeMixer is an MLP-based time series foundation model that uses mixing
    layers instead of attention mechanisms.

    Attributes:
        config: TTM-specific configuration
        preprocessor: TimeSeriesPreprocessor for data normalization and windowing
        column_specifiers: Dictionary mapping data columns to their roles

    Note:
        TTM does not support LoRA fine-tuning (no transformer attention layers)
    """

    def __init__(self, config: TTMConfig, lora_config=None, distributed_config=None):
        """Initialize the TTM forecaster.

        Args:
            config: TTM configuration object
            lora_config: LoRA configuration (ignored for TTM)
            distributed_config: Configuration for distributed training
        """
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
        info_print("TTMForecaster initialized with configuration:")
        for key, value in self.config.__dict__.items():
            info_print(f"  {key}: {value}")
        # TTM-specific attributes (lazily initialized in _prepare_data)
        self.preprocessor: Optional[TimeSeriesPreprocessor] = None
        self.column_specifiers: Optional[ColumnSpecifiers] = None

    # Properties
    @property
    def training_backend(self) -> TrainingBackend:
        """Return the training backend used by TTM.

        Returns:
            TrainingBackend.TRANSFORMERS
        """
        return TrainingBackend.TRANSFORMERS

    @property
    def supports_lora(self) -> bool:
        """Check if TTM supports LoRA fine-tuning.

        Returns:
            False (TTM is MLP-based, lacks attention layers)
        """
        return False

    # Abstract method implementations
    ## Abstract implemented public methods
    def predict(
        self,
        data: Any,
        batch_size: Optional[int] = None,
        inverse_scale: bool = True,
        return_dict: bool = False,
    ) -> Union[np.ndarray, Dict[str, Any]]:
        """Make predictions on new data. This will be very specific to each child class.
            When designing this, just consider what the final data shape should look like.

        Args:
            data: Input data for prediction
            batch_size: Batch size for prediction
            inverse_scale: If True, inverse transform predictions to original scale.
                          Requires preprocessor to have been fitted during training.
            return_dict: If True, return a dictionary with predictions and metadata.

        Returns:
            Predictions as numpy array (in original scale if inverse_scale=True)
            or a dictionary with predictions and metadata if return_dict=True.

        Raises:
            ValueError: If model has not been fitted
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before making predictions")

        # Prepare data for inference using the fitted preprocessor
        # Falls back to training data prep if preprocessor not available (e.g., loaded model)
        try:
            data_loader = self._prepare_inference_data(data)
        except ValueError as e:
            info_print(f"Falling back to _prepare_training_data: {e}")
            data_loader, _, _ = self._prepare_training_data(data)

        # Create trainer for inference
        trainer = self._create_inference_trainer(batch_size)

        # Generate predictions using Trainer
        info_print("Generating predictions using Trainer.predict()...")
        predictions_output = trainer.predict(data_loader.dataset)

        # Extract predictions from PredictionOutput
        # predictions_output.predictions is a tuple: (forecasts, embeddings)
        predictions = predictions_output.predictions[0]  # Get forecasts

        # Convert to numpy if needed
        if hasattr(predictions, "cpu"):
            predictions = predictions.cpu().numpy()  # type: ignore
        elif not isinstance(predictions, np.ndarray):
            predictions = np.array(predictions)

        info_print(f"Predictions shape (scaled): {predictions.shape}")

        # Inverse scale predictions back to original units
        if inverse_scale and self.preprocessor is not None:
            predictions = self._inverse_scale_predictions(predictions, data)
            info_print("Predictions inverse-scaled to original units")

        info_print(f"Predictions shape: {predictions.shape}")

        if return_dict:
            result = {
                "predictions": predictions,
                "model_config": self.config.to_dict(),
                "n_samples": len(predictions),
            }
            # Include backbone embeddings if available
            if len(predictions_output.predictions) > 1:
                result["backbone_embeddings"] = predictions_output.predictions[1]
                info_print(
                    f"Backbone embeddings shape: {predictions_output.predictions[1].shape}"
                )
            return result

        return predictions

    def _inverse_scale_predictions(
        self, predictions: np.ndarray, data: Any
    ) -> np.ndarray:
        """Inverse scale predictions back to original units.

        When the preprocessor uses global scaling (scaling_id_columns=[]),
        we can directly use the global scaler to inverse transform predictions.

        Args:
            predictions: Scaled predictions array of shape (samples, forecast_length, channels)
                        or (forecast_length, channels) or (forecast_length,)
            data: Original data (used for context, not currently needed for global scaling)

        Returns:
            Predictions inverse-scaled to original units
        """
        if self.preprocessor is None:
            info_print("No preprocessor available, returning predictions as-is")
            return predictions

        if not self.preprocessor.scaling:
            info_print("Scaling disabled, returning predictions as-is")
            return predictions

        if len(self.preprocessor.target_scaler_dict) == 0:
            info_print("No scalers trained, returning predictions as-is")
            return predictions

        # Get the target scaler
        # When using global scaling (scaling_id_columns=[]), the key is '__id'
        from tsfm_public.toolkit.time_series_preprocessor import INTERNAL_ID_COLUMN

        scaler_key = INTERNAL_ID_COLUMN  # '__id' for global scaling
        if scaler_key not in self.preprocessor.target_scaler_dict:
            # Fall back to first available scaler if global key not found
            scaler_key = next(iter(self.preprocessor.target_scaler_dict.keys()))
            info_print(f"Using scaler key: {scaler_key}")

        scaler = self.preprocessor.target_scaler_dict[scaler_key]

        # Handle different prediction shapes
        original_shape = predictions.shape
        info_print(f"Inverse scaling predictions with shape: {original_shape}")

        # Reshape to 2D for sklearn scaler (samples, features)
        if len(original_shape) == 1:
            # (forecast_length,) -> (forecast_length, 1)
            predictions_2d = predictions.reshape(-1, 1)
        elif len(original_shape) == 2:
            # (forecast_length, channels) or (samples, forecast_length)
            # Assume (samples, forecast_length) and reshape to (samples * forecast_length, 1)
            predictions_2d = predictions.reshape(-1, 1)
        elif len(original_shape) == 3:
            # (samples, forecast_length, channels)
            # For target channel (channel 0), reshape to (samples * forecast_length, 1)
            n_samples, forecast_len, n_channels = original_shape
            # Only inverse scale the target channel(s) - typically just channel 0
            predictions_2d = predictions[:, :, 0].reshape(-1, 1)
        else:
            info_print(f"Unexpected prediction shape {original_shape}, returning as-is")
            return predictions

        # Inverse transform using the scaler
        try:
            predictions_unscaled = scaler.inverse_transform(predictions_2d)

            # Reshape back to original shape
            if len(original_shape) == 1:
                result = predictions_unscaled.flatten()
            elif len(original_shape) == 2:
                result = predictions_unscaled.reshape(
                    original_shape[0], original_shape[1]
                )
            elif len(original_shape) == 3:
                # Put unscaled values back into channel 0, keep other channels as-is
                result = predictions.copy()
                result[:, :, 0] = predictions_unscaled.reshape(n_samples, forecast_len)

            return result

        except Exception as e:
            error_print(f"Failed to inverse scale predictions: {e}")
            return predictions

    ## Abstract implemented private methods
    def _initialize_model(self) -> None:
        """Initialize the TTM model architecture.

        Loads pre-trained model and configures parameter gradients based
        on training_mode.

        Raises:
            Exception: If model initialization fails
        """
        try:
            info_print(f"Initializing TTM model from {self.config.model_path}")

            # Prepare minimal parameters for TTM model initialization
            model_params = {
                "model_path": self.config.model_path,
                "context_length": self.config.context_length,
                "prediction_length": self.config.forecast_length,
                "freq": f"{self.config.resolution_min}min",
                "return_model_key": False,  # Ensure we get the model object, not a string
            }

            # Only add prediction_filter_length if it's not None
            if self.config.prediction_filter_length is not None:
                model_params["prediction_filter_length"] = (
                    self.config.prediction_filter_length
                )

            info_print(
                f"Attempting to load TTM model with the following parameters: \n {model_params}"
            )
            # Get TTM model using the existing tsfm_public toolkit
            ttm_model = get_model(**model_params)

            # Validate that we received a model object, not a string
            if isinstance(ttm_model, str):
                raise TypeError(
                    f"Expected model object from get_model(), but received string: {ttm_model}"
                )

            # Configure parameter gradients based on training strategy
            if self.config.training_mode == "zero_shot":
                info_print("Freezing all parameters for zero-shot evaluation")
                for param in ttm_model.parameters():
                    param.requires_grad = False
            else:
                # For any training scenario (fine_tune, from_scratch), enable gradients
                info_print(
                    f"Enabling gradients for all parameters ({self.config.training_mode} mode)"
                )
                for param in ttm_model.parameters():
                    param.requires_grad = True

            self.model = ttm_model
            info_print("TTM model initialized successfully")

        except Exception as e:
            error_print(f"Failed to initialize TTM model: {str(e)}")
            raise

    def _prepare_inference_data(self, data: Any) -> DataLoader:
        """Prepare data for inference (prediction) using the existing preprocessor.

        Uses the fitted preprocessor from training to scale the data and creates
        a ForecastDFDataset for inference. Does NOT retrain scalers.

        Args:
            data: Input data for prediction (DataFrame or dict of DataFrames)

        Returns:
            DataLoader for inference

        Raises:
            ValueError: If preprocessor hasn't been trained
        """
        from tsfm_public.toolkit.dataset import ForecastDFDataset

        if self.preprocessor is None:
            raise ValueError(
                "Preprocessor not available. Model must be trained before prediction, "
                "or preprocessor must be loaded with the model."
            )

        # Convert dict format to DataFrame if needed
        if isinstance(data, dict):
            from src.data.data_loading import multi_patient_dict_to_df

            data = multi_patient_dict_to_df(
                patients_dict=data,
                resolution_min=self.config.resolution_min,
                x_features=self.config.input_features,
                y_feature=self.config.target_features,
            )
            data = data.reset_index()

        info_print(f"Preprocessing inference data with shape: {data.shape}")

        # Use the existing preprocessor to scale the data (without retraining)
        # preprocess() applies the fitted scalers
        scaled_data = self.preprocessor.preprocess(data)

        # Create ForecastDFDataset for inference using the preprocessor's column specs
        inference_dataset = ForecastDFDataset(
            data=scaled_data,
            id_columns=self.preprocessor.id_columns,
            timestamp_column=self.preprocessor.timestamp_column,
            target_columns=self.preprocessor.target_columns,
            observable_columns=self.preprocessor.observable_columns,
            control_columns=self.preprocessor.control_columns,
            conditional_columns=self.preprocessor.conditional_columns,
            static_categorical_columns=self.preprocessor.static_categorical_columns,
            context_length=self.config.context_length,
            prediction_length=self.config.forecast_length,
        )

        # Create DataLoader for inference
        inference_loader = DataLoader(
            inference_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.dataloader_num_workers,
        )

        info_print(
            f"Created inference DataLoader with {len(inference_dataset)} samples"
        )
        return inference_loader

    def _prepare_training_data(
        self,
        train_data: Any,
    ) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
        """Prepare data loaders for training, validation, and testing.

        Data splitting is controlled by self.config.split_config.

        Args:
            train_data: Training data (DataFrame or dict of patient DataFrames)

        Returns:
            Tuple of train, validation, and test DataLoaders (split based on config)

        Raises:
            ValueError: If train_data is not a DataFrame or dict
            Exception: If data preprocessing fails
        """
        info_print("Preparing data for TTM training...")

        # Validate input type
        if not isinstance(train_data, (pd.DataFrame, dict)):
            raise ValueError(
                f"train_data must be a DataFrame or dict, got {type(train_data)}"
            )

        data = train_data

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

        info_print("Using column specifiers:")
        for key, value in self.column_specifiers.items():
            info_print(f"  {key}: {value}")
        # Create preprocessor
        if self.preprocessor is None:
            self.preprocessor = TimeSeriesPreprocessor(
                **self.column_specifiers,
                context_length=self.config.context_length,
                prediction_length=self.config.forecast_length,
                scaling=True,
                scaling_id_columns=[],  # Use global scaler for all patients (supports holdout/new patients)
                encode_categorical=False,
                scaler_type=ScalerType.STANDARD.value,  # type: ignore[arg-type]
            )

        # Create datasets using tsfm_public get_datasets
        # Note: get_datasets returns (train, val, test) datasets but lacks type stubs
        logger.info("\n")
        info_print("Splitting data into train/val/test sets...")
        info_print(f"  Split config: {self.config.split_config}")
        try:
            dset_train, dset_val, dset_test = get_datasets(  # type: ignore[misc]
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
            info_print(f"  Train samples: {len(dset_train) if dset_train else 0:,}")
            info_print(f"  Val samples: {len(dset_val) if dset_val else 0:,}")
            info_print(f"  Test samples: {len(dset_test) if dset_test else 0:,}")

            return train_loader, val_loader, test_loader

        except Exception as e:
            error_print(f"Failed to prepare data: {str(e)}")
            raise

    def _save_checkpoint(self, output_dir: str) -> None:
        """Save model checkpoint and preprocessor.

        Args:
            output_dir: Directory path for saving checkpoint
        """
        import pickle

        if self.model is not None and hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(output_dir)  # type: ignore[union-attr]
            info_print(f"TTM model saved to {output_dir}")

        # Save the preprocessor for inference (critical for new/holdout patients)
        # Using pickle because tsfm_public's save_pretrained uses json.dumps(sort_keys=True)
        # which fails when preprocessor has mixed key types (float patient IDs + string keys)
        if self.preprocessor is not None:
            preprocessor_path = os.path.join(output_dir, "preprocessor.pkl")
            with open(preprocessor_path, "wb") as f:
                pickle.dump(self.preprocessor, f)
            info_print(f"Preprocessor saved to {preprocessor_path}")

    def _load_checkpoint(self, model_dir: str) -> None:
        """Load model checkpoint.

        Args:
            model_dir: Directory containing saved checkpoint

        Raises:
            Exception: If loading fails
        """
        import pickle

        try:
            # Use get_model() to load the TTM architecture from the checkpoint directory
            # This properly handles the custom TTM model type
            model_params = {
                "model_path": model_dir,  # Load from checkpoint directory
                "context_length": self.config.context_length,
                "prediction_length": self.config.forecast_length,
                "freq": f"{self.config.resolution_min}min",
                "return_model_key": False,  # Ensure we get the model object, not a string
            }

            # Only add prediction_filter_length if it's not None
            if self.config.prediction_filter_length is not None:
                model_params["prediction_filter_length"] = (
                    self.config.prediction_filter_length
                )

            info_print(
                f"Loading TTM checkpoint from {model_dir} with params: {model_params}"
            )
            ttm_model = get_model(**model_params)

            # Validate that we received a model object, not a string
            if isinstance(ttm_model, str):
                raise TypeError(
                    f"Expected model object from get_model(), but received string: {ttm_model}"
                )

            self.model = ttm_model
            info_print(f"TTM model checkpoint loaded from {model_dir}")

            # Load the preprocessor if saved (critical for inference on holdout patients)
            # Try pickle format first (new), then fall back to pretrained format (legacy)
            preprocessor_pkl_path = os.path.join(model_dir, "preprocessor.pkl")
            preprocessor_dir = os.path.join(model_dir, "preprocessor")

            if os.path.exists(preprocessor_pkl_path):
                with open(preprocessor_pkl_path, "rb") as f:
                    self.preprocessor = pickle.load(f)
                info_print(f"Preprocessor loaded from {preprocessor_pkl_path}")
            elif os.path.exists(preprocessor_dir):
                # Legacy format - try from_pretrained
                self.preprocessor = TimeSeriesPreprocessor.from_pretrained(
                    preprocessor_dir
                )
                info_print(
                    f"Preprocessor loaded from {preprocessor_dir} (legacy format)"
                )
            else:
                info_print(
                    f"No preprocessor found at {model_dir}, inference may require refitting"
                )

        except Exception as e:
            error_print(f"Failed to load model checkpoint: {str(e)}")
            raise

    def _train_model(
        self,
        train_data: Any,
        output_dir: str = "./output",
        **kwargs,
    ) -> Dict[str, Any]:
        """Execute model training.

        Data splitting for train/val/test is controlled by self.config.split_config.

        Args:
            train_data: Training data (will be split based on config)
            output_dir: Directory for saving checkpoints and logs
            **kwargs: Additional arguments (e.g., resume_from_checkpoint)

        Returns:
            Dictionary containing train_metrics and test_metrics
        """
        # Configure tqdm to update less frequently (every 30 seconds instead of constantly)
        # This reduces log file bloat while still showing progress
        import os

        os.environ["TQDM_MININTERVAL"] = "30"  # Update progress bar every 30 seconds

        # Prevent GPU memory fragmentation. PyTorch's default CUDA allocator uses
        # fixed-size blocks that can't be merged when freed, causing "reserved but
        # unallocated" memory to grow over time (especially when switching between
        # training and evaluation, which produce different tensor sizes).
        # expandable_segments:True allows memory segments to grow/shrink dynamically,
        # making freed memory actually reclaimable. This is safe and stable since
        # PyTorch 2.1 with no performance downside.
        #
        # NOTE: This is a fallback for users running the training module directly.
        # For reliable configuration, set PYTORCH_ALLOC_CONF in your shell/runner
        # script BEFORE invoking Python (e.g., in run_holdout_generic_workflow.sh).
        # Setting it here may not take effect if CUDA was already initialized.
        if "PYTORCH_ALLOC_CONF" not in os.environ:
            os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
            # Check if CUDA allocator was already initialized (setting won't take effect)
            import torch

            if torch.cuda.is_initialized():
                logger.warning(
                    "PYTORCH_ALLOC_CONF was set after CUDA initialization. "
                    "The 'expandable_segments:True' setting may not take effect. "
                    "For reliable memory fragmentation prevention, set "
                    "PYTORCH_ALLOC_CONF in your shell before running Python."
                )

        info_print("Starting TTM training using HuggingFace Trainer...")
        # Prepare data loaders (splits based on config)
        train_loader, val_loader, test_loader = self._prepare_training_data(train_data)

        # Create training arguments
        training_args = self._create_training_arguments(output_dir)

        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_loader.dataset if train_loader else None,
            eval_dataset=val_loader.dataset if val_loader else None,
            compute_metrics=self._compute_trainer_metrics,
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
        trainer.save_model(output_dir=output_dir)

        # Get training history directly from trainer.state (in memory)
        # This is more reliable than reading from file since trainer_state.json
        # is only saved in checkpoint directories, not at output_dir root
        training_history = {}
        if hasattr(trainer, "state") and trainer.state is not None:
            training_history = {
                "log_history": trainer.state.log_history,
                "best_metric": trainer.state.best_metric,
                "best_model_checkpoint": trainer.state.best_model_checkpoint,
                "global_step": trainer.state.global_step,
                "epoch": trainer.state.epoch,
            }
            info_print("Captured training history from trainer state")
            info_print(f"  Total log entries: {len(trainer.state.log_history)}")
        else:
            info_print("Warning: Could not access trainer.state")

        # Evaluate on test set if provided
        # Note: This evaluation can be memory-intensive for large test sets
        # because HF Trainer accumulates all prediction tensors on GPU.
        # Free training state first to maximize available memory.
        test_metrics = {}
        if test_loader is not None:
            import torch

            # Clear training-related GPU cache before evaluation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            try:
                test_metrics = trainer.evaluate(eval_dataset=test_loader.dataset)
                info_print(f"Test metrics: {test_metrics}")
            except torch.cuda.OutOfMemoryError:
                info_print(
                    "Warning: Test evaluation skipped due to GPU OOM. "
                    "This is non-fatal — full holdout evaluation runs separately in step 8."
                )
                # Clear the failed allocation
                torch.cuda.empty_cache()

        return {
            "train_metrics": train_result.metrics,
            "test_metrics": test_metrics,
            "training_history": training_history,
        }

    # TTM-specific public methods
    # NOTE: evaluate() is inherited from BaseTimeSeriesFoundationModel
    # It calls predict() and computes metrics using _compute_metrics()

    def predict_zero_shot(
        self,
        data: Any,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """Make zero-shot predictions without fine-tuning.

        Uses the tsfm_public pattern: create a TimeSeriesPreprocessor,
        use get_datasets() to prepare the data, then use Trainer.predict()
        directly on the dataset.

        Args:
            data: Input data for prediction (DataFrame)
            batch_size: Batch size for prediction

        Returns:
            Model predictions as numpy array
        """
        import tempfile

        info_print("Making zero-shot predictions with TTM")

        # Convert dict format to DataFrame if needed
        if isinstance(data, dict):
            data = pd.concat(data.values(), ignore_index=True)

        # Create column specifiers if needed
        if self.column_specifiers is None:
            self.column_specifiers = self._create_column_specifiers(data)

        # Create preprocessor for zero-shot inference
        tsp = TimeSeriesPreprocessor(
            **self.column_specifiers,
            context_length=self.config.context_length,
            prediction_length=self.config.forecast_length,
            scaling=True,
            encode_categorical=False,
            scaler_type=ScalerType.STANDARD.value,  # type: ignore[arg-type]
        )

        # Use get_datasets to prepare data (handles preprocessor fitting internally)
        # Use a split that puts all data in test for inference
        split_config = {"train": 0.33, "test": 0.33, "val": 0.34}

        # Check for resolution_prefix_tuning on model config
        use_freq_token = False
        if self.model is not None and hasattr(self.model, "config"):
            model_config = getattr(self.model, "config", None)
            if model_config is not None and hasattr(
                model_config, "resolution_prefix_tuning"
            ):
                use_freq_token = bool(
                    getattr(model_config, "resolution_prefix_tuning", False)
                )
        info_print(f"Passing the following dataset to get_dataset: \n {data}")
        dset_train, dset_val, dset_test = get_datasets(  # type: ignore[misc]
            tsp,
            data,
            split_config,  # type: ignore[arg-type]
            use_frequency_token=use_freq_token,
        )
        info_print("Data prepared for zero-shot inference")
        # Create temporary directory for trainer output
        temp_dir = tempfile.mkdtemp()

        # Create trainer for zero-shot inference
        zeroshot_trainer = Trainer(
            model=self.model,
            args=TrainingArguments(
                output_dir=temp_dir,
                per_device_eval_batch_size=batch_size or self.config.batch_size,
                seed=42,
                report_to="none",
            ),
        )

        # Get predictions using trainer.predict()
        info_print("Generating zero-shot predictions...")
        predictions_output = zeroshot_trainer.predict(dset_test)

        # Extract predictions from output
        # predictions_output.predictions is a tuple: (forecasts, embeddings)
        predictions = predictions_output.predictions[0]

        # Convert to numpy if needed
        if hasattr(predictions, "cpu"):
            predictions = predictions.cpu().numpy()  # type: ignore[union-attr]
        elif not isinstance(predictions, np.ndarray):
            predictions = np.array(predictions)

        info_print(f"Zero-shot predictions shape (scaled): {predictions.shape}")

        # Inverse scale predictions back to original units
        # The preprocessor (tsp) was used to scale the data, so we use it to inverse scale
        predictions = self._inverse_scale_zero_shot_predictions(predictions, tsp)

        info_print(f"Zero-shot predictions shape (unscaled): {predictions.shape}")

        return predictions

    def _inverse_scale_zero_shot_predictions(
        self, predictions: np.ndarray, preprocessor: TimeSeriesPreprocessor
    ) -> np.ndarray:
        """Inverse scale zero-shot predictions back to original units.

        Args:
            predictions: Scaled predictions array
            preprocessor: The TimeSeriesPreprocessor used for scaling

        Returns:
            Predictions in original scale
        """
        from tsfm_public.toolkit.time_series_preprocessor import INTERNAL_ID_COLUMN

        if not preprocessor.scaling:
            info_print("Preprocessor scaling is disabled, returning predictions as-is")
            return predictions

        if len(preprocessor.target_scaler_dict) == 0:
            info_print("No scalers in preprocessor, returning predictions as-is")
            return predictions

        # Get the target scaler - for global scaling, key is '__id'
        scaler_key = INTERNAL_ID_COLUMN
        if scaler_key not in preprocessor.target_scaler_dict:
            # Try first available key
            scaler_key = next(iter(preprocessor.target_scaler_dict.keys()))

        scaler = preprocessor.target_scaler_dict[scaler_key]

        # Handle different prediction shapes
        original_shape = predictions.shape
        info_print(
            f"Inverse scaling zero-shot predictions with shape: {original_shape}"
        )

        # Reshape to 2D for sklearn scaler (samples, features)
        if len(original_shape) == 1:
            predictions_2d = predictions.reshape(-1, 1)
            needs_squeeze = True
        elif len(original_shape) == 2:
            predictions_2d = predictions
            needs_squeeze = False
        elif len(original_shape) == 3:
            # Shape: (samples, forecast_length, channels)
            n_samples, forecast_len, n_channels = original_shape
            predictions_2d = predictions.reshape(-1, n_channels)
            needs_squeeze = False
        else:
            info_print(
                f"Unexpected predictions shape: {original_shape}, returning as-is"
            )
            return predictions

        # Inverse transform
        try:
            # Only inverse scale the target column(s)
            n_target_cols = len(preprocessor.target_columns)
            predictions_unscaled = predictions_2d.copy()
            predictions_unscaled[:, :n_target_cols] = scaler.inverse_transform(
                predictions_2d[:, :n_target_cols]
            )

            # Reshape back to original shape
            if len(original_shape) == 3:
                predictions_unscaled = predictions_unscaled.reshape(original_shape)
            elif needs_squeeze:
                predictions_unscaled = predictions_unscaled.squeeze()

            info_print("Successfully inverse-scaled zero-shot predictions")
            return predictions_unscaled

        except Exception as e:
            info_print(f"Warning: Failed to inverse scale predictions: {e}")
            return predictions

    def get_ttm_specific_info(self) -> Dict[str, Any]:
        """Get TTM-specific model information.

        Returns:
            Dictionary containing model info with TTM-specific details
        """
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

    # TTM-specific private methods
    def _create_column_specifiers(self, data: pd.DataFrame) -> ColumnSpecifiers:
        """Create column specifiers for TimeSeriesPreprocessor.

        Args:
            data: DataFrame containing time series data

        Returns:
            ColumnSpecifiers with properly typed column configuration
        """
        # Default mappings - adapt these to your data structure
        # NOTE: target_columns are the ONLY columns that will be forecasted by the model
        # control_columns, observable_columns, and conditional_columns are used as INPUT features only
        column_specifiers: ColumnSpecifiers = {
            "id_columns": [ColumnNames.P_NUM.value],
            "timestamp_column": ColumnNames.DATETIME.value,
            "target_columns": [ColumnNames.BG.value],  # Only forecast blood glucose
            "observable_columns": [
                # Observable columns: known in the past, unknown in the future
                # These are used as inputs but NOT forecasted
                ColumnNames.STEPS.value,
                ColumnNames.COB.value,
                ColumnNames.CARB_AVAILABILITY.value,
                ColumnNames.INSULIN_AVAILABILITY.value,
                ColumnNames.IOB.value,
            ],
            "control_columns": [],  # Control columns: known in past AND future (we don't have any)
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
        """Get distributed training arguments.

        Returns:
            Dictionary of distributed training parameters for TrainingArguments
        """
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

    def _compute_trainer_metrics(self, eval_pred) -> Dict[str, Any]:
        """Compute evaluation metrics for Trainer.

        The HuggingFace Trainer passes an EvalPrediction object containing:
        - predictions: Model outputs (for TTM, this is a tuple of (forecasts, embeddings))
        - label_ids: Ground truth labels (requires label_names=["future_values"] in TrainingArguments)

        Args:
            eval_pred: EvalPrediction object from Trainer

        Returns:
            Dictionary containing computed metrics (mse, rmse, mae, mape)
        """
        try:
            # Extract predictions and labels from EvalPrediction
            predictions = eval_pred.predictions
            labels = eval_pred.label_ids
            # Log initial shapes for debugging
            debug_print(f"Raw predictions type: {type(predictions)}")
            debug_print(f"Raw labels type: {type(labels)}")

            # Handle TTM's output format: predictions is (forecasts, embeddings) tuple
            if isinstance(predictions, (tuple, list)):
                info_print(
                    f"Predictions is tuple/list with {len(predictions)} elements - extracting forecasts only"
                )
                if len(predictions) > 0 and hasattr(predictions[0], "shape"):
                    # First element is the forecasts, second is embeddings (discarded)
                    predictions = predictions[0]
                    info_print(f"Extracted forecasts shape: {predictions.shape}")
                else:
                    info_print(
                        "WARNING: Could not extract forecasts from tuple - using raw predictions"
                    )

            # Handle labels - may be tuple/list or direct array
            if isinstance(labels, (tuple, list)):
                debug_print(f"Labels is tuple/list with {len(labels)} elements")
                if len(labels) > 0 and hasattr(labels[0], "shape"):
                    labels = labels[0]
                    debug_print(f"Extracted labels shape: {labels.shape}")

            # Convert to numpy arrays
            if not isinstance(predictions, np.ndarray):
                predictions = np.array(predictions)
            if not isinstance(labels, np.ndarray):
                labels = np.array(labels)

            info_print(f"Final predictions shape: {predictions.shape}")
            info_print(f"Final labels shape: {labels.shape}")
            # Print first few values AFTER extraction to see actual scaled values
            info_print(
                f"  Predictions (first 5 of first sample): {predictions[0, :5, 0] if len(predictions.shape) == 3 else predictions[:5]}"
            )
            info_print(
                f"  Labels (first 5 of first sample): {labels[0, :5, 0] if len(labels.shape) == 3 else labels[:5]}"
            )

            # Check for empty labels (indicates label_names not configured properly)
            if labels.size == 0:
                error_print(
                    "Labels array is empty. Ensure TrainingArguments has "
                    "label_names=['future_values'] configured."
                )
                return {"custom_error": "Empty labels - check label_names config"}

            # Handle shape mismatch - predictions and labels should align
            # Predictions shape: (batch, forecast_length, num_output_channels)
            # Labels shape: (batch, forecast_length, num_channels) where num_channels >= num_output_channels
            if predictions.shape != labels.shape:
                # If predictions has fewer channels than labels (target_columns subset),
                # slice labels to match the number of output channels
                if (
                    len(predictions.shape) == 3
                    and len(labels.shape) == 3
                    and predictions.shape[:2] == labels.shape[:2]
                ):
                    num_output_channels = predictions.shape[2]
                    labels = labels[:, :, :num_output_channels]
                    debug_print(f"Sliced labels to match predictions: {labels.shape}")
                else:
                    error_print(
                        f"Shape mismatch: predictions {predictions.shape} vs "
                        f"labels {labels.shape}"
                    )
                    return {
                        "custom_error": f"Shape mismatch: {predictions.shape} vs {labels.shape}"
                    }

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

            info_print(f"Computed evaluation metrics: {metrics}")
            return metrics

        except Exception as e:
            error_print(f"\nError computing metrics: {str(e)}")
            import traceback

            debug_print(traceback.format_exc())
            return {"custom_error": str(e)}

    def _get_callbacks(self) -> List:
        """Get training callbacks.

        Returns:
            List of callback instances for Trainer
        """

        callbacks = []

        # Early stopping only works if evaluation is enabled
        # Since we use eval_strategy="no" for speed, skip early stopping
        # if self.config.early_stopping_patience > 0:
        #     callbacks.append(
        #         EarlyStoppingCallback(
        #             early_stopping_patience=self.config.early_stopping_patience,
        #             early_stopping_threshold=0.0,
        #         )
        #     )

        return callbacks

    def _create_inference_trainer(self, batch_size: Optional[int] = None) -> Trainer:
        """Create a Trainer instance configured for inference (predict/evaluate).

        Args:
            batch_size: Batch size for inference (defaults to config.batch_size)

        Returns:
            Configured Trainer instance for inference
        """
        import tempfile

        batch_size_to_use = (
            batch_size if batch_size is not None else self.config.batch_size
        )
        output_dir = getattr(self.config, "output_dir", tempfile.mkdtemp())

        return Trainer(
            model=self.model,
            args=TrainingArguments(
                output_dir=output_dir,
                per_device_eval_batch_size=batch_size_to_use,
                dataloader_num_workers=self.config.dataloader_num_workers,
                report_to="none",
                seed=42,  # For reproducibility
            ),
        )

    def _create_training_arguments(self, output_dir: str) -> TrainingArguments:
        """Create TrainingArguments for model training.

        Args:
            output_dir: Directory for checkpoints and logs

        Returns:
            Configured TrainingArguments instance
        """
        # Store checkpoints in a dedicated subdirectory to keep the output dir clean
        checkpoint_dir = os.path.join(output_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Base training arguments
        base_args = {
            "output_dir": checkpoint_dir,
            "learning_rate": self.config.learning_rate,
            "num_train_epochs": self.config.num_epochs,
            "per_device_train_batch_size": self.config.batch_size,
            "per_device_eval_batch_size": self.config.batch_size,
            "warmup_steps": self.config.warmup_steps,
            "weight_decay": self.config.weight_decay,
            "logging_dir": self.config.logging_dir
            if self.config.logging_dir
            else os.path.join(output_dir, "logs"),
            "logging_steps": 1000,  # Log every 1000 steps (reduces log verbosity significantly)
            "eval_strategy": "no",  # Disable evaluation during training for speed
            "save_strategy": "epoch",  # Only save at end of epoch
            "eval_steps": None,
            "save_steps": self.config.save_steps,
            "metric_for_best_model": self.config.metric_for_best_model,
            "greater_is_better": self.config.greater_is_better,
            "load_best_model_at_end": False,  # Disabled since eval is off
            "fp16": self.config.fp16,
            "dataloader_num_workers": self.config.dataloader_num_workers,
            # Tell Trainer that 'future_values' in the batch is the labels field
            # This ensures EvalPrediction.label_ids is populated correctly
            "label_names": ["future_values"],
            "use_cpu": self.config.use_cpu,
            "report_to": "none",  # Disable wandb/tensorboard by default
            "disable_tqdm": False,  # Keep progress bar enabled
            "logging_first_step": True,
            "logging_nan_inf_filter": False,
        }

        # Add distributed training arguments
        distributed_args = self._get_distributed_training_args()
        base_args.update(distributed_args)

        return TrainingArguments(**base_args)
