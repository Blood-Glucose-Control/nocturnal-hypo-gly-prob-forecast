"""
TTM (TinyTimeMixer) model implementation using the base TSFM framework.

This module provides a concrete implementation of TTM that inherits from
the base TSFM framework, demonstrating how to integrate existing models.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple, TypedDict, cast

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from transformers import (
    Trainer,
    TrainingArguments,
)

# Import your existing TTM-related modules
from tsfm_public import (
    TimeSeriesForecastingPipeline,
    TimeSeriesPreprocessor,
    get_datasets,
)
from tsfm_public.toolkit.get_model import get_model

from ...data.models import ColumnNames
from ...data.preprocessing.split_or_combine_patients import (
    reduce_features_multi_patient,
)
from ...utils.logging_helper import debug_print, error_print, info_print

# Local imports
from ..base import BaseTimeSeriesFoundationModel, ModelConfig, TrainingBackend
from ..base.registry import ModelRegistry
from .config import TTMConfig

logger = logging.getLogger(__name__)


def _validate_preprocessor_schema(preprocessor: TimeSeriesPreprocessor) -> None:
    """Validate preprocessor schema required by the current runtime.

    Unsupported checkpoint preprocessors are intentionally not shimmed. If a
    loaded checkpoint does not satisfy the current schema contract, fail fast
    with an actionable message to retrain with the current repository/runtime.
    """
    if not hasattr(preprocessor, "other_columns_to_scale"):
        raise ValueError(
            "TTM checkpoint preprocessor schema is unsupported by the current "
            "runtime (missing required attribute 'other_columns_to_scale'). "
            "Please retrain the Stage-1 checkpoint using the current repository "
            "before running Stage-2 personalization."
        )


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


@ModelRegistry.register("ttm")
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

    config_class = TTMConfig

    def __init__(self, config: TTMConfig):
        """Initialize the TTM forecaster.

        Args:
            config: TTM configuration object
        """
        if not isinstance(config, TTMConfig):
            if isinstance(config, ModelConfig):
                config = TTMConfig.from_dict(config.to_dict())
            else:
                raise TypeError(
                    "TTMForecaster requires a TTMConfig (or ModelConfig-compatible) "
                    f"instance, got {type(config)}"
                )

        super().__init__(config)

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
        """Return the training backend for this model family."""
        return TrainingBackend.TRANSFORMERS

    @property
    def supports_zero_shot(self) -> bool:
        """Return whether this model supports zero-shot inference."""
        return True

    @property
    def supports_probabilistic_forecast(self) -> bool:
        """Return whether this model supports probabilistic forecasts."""
        return False

    # Abstract method implementations

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
        _ = data
        if self.preprocessor is None:
            logger.warning(
                "No preprocessor available - predictions will be returned in SCALED units (z-scores). "
                "This will cause incorrect metrics if comparing to unscaled ground truth."
            )
            return predictions

        if not self.preprocessor.scaling:
            info_print("Scaling disabled, returning predictions as-is")
            return predictions

        if len(self.preprocessor.target_scaler_dict) == 0:
            logger.warning(
                "No scalers trained in preprocessor - predictions will be returned in SCALED units. "
                "The preprocessor may not have been fitted correctly during training."
            )
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

        # Log scaler parameters for debugging
        if hasattr(scaler, "mean_"):
            scaler_mean = (
                scaler.mean_[0] if hasattr(scaler.mean_, "__len__") else scaler.mean_
            )
            scaler_scale = (
                scaler.scale_[0] if hasattr(scaler.scale_, "__len__") else scaler.scale_
            )
            debug_print(
                f"Using scaler with mean={scaler_mean:.4f}, scale={scaler_scale:.4f} "
                f"(key: {scaler_key})"
            )

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
            # Only inverse scale the target channel(s) - typically just channel 0
            predictions_2d = predictions[:, :, 0].reshape(-1, 1)
        else:
            raise ValueError(
                "Unsupported prediction shape for inverse scaling: "
                f"{original_shape}. Expected 1D, 2D, or 3D array."
            )

        # Inverse transform using the scaler
        predictions_unscaled = scaler.inverse_transform(predictions_2d)

        # Reshape back to original shape
        if len(original_shape) == 1:
            return predictions_unscaled.flatten()
        if len(original_shape) == 2:
            return predictions_unscaled.reshape(original_shape[0], original_shape[1])
        if len(original_shape) == 3:
            # Put unscaled values back into channel 0, keep other channels as-is
            result = predictions.copy()
            result[:, :, 0] = predictions_unscaled.reshape(
                original_shape[0], original_shape[1]
            )
            return result

        raise RuntimeError(
            f"Unhandled prediction shape after inverse scaling: {original_shape}"
        )

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
        data = self._normalize_training_input(train_data)
        column_specifiers = self._get_or_create_column_specifiers(data)
        self._log_column_specifiers(column_specifiers)
        preprocessor = self._ensure_training_preprocessor(column_specifiers)

        logger.info("\n")
        info_print("Splitting data into train/val/test sets...")
        info_print(f"  Split config: {self.config.split_config}")
        try:
            dset_train, dset_val, dset_test = self._build_training_datasets(
                data=data,
                preprocessor=preprocessor,
            )
            train_loader = self._build_data_loader(
                dset_train,
                shuffle=True,
            )
            val_loader = self._build_optional_data_loader(
                dset_val,
                shuffle=False,
            )
            test_loader = self._build_optional_data_loader(
                dset_test,
                shuffle=False,
            )
            self._log_dataset_sizes(dset_train, dset_val, dset_test)

            return train_loader, val_loader, test_loader

        except Exception as e:
            error_print(f"Failed to prepare data: {str(e)}")
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
        self._configure_training_environment()

        info_print("Starting TTM training using HuggingFace Trainer...")
        # Prepare data loaders (splits based on config)
        train_loader, val_loader, test_loader = self._prepare_training_data(train_data)

        # Create training arguments
        training_args = self._create_training_arguments(output_dir)

        # Create trainer
        trainer = self._build_trainer(
            training_args=training_args,
            train_loader=train_loader,
            val_loader=val_loader,
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

        # Save the preprocessor (critical for inference on holdout patients)
        self._save_checkpoint(output_dir)

        training_history = self._capture_training_history(trainer)
        test_metrics = self._evaluate_test_loader(trainer, test_loader)

        return {
            "train_metrics": train_result.metrics,
            "test_metrics": test_metrics,
            "training_history": training_history,
        }

    def _predict(
        self,
        data: Any,
        quantile_levels: Optional[List[float]] = None,
        *,
        batch_size: Optional[int] = None,
        inverse_scale: bool = True,
        **kwargs,
    ) -> np.ndarray:
        """Make predictions on new data using TTM pipeline.

        Branches on is_fitted to select the inference path:
        - Fitted: uses TimeSeriesForecastingPipeline with the preprocessor
          (external scaling fitted during training).
        - Not fitted (zero-shot): uses pipeline without preprocessor;
          TTM's internal RevIN handles per-window standardization.

        Args:
            data: Input data for prediction
            batch_size: Batch size for prediction
            inverse_scale: If True, inverse transform predictions to original scale.
                          Requires preprocessor to have been fitted during training.

        Returns:
            Predictions as numpy array (in original scale if inverse_scale=True).
        """
        self._warn_quantiles_not_supported(quantile_levels)

        if kwargs:
            logger.debug("Ignoring unsupported TTM predict kwargs: %s", list(kwargs))

        model = self._require_initialized_model()

        if self.is_fitted:
            # Fine-tuned path: preprocessor handles scaling + inverse scaling
            if self.preprocessor is None:
                raise RuntimeError(
                    "Model is marked as fitted but preprocessor is None. "
                    "The checkpoint was likely saved without a preprocessor. "
                    "Re-train the model or use zero-shot inference instead."
                )
            pipeline = TimeSeriesForecastingPipeline(
                model=model,
                feature_extractor=self.preprocessor,
                explode_forecasts=True,
                inverse_scale_outputs=inverse_scale,
                batch_size=batch_size or self.config.batch_size,
            )
            forecast_df = pipeline(data)
            target_col = self.preprocessor.target_columns[0]
            predictions = forecast_df[target_col].values
        else:
            # Zero-shot path: no preprocessor, TTM's internal RevIN
            # handles per-window standardization automatically
            if not isinstance(data, pd.DataFrame):
                raise TypeError(
                    "TTM zero-shot predict expects a pandas DataFrame, got "
                    f"{type(data)}"
                )
            pipeline, target_col = self._build_zero_shot_pipeline_for_data(
                data=data,
                model=model,
            )
            forecast_df = pipeline(data)
            predictions = forecast_df[target_col].values

        return predictions

    # TTM-specific public methods
    # NOTE: evaluate() is inherited from BaseTimeSeriesFoundationModel
    # It calls predict() and computes metrics using _compute_metrics()

    def _predict_batch(
        self,
        data: pd.DataFrame,
        episode_col: str,
        quantile_levels=None,
    ) -> Dict[str, np.ndarray]:
        """Batch prediction using TimeSeriesForecastingPipeline.

        For zero-shot mode, passes all episodes to the pipeline in a single
        call with ``episode_col`` included as an id column so that TTM's
        internal DataLoader can process them together.

        For fitted mode the preprocessor was fitted without ``episode_col``
        as an id column, so falls back to the default sequential loop.

        Args:
            data: Panel DataFrame containing episode_col and the columns
                required by the model (timestamp, target, covariates).
            episode_col: Column name that identifies individual episodes.

        Returns:
            Dict mapping episode ID (as str) to 1-D numpy forecast array.
        """
        episode_ids = data[episode_col].unique()
        self._warn_quantiles_not_supported(quantile_levels, source="batch predict")

        # Zero-shot path: include episode_col in id_columns so the pipeline
        # groups episodes correctly in a single batched forward pass.
        if self.preprocessor is None and self.config.training_mode == "zero_shot":
            model = self._require_initialized_model()
            pipeline, target_col = self._build_zero_shot_pipeline_for_data(
                data=data,
                model=model,
                episode_col=episode_col,
            )
            forecast_df = pipeline(data)

            results: Dict[str, np.ndarray] = {}
            for ep_id in episode_ids:
                mask = forecast_df[episode_col] == ep_id
                if mask.any():
                    results[str(ep_id)] = forecast_df.loc[mask, target_col].values
            return results

        # Fitted path: delegate to base class sequential loop.
        return super()._predict_batch(data, episode_col)

    def _save_checkpoint(self, output_dir: str) -> None:
        """Save model checkpoint and preprocessor.

        Args:
            output_dir: Directory path for saving checkpoint
        """
        import pickle

        if self.model is not None and hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(output_dir)  # type: ignore[union-attr]
            info_print(f"TTM model saved to {output_dir}")

        self._save_preprocessor_checkpoint(output_dir, pickle_module=pickle)

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

            self.preprocessor = self._load_preprocessor_checkpoint(
                model_dir, pickle_module=pickle
            )

            if self.preprocessor is not None:
                _validate_preprocessor_schema(self.preprocessor)

            # Only mark as fitted if the preprocessor was also successfully loaded.
            # The fitted inference path in _predict() unconditionally dereferences
            # self.preprocessor, so setting is_fitted=True without a preprocessor
            # would cause an AttributeError at inference time.
            if self.preprocessor is not None:
                self.is_fitted = True
                info_print("Model marked as fitted (preprocessor loaded successfully).")
            else:
                self.is_fitted = False
                logger.warning(
                    "Model checkpoint loaded but is_fitted=False: preprocessor is missing. "
                    "Falling back to zero-shot inference path (TTM internal RevIN only)."
                )

        except Exception as e:
            error_print(f"Failed to load model checkpoint: {str(e)}")
            raise

    # TTM-specific private methods
    def _require_initialized_model(self) -> Any:
        """Return model object or raise if weights are not initialized."""
        if self.model is None:
            raise RuntimeError("TTM model weights are not initialized.")
        return cast(Any, self.model)

    def _get_or_create_column_specifiers(self, data: pd.DataFrame) -> ColumnSpecifiers:
        """Get cached column specifiers, creating them lazily on first use."""
        if self.column_specifiers is None:
            self.column_specifiers = self._create_column_specifiers(data)
        return self.column_specifiers

    def _resolve_target_columns(
        self,
        *,
        column_specifiers: ColumnSpecifiers,
        data: pd.DataFrame,
    ) -> List[str]:
        """Resolve target columns and fail fast when no valid target remains."""
        target_columns = list(column_specifiers.get("target_columns", []))
        if target_columns:
            return target_columns

        expected = self.config.target_features
        raise ValueError(
            f"target_columns is empty after filtering: none of the configured "
            f"target features {expected} were found (or had non-NaN values) in "
            f"the input data. Available columns: {list(data.columns)}"
        )

    def _normalize_training_input(self, train_data: Any) -> pd.DataFrame:
        """Normalize supported training inputs to a DataFrame."""
        if isinstance(train_data, pd.DataFrame):
            return train_data
        if isinstance(train_data, dict):
            reduced = reduce_features_multi_patient(
                patients_dict=train_data,
                resolution_min=self.config.resolution_min,
                x_features=self.config.input_features,
                y_feature=self.config.target_features,
            )
            info_print(
                "Converted multi-patient dict to DataFrame\n"
                f"dataset now has the following columns available: {reduced.columns}"
            )
            return reduced.reset_index()
        raise ValueError(
            f"train_data must be a DataFrame or dict, got {type(train_data)}"
        )

    @staticmethod
    def _log_column_specifiers(column_specifiers: ColumnSpecifiers) -> None:
        info_print("Using column specifiers:")
        for key, value in column_specifiers.items():
            info_print(f"  {key}: {value}")

    def _ensure_training_preprocessor(
        self, column_specifiers: ColumnSpecifiers
    ) -> TimeSeriesPreprocessor:
        """Create or validate the training preprocessor."""
        if self.preprocessor is None:
            self.preprocessor = TimeSeriesPreprocessor(
                **column_specifiers,
                context_length=self.config.context_length,
                prediction_length=self.config.forecast_length,
                scaling=True,
                # Use a global scaler across patients to support holdout/new patients.
                scaling_id_columns=[],
                encode_categorical=False,
                scaler_type=self.config.get_scaler_type().value,  # type: ignore[arg-type]
            )
        else:
            _validate_preprocessor_schema(self.preprocessor)

        return self.preprocessor

    def _build_training_datasets(
        self,
        *,
        data: pd.DataFrame,
        preprocessor: TimeSeriesPreprocessor,
    ) -> Tuple[Any, Optional[Any], Optional[Any]]:
        """Build train/val/test datasets from normalized training input."""
        split_config = cast(
            Dict[str, List[int | float] | float],
            self.config.split_config,
        )
        return get_datasets(  # type: ignore[misc]
            ts_preprocessor=preprocessor,
            dataset=data,
            split_config=split_config,
            fewshot_fraction=self.config.fewshot_percent / 100,
            fewshot_location="last",
        )

    def _build_data_loader(self, dataset: Any, *, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.dataloader_num_workers,
        )

    def _build_optional_data_loader(
        self,
        dataset: Optional[Any],
        *,
        shuffle: bool,
    ) -> Optional[DataLoader]:
        if dataset is None:
            return None
        return self._build_data_loader(dataset, shuffle=shuffle)

    @staticmethod
    def _log_dataset_sizes(
        train_dataset: Any,
        val_dataset: Optional[Any],
        test_dataset: Optional[Any],
    ) -> None:
        info_print("Data preparation complete:")
        info_print(f"  Train samples: {len(train_dataset) if train_dataset else 0:,}")
        info_print(f"  Val samples: {len(val_dataset) if val_dataset else 0:,}")
        info_print(f"  Test samples: {len(test_dataset) if test_dataset else 0:,}")

    def _configure_training_environment(self) -> None:
        """Set runtime environment knobs for stable Trainer execution."""
        # Reduce tqdm log noise.
        os.environ["TQDM_MININTERVAL"] = "30"

        if "PYTORCH_ALLOC_CONF" in os.environ:
            return

        os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
        import torch

        if torch.cuda.is_initialized():
            logger.warning(
                "PYTORCH_ALLOC_CONF was set after CUDA initialization. "
                "The 'expandable_segments:True' setting may not take effect. "
                "For reliable memory fragmentation prevention, set "
                "PYTORCH_ALLOC_CONF in your shell before running Python."
            )

    def _build_trainer(
        self,
        *,
        training_args: TrainingArguments,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
    ) -> Trainer:
        return Trainer(
            model=self._require_initialized_model(),
            args=training_args,
            train_dataset=train_loader.dataset,
            eval_dataset=val_loader.dataset if val_loader else None,
            compute_metrics=self._compute_trainer_metrics,
            callbacks=self._get_callbacks(),
        )

    def _capture_training_history(self, trainer: Trainer) -> Dict[str, Any]:
        """Capture in-memory trainer history for metadata/reporting."""
        if not hasattr(trainer, "state") or trainer.state is None:
            info_print("Warning: Could not access trainer.state")
            return {}

        history = {
            "log_history": trainer.state.log_history,
            "best_metric": trainer.state.best_metric,
            "best_model_checkpoint": trainer.state.best_model_checkpoint,
            "global_step": trainer.state.global_step,
            "epoch": trainer.state.epoch,
        }
        info_print("Captured training history from trainer state")
        info_print(f"  Total log entries: {len(trainer.state.log_history)}")
        return history

    def _evaluate_test_loader(
        self,
        trainer: Trainer,
        test_loader: Optional[DataLoader],
    ) -> Dict[str, Any]:
        """Evaluate trainer on the test dataset when available."""
        if test_loader is None:
            return {}

        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            test_metrics = trainer.evaluate(eval_dataset=test_loader.dataset)
            info_print(f"Test metrics: {test_metrics}")
            return cast(Dict[str, Any], test_metrics)
        except torch.cuda.OutOfMemoryError:
            info_print(
                "Warning: Test evaluation skipped due to GPU OOM. "
                "This is non-fatal — full holdout evaluation runs separately in step 8."
            )
            torch.cuda.empty_cache()
            return {}

    def _warn_quantiles_not_supported(
        self,
        quantile_levels: Optional[List[float]],
        *,
        source: str = "predict",
    ) -> None:
        if quantile_levels is None:
            return
        logger.warning(
            "TTM %s does not provide quantile outputs; returning point forecasts only.",
            source,
        )

    def _build_zero_shot_pipeline_for_data(
        self,
        *,
        data: pd.DataFrame,
        model: Any,
        episode_col: Optional[str] = None,
    ) -> Tuple[TimeSeriesForecastingPipeline, str]:
        """Build a zero-shot pipeline and return the resolved target column."""
        column_specifiers = self._get_or_create_column_specifiers(data)
        id_columns: List[str] = list(column_specifiers.get("id_columns", []))
        if episode_col and episode_col not in id_columns:
            id_columns = [episode_col] + id_columns

        target_columns = self._resolve_target_columns(
            column_specifiers=column_specifiers,
            data=data,
        )
        pipeline = self._build_zero_shot_pipeline(
            model=model,
            column_specifiers=column_specifiers,
            id_columns=id_columns,
            target_columns=target_columns,
        )
        return pipeline, target_columns[0]

    @staticmethod
    def _preprocessor_paths(model_dir: str) -> Tuple[str, str]:
        """Return supported preprocessor artifact locations for a checkpoint."""
        return (
            os.path.join(model_dir, "preprocessor.pkl"),
            os.path.join(model_dir, "model.pt", "preprocessor.pkl"),
        )

    def _save_preprocessor_checkpoint(
        self, output_dir: str, *, pickle_module: Any
    ) -> None:
        """Persist preprocessor artifacts to checkpoint-compatible locations."""
        if self.preprocessor is None:
            logger.warning(
                "Preprocessor is None - not saved. "
                "This will cause inference to return scaled predictions instead of original units."
            )
            return

        root_path, model_pt_path = self._preprocessor_paths(output_dir)
        with open(root_path, "wb") as f:
            pickle_module.dump(self.preprocessor, f)
        info_print(f"Preprocessor saved to {root_path}")

        model_pt_dir = os.path.dirname(model_pt_path)
        if os.path.exists(model_pt_dir):
            with open(model_pt_path, "wb") as f:
                pickle_module.dump(self.preprocessor, f)
            info_print(f"Preprocessor also saved to {model_pt_path}")

    def _load_preprocessor_checkpoint(
        self, model_dir: str, *, pickle_module: Any
    ) -> Optional[TimeSeriesPreprocessor]:
        """Load preprocessor artifact from known checkpoint locations."""
        root_path, model_pt_path = self._preprocessor_paths(model_dir)
        for path in (root_path, model_pt_path):
            if not os.path.exists(path):
                continue
            with open(path, "rb") as f:
                loaded = cast(TimeSeriesPreprocessor, pickle_module.load(f))
            info_print(f"Preprocessor loaded from {path}")
            return loaded

        logger.warning(
            f"No preprocessor found at {model_dir}. "
            "Predictions will return SCALED values (z-scores) instead of original units. "
            "This will cause incorrect metrics if comparing to unscaled ground truth. "
            "Ensure preprocessor.pkl was saved during training."
        )
        return None

    def _build_zero_shot_pipeline(
        self,
        *,
        model: Any,
        column_specifiers: ColumnSpecifiers,
        id_columns: List[str],
        target_columns: List[str],
    ) -> TimeSeriesForecastingPipeline:
        """Create the zero-shot inference pipeline with shared settings."""
        return TimeSeriesForecastingPipeline(
            model=model,
            timestamp_column=column_specifiers.get(
                "timestamp_column", ColumnNames.DATETIME.value
            ),
            id_columns=id_columns,
            target_columns=target_columns,
            observable_columns=column_specifiers.get("observable_columns", []),
            explode_forecasts=True,
            freq=f"{self.config.resolution_min}min",
        )

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

        # Use input_features from config instead of hardcoded list
        observable_cols = (
            self.config.input_features if self.config.input_features else []
        )

        column_specifiers: ColumnSpecifiers = {
            "id_columns": [ColumnNames.P_NUM.value],
            "timestamp_column": ColumnNames.DATETIME.value,
            "target_columns": self.config.target_features,  # Use config target features
            "observable_columns": observable_cols,  # Use config input features
            "control_columns": [],  # Control columns: known in past AND future (we don't have any)
            "conditional_columns": [],
            "static_categorical_columns": [],
        }

        # Filter to only include columns that exist in the data AND have
        # at least some non-NaN values (e.g. Brown 2019 has cob/carb_availability
        # columns as all-NaN placeholders because no meal data exists)
        available_columns = set(data.columns)

        for key, columns in column_specifiers.items():
            if isinstance(columns, list):
                column_specifiers[key] = [
                    col
                    for col in columns
                    if col in available_columns and not data[col].isna().all()
                ]

        return column_specifiers

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

            predictions_array = np.asarray(predictions)
            labels_array = np.asarray(labels)

            info_print(f"Final predictions shape: {predictions_array.shape}")
            info_print(f"Final labels shape: {labels_array.shape}")
            # Print first few values AFTER extraction to see actual scaled values
            info_print(
                "  Predictions (first 5 of first sample): "
                f"{predictions_array[0, :5, 0] if predictions_array.ndim == 3 else predictions_array[:5]}"
            )
            info_print(
                "  Labels (first 5 of first sample): "
                f"{labels_array[0, :5, 0] if labels_array.ndim == 3 else labels_array[:5]}"
            )

            # Check for empty labels (indicates label_names not configured properly)
            if labels_array.size == 0:
                error_print(
                    "Labels array is empty. Ensure TrainingArguments has "
                    "label_names=['future_values'] configured."
                )
                return {"custom_error": "Empty labels - check label_names config"}

            # Handle shape mismatch - predictions and labels should align
            # Predictions shape: (batch, forecast_length, num_output_channels)
            # Labels shape: (batch, forecast_length, num_channels) where num_channels >= num_output_channels
            if predictions_array.shape != labels_array.shape:
                # If predictions has fewer channels than labels (target_columns subset),
                # slice labels to match the number of output channels
                if (
                    predictions_array.ndim == 3
                    and labels_array.ndim == 3
                    and predictions_array.shape[:2] == labels_array.shape[:2]
                ):
                    num_output_channels = predictions_array.shape[2]
                    labels_array = labels_array[:, :, :num_output_channels]
                    debug_print(
                        f"Sliced labels to match predictions: {labels_array.shape}"
                    )
                else:
                    error_print(
                        f"Shape mismatch: predictions {predictions_array.shape} vs "
                        f"labels {labels_array.shape}"
                    )
                    return {
                        "custom_error": "Shape mismatch: "
                        f"{predictions_array.shape} vs {labels_array.shape}"
                    }

            # Compute metrics
            mse = np.mean((predictions_array - labels_array) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(predictions_array - labels_array))

            # MAPE with handling for zero values
            mask = labels_array != 0
            mape = (
                np.mean(
                    np.abs(
                        (predictions_array[mask] - labels_array[mask])
                        / labels_array[mask]
                    )
                    * 100
                )
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

        return TrainingArguments(**base_args)
