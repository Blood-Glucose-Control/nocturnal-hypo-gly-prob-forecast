"""
Moment model implementation using the base TSFM framework.

This module provides a concrete implementation of Moment that inherits from
the base TSFM framework, demonstrating how to integrate foundation models.
"""

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers.training_args import TrainingArguments

# Local imports
from src.data.models import ColumnNames
from src.models.base import BaseTimeSeriesFoundationModel, TrainingBackend
from src.models.moment.config import MomentConfig
from src.utils.logging_helper import info_print, error_print

# MOMENT sequence length limit (from notebook)
MOMENT_MAX_LEN = 512


def _optional_moment_import():
    """Import MOMENTPipeline if momentfm is installed."""
    try:
        from momentfm import MOMENTPipeline
        return MOMENTPipeline
    except ImportError as e:
        raise ImportError(
            "MOMENT model requires the 'momentfm' package. "
            "Install with: pip install momentfm"
        ) from e


class _ContextTargetDataset(Dataset):
    """Dataset of (context, target) arrays for Moment forecasting."""

    def __init__(
        self,
        contexts: List[np.ndarray],
        targets: List[np.ndarray],
        context_lengths: Optional[List[int]] = None,
    ):
        self.contexts = contexts
        self.targets = targets
        self.context_lengths = context_lengths or [c.shape[0] for c in contexts]

    def __len__(self) -> int:
        return len(self.contexts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "context": self.contexts[idx].astype(np.float32),
            "target": self.targets[idx].astype(np.float32),
            "context_len": int(self.context_lengths[idx]),
        }


class MomentForecaster(BaseTimeSeriesFoundationModel):
    """Moment forecaster implementation using the base TSFM framework.

    Moment is a time series foundation model (CMU) with transformer-based
    masked reconstruction. This implementation supports zero-shot forecasting
    via MOMENTPipeline.forecast() (context + mask -> impute future).

    Attributes:
        config: Moment-specific configuration (MomentConfig instance).
        preprocessor: Unused; scaling is per-context in forecast.
        column_specifiers: Unused; target column from config.
    """

    def __init__(
        self,
        config: MomentConfig,
        lora_config=None,
        distributed_config=None,
    ):
        if not isinstance(config, MomentConfig):
            essential_params = {
                "model_path": getattr(config, "model_path", "AutonLab/MOMENT-1-large"),
                "context_length": getattr(config, "context_length", 512),
                "forecast_length": getattr(config, "forecast_length", 96),
                "learning_rate": getattr(config, "learning_rate", 1e-4),
                "batch_size": getattr(config, "batch_size", 32),
                "num_epochs": getattr(config, "num_epochs", 10),
            }
            config = MomentConfig(**essential_params)

        # Set device BEFORE super().__init__() because _initialize_model() is called during init
        use_cpu = getattr(config, "use_cpu", False)
        self._device = torch.device("cuda" if torch.cuda.is_available() and not use_cpu else "cpu")
        
        super().__init__(config, lora_config, distributed_config)
        self.config: MomentConfig = self.config
        self.preprocessor = None
        self.column_specifiers = None
        self._target_col: Optional[str] = None

    # --- Properties (base contract) ---
    @property
    def training_backend(self) -> TrainingBackend:
        # Check training_mode for zero-shot, otherwise use training_backend from config
        if hasattr(self.config, "training_mode") and self.config.training_mode == "zero_shot":
            return TrainingBackend.TRANSFORMERS  # Zero-shot uses transformers backend
        return getattr(
            self.config,
            "training_backend",
            TrainingBackend.TRANSFORMERS,
        )

    @property
    def supports_lora(self) -> bool:
        return True

    # --- Model init ---
    def _initialize_model(self) -> None:
        if not self.config.model_path:
            raise ValueError("model_path must be specified in config")

        info_print(f"Initializing Moment model from {self.config.model_path}")

        MOMENTPipeline = _optional_moment_import()

        try:
            self.model = MOMENTPipeline.from_pretrained(
                self.config.model_path,
                model_kwargs={"enable_gradient_checkpointing": True},
            )
            self.model.init()
            self.model.to(self._device)
            self.model.eval()
            # Zero-shot: model is ready for inference without fit()
            self.is_fitted = True
            info_print("Moment model initialized (zero-shot ready)")
        except Exception as e:
            error_print(f"Failed to initialize Moment model: {str(e)}")
            raise

    # --- Forecast helpers (notebook logic) ---
    def _forecast_single(
        self,
        context: np.ndarray,
        prediction_length: int,
        scaler: Optional[Any] = None,
    ) -> np.ndarray:
        """Single univariate forecast: scale context, mask future, impute, unscale."""
        from sklearn.preprocessing import StandardScaler

        scaler = scaler or StandardScaler()
        context_flat = np.asarray(context, dtype=np.float64).reshape(-1, 1)
        context_scaled = scaler.fit_transform(context_flat).flatten()  # type: ignore[union-attr]

        full_len = len(context_scaled) + prediction_length
        if full_len > MOMENT_MAX_LEN:
            keep = MOMENT_MAX_LEN - prediction_length
            context_scaled = context_scaled[-keep:]
            full_len = MOMENT_MAX_LEN

        input_seq = np.zeros(full_len, dtype=np.float32)
        input_seq[: len(context_scaled)] = context_scaled

        # [1, 1, T]
        x_enc = (
            torch.tensor(input_seq, dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(self._device)
        )
        # [1, T]: 1 = observed, 0 = to predict
        input_mask = torch.ones(full_len, dtype=torch.long, device=self._device)
        input_mask[len(context_scaled) :] = 0
        input_mask = input_mask.unsqueeze(0)

        if self.model is None:
            raise ValueError("Model not initialized")
        with torch.no_grad():
            output = self.model.forecast(x_enc=x_enc, input_mask=input_mask)  # type: ignore[call-overload]

        forecast_scaled = output.forecast.cpu().numpy()[0, 0, :]  # type: ignore[union-attr, call-overload]
        if len(forecast_scaled) > prediction_length:
            forecast_scaled = forecast_scaled[-prediction_length:]
        pred = scaler.inverse_transform(forecast_scaled.reshape(-1, 1)).flatten()  # type: ignore[union-attr]
        return pred.astype(np.float32)

    def _forecast_batch(
        self,
        contexts: np.ndarray,
        prediction_length: int,
        context_lengths: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Batch of forecasts (loop over batch to match notebook scaling)."""
        from sklearn.preprocessing import StandardScaler

        preds = []
        for i in range(contexts.shape[0]):
            ctx = np.asarray(contexts[i], dtype=np.float64)
            if context_lengths is not None and i < len(context_lengths):
                L = int(context_lengths[i])
                if L < ctx.shape[0]:
                    ctx = ctx[-L:]
            pred = self._forecast_single(ctx, prediction_length, scaler=StandardScaler())
            preds.append(pred)
        return np.stack(preds, axis=0)

    # --- Data: context/target pairs ---
    def _get_target_column(self, df: pd.DataFrame) -> str:
        if self._target_col is not None and self._target_col in df.columns:
            return self._target_col
        col = getattr(
            self.config.data_config,
            "target_features",
            [None],
        )
        if col and len(col) > 0 and col[0] is not None and col[0] in df.columns:
            self._target_col = col[0]
            return col[0]
        bg = ColumnNames.BG.value
        if bg in df.columns:
            self._target_col = bg
            return bg
        raise ValueError(
            f"Target column not found in DataFrame. Columns: {list(df.columns)}"
        )

    def _get_context_target_pairs(
        self,
        data: Any,
        context_length: Optional[int] = None,
        forecast_length: Optional[int] = None,
        require_target: bool = True,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Build list of (context, target) from dataset name, DataFrame, or dict of DataFrames."""
        ctx_len = context_length or self.config.context_length
        fcast_len = forecast_length or self.config.forecast_length

        if isinstance(data, str):
            from src.data.diabetes_datasets.data_loader import get_loader
            # Type checker sees overloads with Literal types, but we accept any str at runtime
            loader = get_loader(  # type: ignore[call-overload, assignment]
                data_source_name=data,  # type: ignore[arg-type]
                dataset_type="train",
                use_cached=True,
            )
            # Use validation data for eval-style usage
            if hasattr(loader, "validation_data") and loader.validation_data:
                data = loader.validation_data
            else:
                data = getattr(loader, "processed_data", None) or getattr(loader, "train_data", None)
            if data is None:
                raise ValueError(f"No data available from loader for {data}")

        pairs: List[Tuple[np.ndarray, np.ndarray]] = []

        if isinstance(data, dict):
            # patient_id -> DataFrame
            from src.data.preprocessing.time_processing import iter_daily_context_forecast_splits
            for _pid, patient_df in data.items():
                target_col = self._get_target_column(patient_df)
                for daytime, nocturnal in iter_daily_context_forecast_splits(patient_df):
                    ctx = daytime[target_col].values
                    tgt = nocturnal[target_col].values[:fcast_len]
                    if np.isnan(ctx).any() or np.isnan(tgt).any():
                        continue
                    if len(ctx) < 10 or len(tgt) < fcast_len:
                        continue
                    if len(ctx) > ctx_len:
                        ctx = ctx[-ctx_len:]
                    pairs.append((ctx.astype(np.float32), tgt.astype(np.float32)))
        elif isinstance(data, pd.DataFrame):
            target_col = self._get_target_column(data)
            # Single context window (e.g. from holdout_eval): exactly ctx_len rows, no future target
            if len(data) == ctx_len:
                ctx = data[target_col].values.astype(np.float32)
                if np.isnan(ctx).any():
                    return pairs  # No valid pair
                if len(ctx) < 10:
                    return pairs
                # Dummy target for loader; only context is used in predict()
                dummy_tgt = np.zeros(fcast_len, dtype=np.float32)
                pairs.append((ctx, dummy_tgt))
                return pairs
            if not require_target:
                # Single long series: last ctx_len -> predict next fcast_len
                vals = data[target_col].dropna().values
                if len(vals) >= ctx_len + fcast_len:
                    ctx = vals[-ctx_len - fcast_len : -fcast_len]
                    tgt = vals[-fcast_len:]
                    pairs.append((ctx.astype(np.float32), tgt.astype(np.float32)))
            else:
                from src.data.preprocessing.time_processing import iter_daily_context_forecast_splits

                if ColumnNames.P_NUM.value in data.columns:
                    for _pid, patient_df in data.groupby(ColumnNames.P_NUM.value):
                        for daytime, nocturnal in iter_daily_context_forecast_splits(patient_df):
                            ctx = daytime[target_col].values
                            tgt = nocturnal[target_col].values[:fcast_len]
                            if np.isnan(ctx).any() or np.isnan(tgt).any():
                                continue
                            if len(ctx) < 10 or len(tgt) < fcast_len:
                                continue
                            if len(ctx) > ctx_len:
                                ctx = ctx[-ctx_len:]
                            pairs.append((ctx.astype(np.float32), tgt.astype(np.float32)))
                else:
                    # Single patient
                    for daytime, nocturnal in iter_daily_context_forecast_splits(data):
                        ctx = daytime[target_col].values
                        tgt = nocturnal[target_col].values[:fcast_len]
                        if np.isnan(ctx).any() or np.isnan(tgt).any():
                            continue
                        if len(ctx) < 10 or len(tgt) < fcast_len:
                            continue
                        if len(ctx) > ctx_len:
                            ctx = ctx[-ctx_len:]
                        pairs.append((ctx.astype(np.float32), tgt.astype(np.float32)))
        else:
            raise ValueError(
                "data must be a dataset name (str), a DataFrame, or a dict of DataFrames"
            )

        return pairs

    def _prepare_training_data(
        self,
        train_data: Any,
        batch_size: Optional[int] = None,
    ) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
        """Build DataLoaders of (context, target) batches with train/val/test splitting.
        
        Splits data based on self.config.split_config if available, otherwise uses all data for training.
        """
        pairs = self._get_context_target_pairs(train_data, require_target=True)
        if not pairs:
            raise ValueError("No (context, target) pairs produced from train_data")

        # Split data if split_config is provided and we have multiple pairs (skip for single-window inference)
        split_config = getattr(self.config, "split_config", None)
        if split_config is None and hasattr(self.config, "data_config"):
            split_config = getattr(self.config.data_config, "split_config", None)
        
        train_pairs = pairs
        val_pairs = []
        test_pairs = []
        
        if split_config and len(pairs) > 1:
            import random
            random.seed(42)  # For reproducibility
            random.shuffle(pairs)
            
            train_ratio = split_config.get("train", 0.7)
            val_ratio = split_config.get("val", 0.2)
            test_ratio = split_config.get("test", 0.1)
            
            # Normalize ratios
            total = train_ratio + val_ratio + test_ratio
            train_ratio /= total
            val_ratio /= total
            test_ratio /= total
            
            n_total = len(pairs)
            n_train = int(n_total * train_ratio)
            n_val = int(n_total * val_ratio)
            
            train_pairs = pairs[:n_train]
            val_pairs = pairs[n_train:n_train + n_val]
            test_pairs = pairs[n_train + n_val:]
            
            info_print(f"Data split: {len(train_pairs)} train, {len(val_pairs)} val, {len(test_pairs)} test")
        
        def _create_loader(pair_list, shuffle=False):
            if not pair_list:
                return None
            
            contexts = [p[0] for p in pair_list]
            targets = [p[1] for p in pair_list]
            context_lengths = [c.shape[0] for c in contexts]
            max_ctx = max(context_lengths) if contexts else 0
            fcast_len = targets[0].shape[0] if targets else 0
            
            ctx_padded = np.zeros((len(contexts), max_ctx), dtype=np.float32)
            tgt_stacked = np.zeros((len(targets), fcast_len), dtype=np.float32)
            for i, (c, t) in enumerate(zip(contexts, targets)):
                ctx_padded[i, -len(c) :] = c
                tgt_stacked[i] = t

            dataset = _ContextTargetDataset(
                [ctx_padded[i] for i in range(len(ctx_padded))],
                [tgt_stacked[i] for i in range(len(tgt_stacked))],
                context_lengths=context_lengths,
            )
            bs = batch_size if batch_size is not None else self.config.batch_size
            return DataLoader(
                dataset,
                batch_size=bs,
                shuffle=shuffle,
                num_workers=0,
            )
        
        train_loader = _create_loader(train_pairs, shuffle=True)
        val_loader = _create_loader(val_pairs, shuffle=False)
        test_loader = _create_loader(test_pairs, shuffle=False)
        
        return train_loader, val_loader, test_loader

    def _extract_ground_truth(self, test_data: Any) -> np.ndarray:
        """Extract ground truth targets in same order as _prepare_training_data.
        
        Returns shape (N, forecast_length) to match predict() output shape.
        """
        pairs = self._get_context_target_pairs(test_data, require_target=True)
        if not pairs:
            return np.array([]).reshape(0, self.config.forecast_length)
        # Stack targets to match predict() output shape: (N, forecast_length)
        targets = np.stack([p[1] for p in pairs], axis=0)
        return targets

    # --- Public API ---
    def predict(
        self,
        data: Any,
        batch_size: Optional[int] = None,
        return_dict: bool = False,
    ) -> np.ndarray:
        """Make zero-shot predictions. Model must be initialized (no fit required).
        
        Note: Base class signature returns np.ndarray only. Use predict_with_metadata()
        for dict return with additional info.
        """
        if self.model is None:
            raise ValueError("Model not initialized; call constructor with valid config")

        bs = batch_size if batch_size is not None else self.config.batch_size
        loader, _, _ = self._prepare_training_data(data, batch_size=bs)
        all_preds = []
        for batch in loader:
            ctx = batch["context"]
            if isinstance(ctx, torch.Tensor):
                ctx = ctx.numpy()
            ctx_lens = batch.get("context_len")
            if isinstance(ctx_lens, torch.Tensor):
                ctx_lens = ctx_lens.numpy()
            preds = self._forecast_batch(
                ctx,
                self.config.forecast_length,
                context_lengths=ctx_lens,
            )
            all_preds.append(preds)
        predictions = np.concatenate(all_preds, axis=0)

        if return_dict:
            # For compatibility, but base signature doesn't support this
            # Users should use predict_with_metadata() if they need dict
            return predictions  # type: ignore[return-value]
        return predictions
    
    def predict_with_metadata(
        self,
        data: Any,
        batch_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Make predictions and return dict with metadata."""
        predictions = self.predict(data, batch_size=batch_size, return_dict=False)
        return {
            "predictions": predictions,
            "model_type": "moment",
            "config": self.config.to_dict(),
        }

    def predict_single_window(
        self,
        context: np.ndarray,
        forecast_length: int,
    ) -> np.ndarray:
        """Predict one step ahead for a single context window (for holdout/eval scripts).

        Compatible with model-agnostic evaluation scripts that call
        model.predict_single_window(context_values, forecast_length) per window.

        Args:
            context: 1D array of historical BG values (length >= 10).
            forecast_length: Number of steps to forecast.

        Returns:
            1D array of shape (forecast_length,) with predicted BG values in mmol/L.
        """
        context = np.asarray(context, dtype=np.float64).flatten()
        if len(context) < 10:
            raise ValueError("context must have at least 10 values")
        return self._forecast_single(context, prediction_length=forecast_length)

    def predict_zero_shot(
        self,
        data: Any,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """Zero-shot predictions without requiring fit(). Same as predict() for Moment."""
        return self.predict(data, batch_size=batch_size)

    # --- Checkpoints ---
    def _save_checkpoint(self, output_dir: str) -> None:
        if self.model is not None and hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(output_dir)  # type: ignore[call-overload]
            info_print(f"Moment checkpoint saved to {output_dir}")

    def _load_checkpoint(self, model_dir: str) -> None:
        MOMENTPipeline = _optional_moment_import()
        best_pt = os.path.join(model_dir, "best_model.pt")
        final_pt = os.path.join(model_dir, "final_model.pt")

        if os.path.isfile(best_pt) or os.path.isfile(final_pt):
            # Fine-tuned checkpoint: load base model then state dict
            if not getattr(self.config, "model_path", None):
                raise ValueError(
                    "config.model_path (base MOMENT id) required to load fine-tuned checkpoint; "
                    "ensure config.json includes model_path."
                )
            base_id = self.config.model_path
            info_print(f"Loading base MOMENT from {base_id}, then fine-tuned weights from {model_dir}")
            self.model = MOMENTPipeline.from_pretrained(
                base_id,
                model_kwargs={"enable_gradient_checkpointing": True},
            )
            ckpt_path = best_pt if os.path.isfile(best_pt) else final_pt
            ckpt = torch.load(
                ckpt_path, map_location=self._device, weights_only=False
            )  # False: checkpoint may contain optimizer state with numpy scalars
            state = ckpt.get("model_state_dict", ckpt)
            self.model.load_state_dict(state, strict=True)
        else:
            # HuggingFace-style directory
            self.model = MOMENTPipeline.from_pretrained(model_dir)

        self.model.init()
        self.model.to(self._device)
        self.model.eval()
        self.is_fitted = True
        info_print(f"Moment checkpoint loaded from {model_dir}")

    # --- Training ---
    def _train_model(
        self,
        train_data: Any,
        output_dir: str = "./output",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Fine-tune MOMENT model on training data.
        
        Uses masked reconstruction training: given context + target, masks the target
        portion and trains the model to reconstruct it.
        
        Args:
            train_data: Training data (dataset name, DataFrame, or dict of DataFrames)
            output_dir: Directory for saving checkpoints and logs
            **kwargs: Additional arguments (e.g., resume_from_checkpoint)
            
        Returns:
            Dictionary containing train_metrics and training_history
        """
        # Check training_mode for zero-shot (TTM pattern)
        if hasattr(self.config, "training_mode") and self.config.training_mode == "zero_shot":
            info_print("Moment zero-shot mode: skipping training")
            return {"train_metrics": {}, "training_history": {}}
        
        if self.model is None:
            raise ValueError("Model not initialized")
        
        info_print("Starting MOMENT fine-tuning...")
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare data loaders
        train_loader, val_loader, _ = self._prepare_training_data(train_data)
        
        if train_loader is None or len(train_loader.dataset) == 0:
            raise ValueError("No training data available")
        
        info_print(f"Training samples: {len(train_loader.dataset)}")
        if val_loader:
            info_print(f"Validation samples: {len(val_loader.dataset)}")
        
        # Set model to training mode
        self.model.train()
        
        # Setup optimizer
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import CosineAnnealingLR
        
        # Freeze backbone if configured
        freeze_backbone = False
        if hasattr(self.config, "training_config") and hasattr(self.config.training_config, "freeze_backbone"):
            freeze_backbone = self.config.training_config.freeze_backbone
        
        if freeze_backbone:
            info_print("Freezing backbone parameters")
            # Try to freeze the underlying model if it exists
            if hasattr(self.model, "model"):
                for param in self.model.model.parameters():
                    param.requires_grad = False
            # Only train head/adapter layers if they exist
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        else:
            trainable_params = list(self.model.parameters())
        
        optimizer = AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=getattr(self.config, "weight_decay", 0.01),
        )
        
        num_epochs = self.config.num_epochs
        total_steps = len(train_loader) * num_epochs
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
        
        # Loss function
        loss_fn = torch.nn.MSELoss()
        
        # Training loop
        training_history = []
        best_val_loss = float("inf")
        best_model_state = None
        
        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx, batch in enumerate(train_loader):
                contexts = batch["context"].to(self._device)  # [B, max_ctx_len]
                targets = batch["target"].to(self._device)  # [B, forecast_len]
                context_lens = batch["context_len"]  # [B]
                
                optimizer.zero_grad()
                
                # Create full sequence: context + target
                # For each sample, concatenate context and target
                batch_size = contexts.shape[0]
                forecast_len = targets.shape[1]
                max_ctx_len = contexts.shape[1]
                
                # Build input sequences and masks
                # Input: [B, 1, context_len + forecast_len]
                # Mask: 1 for context, 0 for target (to predict)
                losses = []
                
                for i in range(batch_size):
                    ctx_len = int(context_lens[i])
                    ctx = contexts[i, -ctx_len:]  # Actual context (remove padding)
                    tgt = targets[i]  # Target to predict
                    
                    # Normalize: fit scaler ONLY on context (matching inference)
                    # Then transform both context and target with same scaler
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    ctx_np = ctx.cpu().numpy()
                    tgt_np = tgt.cpu().numpy()
                    
                    # Fit scaler on context only (like inference)
                    ctx_scaled = scaler.fit_transform(ctx_np.reshape(-1, 1)).flatten()
                    # Transform target with same scaler
                    tgt_scaled_np = scaler.transform(tgt_np.reshape(-1, 1)).flatten()
                    
                    # Build full sequence: context + zeros for target (will be masked)
                    full_len = len(ctx_scaled) + forecast_len
                    if full_len > MOMENT_MAX_LEN:
                        keep = MOMENT_MAX_LEN - forecast_len
                        ctx_scaled = ctx_scaled[-keep:]
                        full_len = MOMENT_MAX_LEN
                        ctx_len = keep
                    
                    # Create input tensor [1, 1, seq_len] with zeros for target
                    input_seq = np.zeros(full_len, dtype=np.float32)
                    input_seq[:len(ctx_scaled)] = ctx_scaled
                    input_seq = torch.tensor(input_seq, dtype=torch.float32).to(self._device)
                    input_seq = input_seq.unsqueeze(0).unsqueeze(0)
                    
                    # Create mask: 1 = observed (context), 0 = to predict (target)
                    input_mask = torch.ones(full_len, dtype=torch.long, device=self._device)
                    input_mask[len(ctx_scaled):] = 0
                    input_mask = input_mask.unsqueeze(0)
                    
                    # Forward pass
                    with torch.set_grad_enabled(True):
                        output = self.model.forecast(x_enc=input_seq, input_mask=input_mask)
                        pred_scaled = output.forecast[0, 0, :forecast_len]  # [forecast_len]
                        
                        # Ground truth (scaled with same scaler as context)
                        tgt_scaled = torch.tensor(
                            tgt_scaled_np[:forecast_len],
                            dtype=torch.float32,
                            device=self._device
                        )
                        
                        # Loss on scaled values
                        loss = loss_fn(pred_scaled, tgt_scaled)
                        losses.append(loss)
                
                # Average loss across batch
                batch_loss = torch.stack(losses).mean()
                
                # Backward pass
                batch_loss.backward()
                
                # Gradient clipping
                grad_clip = getattr(self.config, "gradient_clip_val", 1.0)
                torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
                
                optimizer.step()
                scheduler.step()
                
                epoch_loss += batch_loss.item()
                num_batches += 1
                
                if (batch_idx + 1) % 10 == 0:
                    info_print(
                        f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_loader)}, "
                        f"Loss: {batch_loss.item():.6f}, LR: {scheduler.get_last_lr()[0]:.2e}"
                    )
            
            avg_train_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            
            # Validation
            val_loss = None
            if val_loader:
                self.model.eval()
                val_losses = []
                with torch.no_grad():
                    for val_batch in val_loader:
                        val_contexts = val_batch["context"].to(self._device)
                        val_targets = val_batch["target"].to(self._device)
                        val_context_lens = val_batch["context_len"]
                        
                        for i in range(val_contexts.shape[0]):
                            ctx_len = int(val_context_lens[i])
                            ctx = val_contexts[i, -ctx_len:]
                            tgt = val_targets[i]
                            
                            # Match training: fit scaler on context only, then transform target
                            from sklearn.preprocessing import StandardScaler
                            scaler = StandardScaler()
                            ctx_np = ctx.cpu().numpy()
                            tgt_np = tgt.cpu().numpy()
                            
                            ctx_scaled = scaler.fit_transform(ctx_np.reshape(-1, 1)).flatten()
                            tgt_scaled_np = scaler.transform(tgt_np.reshape(-1, 1)).flatten()
                            
                            full_len = len(ctx_scaled) + len(tgt)
                            if full_len > MOMENT_MAX_LEN:
                                keep = MOMENT_MAX_LEN - len(tgt)
                                ctx_scaled = ctx_scaled[-keep:]
                                full_len = MOMENT_MAX_LEN
                                ctx_len = keep
                            
                            input_seq = np.zeros(full_len, dtype=np.float32)
                            input_seq[:len(ctx_scaled)] = ctx_scaled
                            input_seq = torch.tensor(input_seq, dtype=torch.float32).to(self._device)
                            input_seq = input_seq.unsqueeze(0).unsqueeze(0)
                            
                            input_mask = torch.ones(full_len, dtype=torch.long, device=self._device)
                            input_mask[len(ctx_scaled):] = 0
                            input_mask = input_mask.unsqueeze(0)
                            
                            output = self.model.forecast(x_enc=input_seq, input_mask=input_mask)
                            pred_scaled = output.forecast[0, 0, :len(tgt)]
                            tgt_scaled = torch.tensor(
                                tgt_scaled_np[:len(tgt)],
                                dtype=torch.float32,
                                device=self._device
                            )
                            
                            val_losses.append(loss_fn(pred_scaled, tgt_scaled).item())
                
                val_loss = np.mean(val_losses) if val_losses else None
            
            # Logging
            log_entry = {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
            }
            if val_loss is not None:
                log_entry["val_loss"] = val_loss
            training_history.append(log_entry)
            
            info_print(
                f"Epoch {epoch+1}/{num_epochs} completed - "
                f"Train Loss: {avg_train_loss:.6f}"
                + (f", Val Loss: {val_loss:.6f}" if val_loss is not None else "")
            )
            
            # Save best model
            current_loss = val_loss if val_loss is not None else avg_train_loss
            if current_loss < best_val_loss:
                best_val_loss = current_loss
                best_model_state = {
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "loss": current_loss,
                }
                info_print(f"New best model (loss: {best_val_loss:.6f})")
        
        # Save final model
        if best_model_state:
            checkpoint_path = os.path.join(output_dir, "best_model.pt")
            torch.save(best_model_state, checkpoint_path)
            info_print(f"Saved best model to {checkpoint_path}")
        
        # Save final checkpoint
        final_checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": num_epochs,
            "config": self.config.to_dict(),
        }
        final_path = os.path.join(output_dir, "final_model.pt")
        torch.save(final_checkpoint, final_path)
        info_print(f"Saved final model to {final_path}")
        
        # Save training history
        import json
        history_path = os.path.join(output_dir, "training_history.json")
        with open(history_path, "w") as f:
            json.dump(training_history, f, indent=2)
        
        self.is_fitted = True
        self.model.eval()
        
        return {
            "train_metrics": {
                "final_train_loss": avg_train_loss,
                "best_val_loss": best_val_loss if val_loss is not None else None,
            },
            "training_history": training_history,
        }

    def _get_training_args(self) -> TrainingArguments:
        output_dir = getattr(self.config, "output_dir", None) or "./moment_output"
        return TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            num_train_epochs=self.config.num_epochs,
            learning_rate=self.config.learning_rate,
            save_strategy="epoch",
            eval_strategy="epoch",  # Fixed: evaluation_strategy -> eval_strategy
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            logging_dir=os.path.join(output_dir, "logs"),
            logging_steps=10,
            save_total_limit=3,
            fp16=torch.cuda.is_available() and not self.config.use_cpu,
        )


def create_moment_model(
    model_path: str = "AutonLab/MOMENT-1-large",
    context_length: int = 512,
    forecast_length: int = 96,
    use_lora: bool = False,
    **kwargs: Any,
) -> MomentForecaster:
    """Factory to create a Moment model with sensible defaults."""
    config = MomentConfig(
        model_path=model_path,
        context_length=context_length,
        forecast_length=forecast_length,
        use_lora=use_lora,
        **kwargs,
    )
    return MomentForecaster(config)
