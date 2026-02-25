"""TimesFM model using HuggingFace Transformers (TimesFmModelForPrediction).

Supports zero-shot inference and fine-tuning with per-window normalized loss.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.models.base import BaseTimeSeriesFoundationModel, TrainingBackend
from src.models.timesfm.config import TimesFMConfig
from src.utils.logging_helper import info_print, error_print

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _longest_nan_run(mask: np.ndarray) -> int:
    """Return the length of the longest contiguous True run in a boolean array."""
    if not mask.any():
        return 0
    max_run = 0
    current = 0
    for val in mask:
        if val:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 0
    return max_run


class TimesFMDataset(Dataset):
    """Per-patient sliding-window dataset for TimesFM fine-tuning.

    Creates windows within each patient's series independently, ensuring no
    window spans patient boundaries or data gaps. Windows with NaN values
    are skipped (matching evaluate() behavior).

    Args:
        patient_series: List of 1D numpy arrays, one per patient.
        context_length: Number of past timesteps for context.
        horizon_length: Number of future timesteps to predict.
        freq_type: Frequency type (0=high/5-min, 1=medium/hourly, 2=low/weekly+).
        stride: Step size between windows (default=horizon_length, non-overlapping).
    """

    def __init__(
        self,
        patient_series: List[np.ndarray],
        context_length: int,
        horizon_length: int,
        freq_type: int = 0,
        stride: Optional[int] = None,
    ):
        if freq_type not in [0, 1, 2]:
            raise ValueError("freq_type must be 0, 1, or 2")

        self.context_length = context_length
        self.horizon_length = horizon_length
        self.freq_type = freq_type
        self.stride = stride if stride is not None else horizon_length
        self._build_windows(patient_series)

    def _build_windows(self, patient_series: List[np.ndarray]) -> None:
        """Create windows per-patient, skipping windows with NaN values."""
        self.samples: List[Tuple[np.ndarray, np.ndarray]] = []
        total_length = self.context_length + self.horizon_length
        skipped_nan = 0

        for series in patient_series:
            series = series.astype(np.float32)
            if len(series) < total_length:
                continue

            for start in range(0, len(series) - total_length + 1, self.stride):
                context = series[start : start + self.context_length]
                future = series[start + self.context_length : start + total_length]

                if np.isnan(context).any() or np.isnan(future).any():
                    skipped_nan += 1
                    continue

                self.samples.append((context, future))

        if skipped_nan > 0:
            info_print(f"Skipped {skipped_nan} windows with NaN values")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """Return dict with past_values, past_values_padding, freq, future_values."""
        x_context, x_future = self.samples[index]

        return {
            "past_values": torch.tensor(x_context, dtype=torch.float32),
            "past_values_padding": torch.zeros(len(x_context), dtype=torch.long),
            "freq": torch.tensor(self.freq_type, dtype=torch.long),
            "future_values": torch.tensor(x_future, dtype=torch.float32),
        }


class TimesFMForTrainer(nn.Module):
    """HF Trainer wrapper that computes loss in per-window normalized space.

    Truncates predictions to match target horizon (HF TimesFM outputs 128 steps)
    and normalizes both predictions and targets by context mean/std before MSE,
    preventing high-variance patients from dominating gradients.
    """

    def __init__(self, hf_prediction_model):
        super().__init__()
        self.prediction_model = hf_prediction_model

    def forward(
        self,
        past_values,
        past_values_padding,
        freq,
        future_values=None,
        **kwargs,
    ):
        outputs = self.prediction_model(
            past_values=past_values,
            freq=freq,
        )
        mean_predictions = outputs.mean_predictions  # (B, model_horizon)

        loss = None
        if future_values is not None:
            horizon = future_values.shape[1]
            preds = mean_predictions[:, :horizon].float()
            targets = future_values.float()

            # Per-window normalization (clamp scale >= 0.1 for low-variance windows).
            locs = torch.stack([pv.float().mean() for pv in past_values])
            scales = torch.stack(
                [pv.float().std().clamp(min=0.1) for pv in past_values]
            )
            locs = locs.to(preds.device).unsqueeze(-1)  # (B, 1)
            scales = scales.to(preds.device).unsqueeze(-1)  # (B, 1)

            pred_norm = (preds - locs) / scales
            target_norm = (targets - locs) / scales
            loss = F.mse_loss(pred_norm, target_norm)

        return {"loss": loss, "logits": mean_predictions}


class TimesFMForecaster(BaseTimeSeriesFoundationModel):
    """TimesFM forecaster using HuggingFace Transformers.

    Uses TimesFmModelForPrediction for inference and HF Trainer for fine-tuning.
    """

    def __init__(
        self, config: TimesFMConfig, lora_config=None, distributed_config=None
    ):
        super().__init__(config, lora_config, distributed_config)
        self.config: TimesFMConfig = self.config
        self.hf_model = self.model  # alias set in _initialize_model

    @property
    def training_backend(self) -> TrainingBackend:
        return TrainingBackend.TRANSFORMERS

    @property
    def supports_lora(self) -> bool:
        return False

    def predict(
        self, data: pd.DataFrame, prediction_length: Optional[int] = None
    ) -> np.ndarray:
        """Make predictions given context data.

        Handles NaN values consistently with training: short gaps (<=45 min
        at 5-min resolution = 9 readings) are linearly interpolated, longer
        gaps raise an error.

        Args:
            data: DataFrame with 'bg_mM' column containing context window
            prediction_length: Number of steps to forecast

        Returns:
            Forecast as 1D numpy array
        """
        if self.hf_model is None:
            raise ValueError("Model not initialized.")

        prediction_length = prediction_length or self.config.horizon_length

        bg_col = "bg_mM"
        if bg_col not in data.columns:
            raise ValueError(f"DataFrame must contain '{bg_col}' column")

        context = data[bg_col].values.astype(np.float32)
        context = context[-self.config.context_length :]

        # Handle NaN: interpolate short gaps, reject long gaps
        nan_mask = np.isnan(context)
        if nan_mask.any():
            max_gap = _longest_nan_run(nan_mask)
            # 9 readings = 45 min at 5-min resolution
            max_allowed = self.config.imputation_threshold_mins // 5
            if max_gap > max_allowed:
                raise ValueError(
                    f"Context contains a {max_gap * 5}-minute gap "
                    f"(>{self.config.imputation_threshold_mins} min threshold). "
                    f"Pre-process data to remove large gaps before predict()."
                )
            valid = ~nan_mask
            if not valid.any():
                raise ValueError("Context is entirely NaN.")
            indices = np.arange(len(context))
            context = np.interp(indices, indices[valid], context[valid]).astype(
                np.float32
            )
            logger.info(
                "Interpolated %d NaN values in predict() context", nan_mask.sum()
            )

        model_dtype = next(self.hf_model.parameters()).dtype
        context_tensor = torch.tensor(context, dtype=model_dtype).to(self.device)
        freq_tensor = torch.tensor([0], dtype=torch.long).to(self.device)

        self.hf_model.eval()
        with torch.no_grad():
            outputs = self.hf_model(
                past_values=[context_tensor],
                freq=freq_tensor,
                return_dict=True,
            )

        forecast = outputs.mean_predictions[0].float().cpu().numpy()

        if len(forecast) > prediction_length:
            forecast = forecast[:prediction_length]

        return forecast

    def predict_zero_shot(
        self, data: pd.DataFrame, prediction_length: Optional[int] = None
    ) -> np.ndarray:
        """Zero-shot prediction (identical to predict for TimesFM)."""
        return self.predict(data, prediction_length)

    def _extract_ground_truth(self, test_data: Any) -> np.ndarray:
        """Extract ground truth bg_mM values from the end of the test data."""
        if isinstance(test_data, pd.DataFrame) and "bg_mM" in test_data.columns:
            values = test_data["bg_mM"].dropna().values.astype(np.float32)
            return values[-self.config.horizon_length :]
        raise ValueError("test_data must be a DataFrame with 'bg_mM' column")

    def evaluate(
        self,
        test_data: Any,
        batch_size: Optional[int] = None,
        return_predictions: bool = False,
    ) -> Dict[str, Any]:
        """Evaluate TimesFM on test data using rolling-window evaluation.

        Creates non-overlapping (context, target) windows per patient,
        batch-predicts all windows, and computes aggregate metrics.
        """
        if self.hf_model is None:
            raise ValueError("Model not initialized.")
        if not isinstance(test_data, pd.DataFrame) or "bg_mM" not in test_data.columns:
            raise ValueError("test_data must be a DataFrame with 'bg_mM' column")

        cl = self.config.context_length
        hl = self.config.horizon_length
        total_len = cl + hl
        model_dtype = next(self.hf_model.parameters()).dtype

        patient_col = next((c for c in ["p_num", "id"] if c in test_data.columns), None)

        context_windows: List[torch.Tensor] = []
        target_windows: List[np.ndarray] = []

        patients = test_data[patient_col].dropna().unique() if patient_col else [None]

        for pid in patients:
            patient_data = (
                test_data[test_data[patient_col] == pid]
                if pid is not None
                else test_data
            )
            values = patient_data["bg_mM"].values.astype(np.float32)

            for start in range(0, len(values) - total_len + 1, hl):
                context = values[start : start + cl]
                target = values[start + cl : start + total_len]

                if np.isnan(context).any() or np.isnan(target).any():
                    continue

                context_windows.append(torch.tensor(context, dtype=model_dtype))
                target_windows.append(target)

        num_patients = len(patients)
        num_windows = len(context_windows)

        if num_windows == 0:
            raise ValueError(
                f"No valid evaluation windows found. Need at least "
                f"{total_len} contiguous non-NaN bg_mM values per patient."
            )

        eval_batch_size = batch_size or self.config.batch_size

        info_print(
            f"Evaluating {num_windows} windows across {num_patients} patient(s) "
            f"(batch_size={eval_batch_size})"
        )

        # Batched inference to avoid GPU OOM on large holdout sets
        self.hf_model.eval()
        all_preds = []
        for i in range(0, num_windows, eval_batch_size):
            batch_ctx = [
                t.to(self.device) for t in context_windows[i : i + eval_batch_size]
            ]
            batch_freq = torch.tensor([0] * len(batch_ctx), dtype=torch.long).to(
                self.device
            )

            with torch.no_grad():
                outputs = self.hf_model(
                    past_values=batch_ctx,
                    freq=batch_freq,
                    return_dict=True,
                )

            for j in range(len(batch_ctx)):
                pred = outputs.mean_predictions[j].float().cpu().numpy()[:hl]
                all_preds.append(pred)

        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(target_windows)

        mse = float(np.mean((y_pred - y_true) ** 2))
        metrics = {
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
            "mae": float(np.mean(np.abs(y_pred - y_true))),
        }
        metrics["num_windows"] = num_windows
        metrics["num_patients"] = num_patients

        if return_predictions:
            metrics["predictions"] = all_preds
            metrics["ground_truth"] = target_windows

        return metrics

    def _initialize_model(self) -> None:
        """Load TimesFM from HuggingFace."""
        info_print("Initializing TimesFM model (HuggingFace)...")

        try:
            from transformers import TimesFmModelForPrediction
        except ImportError:
            error_print(
                "transformers package missing or too old. "
                "Install with: pip install transformers>=5.2.0"
            )
            raise

        checkpoint = self.config.checkpoint_path or "google/timesfm-2.0-500m-pytorch"
        info_print(f"Loading TimesFM from: {checkpoint}")

        self.device = (
            "cuda" if torch.cuda.is_available() and not self.config.use_cpu else "cpu"
        )
        info_print(f"Selected device: {self.device}")

        torch_dtype = getattr(torch, self.config.torch_dtype, torch.float32)

        self.hf_model = TimesFmModelForPrediction.from_pretrained(
            checkpoint,
            torch_dtype=torch_dtype,
            attn_implementation="sdpa",
        )

        self.hf_model.to(self.device)
        self.model = self.hf_model  # base class expects self.model

        info_print(
            f"TimesFM initialized on {self.device} (dtype={self.config.torch_dtype})"
        )

    def _prepare_training_data(
        self, train_data: Any
    ) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
        """Prepare DataLoaders with gap handling and per-patient windowing.

        Pipeline: extract per-patient DataFrames → gap handling (interpolate
        small gaps, segment at large gaps) → patient-level train/val split →
        sliding windows within each segment.
        """
        from collections import defaultdict
        from src.data.preprocessing.gap_handling import segment_all_patients

        info_print("Preparing data for TimesFM finetuning...")

        # Step 1: Extract per-patient DataFrames (need DatetimeIndex for gap handling)
        if isinstance(train_data, dict):
            patient_dfs = {}
            for pid, df in train_data.items():
                if "bg_mM" in df.columns:
                    patient_dfs[str(pid)] = df
        elif isinstance(train_data, pd.DataFrame):
            if "bg_mM" not in train_data.columns:
                raise ValueError("DataFrame must contain 'bg_mM' column")
            patient_col = next(
                (c for c in ["p_num", "id"] if c in train_data.columns), None
            )
            if patient_col:
                patient_dfs = {
                    str(pid): group for pid, group in train_data.groupby(patient_col)
                }
            else:
                patient_dfs = {"single": train_data}
        else:
            raise ValueError(
                f"train_data must be DataFrame or dict, got {type(train_data)}"
            )

        total_samples = sum(len(df) for df in patient_dfs.values())
        info_print(
            f"Total samples: {total_samples:,} across {len(patient_dfs)} patients"
        )

        # Step 2: Ensure DataFrames have DatetimeIndex for gap handling
        for pid, df in patient_dfs.items():
            if not isinstance(df.index, pd.DatetimeIndex):
                if "datetime" in df.columns:
                    patient_dfs[pid] = df.set_index("datetime")
                else:
                    info_print(
                        f"Patient {pid}: no datetime index or column, "
                        f"skipping gap handling"
                    )

        # Step 3: Gap handling — interpolate small gaps, segment at large gaps
        min_seg_length = self.config.context_length + self.config.horizon_length
        segments = segment_all_patients(
            patient_dfs,
            imputation_threshold_mins=self.config.imputation_threshold_mins,
            min_segment_length=min_seg_length,
            bg_col="bg_mM",
        )
        info_print(
            f"Gap handling: {len(segments)} segments from "
            f"{len(patient_dfs)} patients "
            f"(interpolated gaps <= {self.config.imputation_threshold_mins} min, "
            f"min segment length = {min_seg_length})"
        )

        # Step 4: Group segments back by original patient for train/val split
        patient_to_segments: Dict[str, List[np.ndarray]] = defaultdict(list)
        for seg_id, seg_df in segments.items():
            original_pid = seg_id.rsplit("_seg_", 1)[0]
            patient_to_segments[original_pid].append(seg_df["bg_mM"].values)

        # Step 5: Patient-level train/val split
        pids = sorted(patient_to_segments.keys(), key=str)
        np.random.seed(42)
        shuffled = list(pids)
        np.random.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * self.config.val_patient_ratio))
        val_pids = set(shuffled[:n_val])
        train_pids = set(shuffled[n_val:])

        # Flatten segment arrays per split
        train_arrays = [arr for pid in train_pids for arr in patient_to_segments[pid]]
        val_arrays = [arr for pid in val_pids for arr in patient_to_segments[pid]]

        stride = self.config.window_stride or self.config.horizon_length

        train_dataset = TimesFMDataset(
            patient_series=train_arrays,
            context_length=self.config.context_length,
            horizon_length=self.config.horizon_length,
            freq_type=self.config.freq_type,
            stride=stride,
        )
        val_dataset = TimesFMDataset(
            patient_series=val_arrays,
            context_length=self.config.context_length,
            horizon_length=self.config.horizon_length,
            freq_type=self.config.freq_type,
            stride=stride,
        )

        info_print(
            f"Patients: {len(train_pids)} train, {len(val_pids)} val | "
            f"Windows: {len(train_dataset):,} train, {len(val_dataset):,} val"
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
        )

        return train_loader, val_loader, None

    @staticmethod
    def _collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, Any]:
        """Custom collator for HF Trainer.

        HF TimesFM expects past_values as a list of 1D tensors (variable-length
        sequences). It handles internal padding via _preprocess.
        """
        return {
            "past_values": [item["past_values"] for item in batch],
            "past_values_padding": torch.stack(
                [item["past_values_padding"] for item in batch]
            ),
            "freq": torch.stack([item["freq"] for item in batch]),
            "future_values": torch.stack([item["future_values"] for item in batch]),
        }

    def _train_model(
        self, train_data: Any, output_dir: str, **kwargs
    ) -> Dict[str, Any]:
        """Fine-tune TimesFM using HF Trainer with per-window normalized loss."""
        from transformers import Trainer, TrainingArguments

        info_print("Starting TimesFM finetuning with HF Trainer...")

        train_loader, val_loader, _ = self._prepare_training_data(train_data)

        # Wrap model for normalized-space loss
        trainer_model = TimesFMForTrainer(self.hf_model)

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            bf16=(self.config.torch_dtype == "bfloat16" and self.device == "cuda"),
            fp16=(self.config.torch_dtype == "float16" and self.device == "cuda"),
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            eval_strategy="no",
            save_strategy="epoch",
            save_total_limit=3,
            remove_unused_columns=False,
            label_names=["future_values"],
            logging_steps=100,
            dataloader_num_workers=2,
            report_to="none",
        )

        trainer = Trainer(
            model=trainer_model,
            args=training_args,
            train_dataset=train_loader.dataset,
            eval_dataset=val_loader.dataset if val_loader else None,
            data_collator=self._collate_fn,
        )

        info_print(
            f"Training for {self.config.num_epochs} epochs, "
            f"batch_size={self.config.batch_size}, "
            f"grad_accum={self.config.gradient_accumulation_steps}, "
            f"LR={self.config.learning_rate}"
        )

        trainer.train()

        # Extract underlying model back from wrapper
        self.hf_model = trainer_model.prediction_model
        self.model = self.hf_model
        self.is_fitted = True

        # Save checkpoint
        os.makedirs(output_dir, exist_ok=True)
        self._save_checkpoint(output_dir)

        info_print(f"Finetuning complete. Model saved to {output_dir}")

        return {"training_history": trainer.state.log_history}

    def _save_checkpoint(self, output_dir: str) -> None:
        """Save checkpoint. Uses 'hf_model/' subdir to avoid config.json conflicts."""
        os.makedirs(output_dir, exist_ok=True)

        # Save HF model weights + config
        hf_model_dir = os.path.join(output_dir, "hf_model")
        if self.hf_model is not None:
            self.hf_model.save_pretrained(hf_model_dir)
            info_print(f"HF model saved to {hf_model_dir}")

        # Save our custom config
        timesfm_config_path = os.path.join(output_dir, "timesfm_config.json")
        config_dict = {
            "checkpoint_path": self.config.checkpoint_path,
            "context_length": self.config.context_length,
            "horizon_length": self.config.horizon_length,
            "use_cpu": self.config.use_cpu,
            "is_finetuned": self.is_fitted,
        }
        with open(timesfm_config_path, "w") as f:
            json.dump(config_dict, f, indent=2)

        info_print(f"TimesFM config saved to {timesfm_config_path}")

    def _load_checkpoint(self, model_dir: str) -> None:
        """Load model checkpoint from HF save_pretrained format."""
        from transformers import TimesFmModelForPrediction

        # Load our custom config
        timesfm_config_path = os.path.join(model_dir, "timesfm_config.json")
        if os.path.exists(timesfm_config_path):
            with open(timesfm_config_path, "r") as f:
                saved_config = json.load(f)

            if saved_config.get("checkpoint_path"):
                self.config.checkpoint_path = saved_config["checkpoint_path"]

            info_print(f"TimesFM config loaded from {timesfm_config_path}")

        # Load HF model weights
        hf_model_dir = os.path.join(model_dir, "hf_model")
        if os.path.exists(hf_model_dir):
            torch_dtype = getattr(torch, self.config.torch_dtype, torch.float32)
            self.hf_model = TimesFmModelForPrediction.from_pretrained(
                hf_model_dir,
                torch_dtype=torch_dtype,
            )
            self.hf_model.to(self.device)
            self.model = self.hf_model
            self.is_fitted = True
            info_print(f"HF model loaded from {hf_model_dir}")
        else:
            info_print(
                f"No HF model directory found at {hf_model_dir}, "
                f"using pretrained weights"
            )


def create_timesfm_model(
    checkpoint_path: Optional[str] = None,
    context_length: int = 512,
    horizon_length: int = 128,
    **kwargs,
) -> TimesFMForecaster:
    """Factory function to create a TimesFM model with sensible defaults.

    Args:
        checkpoint_path: HF model ID or local path.
        context_length: Input sequence length.
        horizon_length: Output prediction horizon.
        **kwargs: Additional configuration parameters.

    Returns:
        Initialized TimesFMForecaster instance.
    """
    config = TimesFMConfig(
        checkpoint_path=checkpoint_path,
        context_length=context_length,
        horizon_length=horizon_length,
        **kwargs,
    )

    return TimesFMForecaster(config)
