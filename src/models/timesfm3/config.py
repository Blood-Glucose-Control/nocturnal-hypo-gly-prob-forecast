# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)
# For commercial licensing, contact: cjrisi/christopher AT uwaterloo/gluroo DOT ca/com

"""TimesFM 3.0 configuration classes.

Backend/runtime role:
    Configuration for the standalone TimesFM 3.0 adapter
    (:class:`~src.models.timesfm3.model.TimesFM3Forecaster`). This module is a
    SEPARATE model family from the published ``src/models/timesfm`` adapter
    (which wraps the HuggingFace-Transformers TimesFM used by published papers).
    TimesFM 3.0 runs on the native ``timesfm3`` package and supports native
    covariates plus LoRA fine-tuning via ``peft``.

Key behavior differences:
    - Zero-shot inference with native past-only covariates (e.g. IOB).
    - LoRA fine-tuning with validation-based early stopping.
    - Fixed 9-level (decile) quantile output head from the pretrained model.

Invariants/limitations:
    - Target is expected in mmol/L under the ``bg_mM`` column (matching the
      repository convention), NOT mg/dL.
    - Covariates are consumed as *past-only* features spanning the context
      window; static covariates are not supported by TimesFM 3.0.
"""

from typing import List, Optional

from ..base import ModelConfig, TrainingBackend

# Fixed decile grid produced by the pretrained TimesFM 3.0 quantile head.
_DEFAULT_QUANTILES: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


class TimesFM3Config(ModelConfig):
    """Configuration for the standalone TimesFM 3.0 forecaster.

    Extends :class:`~src.models.base.base_model.ModelConfig` with TimesFM 3.0
    specific fields covering covariate selection, quantile output, LoRA adapter
    hyperparameters, and fine-tuning/early-stopping controls. Unknown keyword
    arguments that match base :class:`ModelConfig` fields are forwarded to the
    parent; TimesFM 3.0 specific keys are consumed here.

    Contract obligations:
        - ``model_type`` is always ``"timesfm3"``.
        - Field defaults mirror the proven zero-shot + LoRA fine-tune recipe.
    """

    def __init__(self, **kwargs):
        """Initialize a TimesFM 3.0 configuration.

        Args:
            **kwargs: Base :class:`ModelConfig` fields plus TimesFM 3.0 specific
                keys documented as instance attributes below. Numeric strings are
                accepted for float fields for YAML compatibility.
        """
        timesfm3_specific_params = {
            "device",
            "target_col",
            "covariate_cols",
            "make_positive",
            "quantiles",
            "lora_r",
            "lora_alpha",
            "lora_dropout",
            "lora_target_modules",
            "val_fraction",
            "window_stride",
            "max_train_windows",
            "grad_clip_norm",
            "covariate_jitter",
        }

        base_kwargs = {
            k: v for k, v in kwargs.items() if k not in timesfm3_specific_params
        }
        super().__init__(**base_kwargs)

        # Identity + backend.
        self.model_type = "timesfm3"
        if self.model_path is None:
            self.model_path = "google/timesfm-3.0-pytorch"
        self.training_backend = TrainingBackend.PYTORCH

        # Inference wiring.
        # Preferred device string (e.g. "cuda", "cpu"). None -> auto-select.
        self.device: Optional[str] = kwargs.get("device", None)
        # Target column read from the context DataFrame (mmol/L, NOT mg/dL).
        self.target_col: str = kwargs.get("target_col", "bg_mM")
        # Past-only covariate columns supplied to TimesFM 3.0 (context window
        # only). Missing columns are ignored gracefully at runtime.
        self.covariate_cols: List[str] = list(kwargs.get("covariate_cols", ["iob"]))
        # Clamp forecasts to be non-negative (blood glucose is strictly positive).
        self.make_positive: bool = kwargs.get("make_positive", True)
        # Quantile levels requested from the fixed decile output head.
        self.quantiles: List[float] = list(kwargs.get("quantiles", _DEFAULT_QUANTILES))

        # LoRA adapter hyperparameters (peft.LoraConfig).
        self.lora_r: int = kwargs.get("lora_r", 8)
        self.lora_alpha: int = kwargs.get("lora_alpha", 16)
        self.lora_dropout: float = kwargs.get("lora_dropout", 0.0)
        self.lora_target_modules = kwargs.get("lora_target_modules", "all-linear")

        # Fine-tuning / early-stopping controls.
        # Fraction of training windows held out for validation-based early
        # stopping. 0.0 disables the val split (best-by-train-loss fallback).
        self.val_fraction: float = kwargs.get("val_fraction", 0.2)
        # Sliding-window stride when building training windows. None -> horizon.
        self.window_stride: Optional[int] = kwargs.get("window_stride", None)
        # Cap on the number of training windows built from the input data.
        self.max_train_windows: int = kwargs.get("max_train_windows", 2048)
        # Max-norm gradient clip applied to trainable LoRA parameters.
        self.grad_clip_norm: float = kwargs.get("grad_clip_norm", 1.0)
        # Tiny jitter added to covariate windows during training. All-zero
        # covariate channels (e.g. no insulin) otherwise yield NaN gradients
        # through the model's per-variate normalization.
        self.covariate_jitter: float = kwargs.get("covariate_jitter", 1e-3)

        # Recipe-aligned defaults for shared base fields when not overridden.
        if "learning_rate" not in base_kwargs:
            self.learning_rate = 1e-4
        if "weight_decay" not in base_kwargs:
            self.weight_decay = 0.01
        if "batch_size" not in base_kwargs:
            self.batch_size = 32
        if "early_stopping_patience" not in base_kwargs:
            self.early_stopping_patience = 3
