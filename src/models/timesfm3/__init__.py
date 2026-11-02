"""TimesFM 3.0 model family (standalone from the published ``timesfm`` adapter).

This package provides TimesFM 3.0 zero-shot inference with native past-only
covariates plus LoRA fine-tuning. It is intentionally separate from
``src/models/timesfm`` (the HuggingFace-Transformers TimesFM used by published
papers) and shares no code or checkpoints with it.
"""

from .config import TimesFM3Config
from .model import TimesFM3Forecaster

__all__ = ["TimesFM3Config", "TimesFM3Forecaster"]
