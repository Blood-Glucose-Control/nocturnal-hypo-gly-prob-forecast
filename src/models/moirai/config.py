"""
Moirai configuration classes.
 
This module provides configuration classes specific to Moirai models,
extending the base model configuration with Moirai-specific parameters.
"""
 
from dataclasses import dataclass, field
from typing import Dict, List, Optional
 
from src.models.base import ModelConfig, TrainingBackend
 
 
@dataclass
class MoiraiTrainingConfig:
    """Moirai-specific training configuration.
 
    Contains training parameters specific to Moirai models that are not part
    of the base ModelConfig.
 
    Attributes:
        freeze_backbone: Whether to freeze the pre-trained Moirai backbone weights.
        use_tracking_callback: Whether to use experiment tracking callbacks.
        find_optimal_lr: Whether to run learning rate finder before training.
        loss_function: Loss function for training. Options: "mse", "mae",
            "huber", or "custom".
        scaler_type: Data normalization method. Options: "standard", "minmax",
            or "robust".
    """
 
    freeze_backbone: bool = False
    use_tracking_callback: bool = True
    find_optimal_lr: bool = False
    loss_function: str = "mse"  # "mse", "mae", "huber", "custom"
    scaler_type: str = "standard"  # "standard", "minmax", "robust"
 
 
@dataclass
class MoiraiDataConfig:
    """Moirai-specific data configuration.
 
    Contains data preprocessing and feature configuration specific to Moirai
    models, particularly for blood glucose forecasting applications.
 
    Attributes:
        input_features: List of input feature column names (e.g., ["cob", "iob"]).
        target_features: List of target column names to predict (e.g., ["bg_mM"]).
        split_config: Dictionary defining train/val/test split ratios.
            Must sum to 1.0. Example: {"train": 0.7, "val": 0.2, "test": 0.1}.
        num_input_channels: Number of input feature channels.
        num_output_channels: Number of output prediction channels.
        prediction_filter_length: Optional filter length for predictions.
            If None, uses full forecast_length.
    """
 
    input_features: List[str] | None = None
    target_features: List[str] | None = None
    split_config: Dict[str, float] | None = None
    num_input_channels: int = 5
    num_output_channels: int = 1
    prediction_filter_length: Optional[int] = None
 
    def __post_init__(self):
        if self.input_features is None:
            self.input_features = [
                "cob",
                "carb_availability",
                "insulin_availability",
                "iob",
                "steps",
            ]
        if self.target_features is None:
            self.target_features = ["bg_mM"]
        if self.split_config is None:
            self.split_config = {"train": 0.7, "val": 0.2, "test": 0.1}
 
 
class MoiraiConfig(ModelConfig):
    """Extended configuration class for Moirai-specific parameters.
 
    Inherits from ModelConfig and adds Moirai-specific attributes for
    inference, data preprocessing, and model architecture configuration.
 
    Attributes:
        model_type: Always "moirai" for this configuration.
        model_path: HuggingFace model identifier or local path.
        context_length: Input sequence length (lookback window).
        forecast_length: Output prediction horizon.
 
        --- Inference ---
        patch_size: Patch size for Moirai's patching mechanism.
            Use ``"auto"`` (default) to let Moirai choose.
        num_samples: Number of Monte Carlo samples for probabilistic
            forecasting (default: 100).
        past_covariate_dim: Number of past-only covariates passed as
            ``past_feat_dynamic_real`` in the GluonTS dataset (e.g. 2
            for IOB + COB). 0 means BG-only inference.
        checkpoint_path: Optional path to a ``.ckpt`` fine-tuned checkpoint
            produced by ``uni2ts`` CLI training. When set, the fine-tuned
            weights are loaded instead of the pretrained HuggingFace weights.
 
        --- Data helpers ---
        interval_mins: CGM sampling interval in minutes (5 for BrisT1D).
        target_col: Name of the target blood-glucose column.
        covariate_cols: Ordered list of covariate column names matching
            ``past_covariate_dim`` (e.g. ``["iob", "cob"]``).
 
        --- Architecture (informational) ---
        d_model: Model embedding dimension.
        n_heads: Number of attention heads.
        n_layers: Number of transformer layers.
        dropout: Dropout rate.
 
        --- Sub-configs ---
        training_config: Moirai-specific training parameters.
        data_config: Moirai-specific data configuration.
    """
 
    def __init__(self, **kwargs):
        """Initialize Moirai configuration.
 
        Args:
            **kwargs: Any ModelConfig or Moirai-specific parameters. Unknown
                keys are silently accepted and stored as attributes so that
                ``create_moirai_model()`` can forward arbitrary kwargs cleanly.
        """
        # Keys that belong only to MoiraiConfig, not ModelConfig
        moirai_only_keys = {
            "patch_size",
            "num_samples",
            "past_covariate_dim",
            "checkpoint_path",
            "interval_mins",
            "target_col",
            "covariate_cols",
            "d_model",
            "n_heads",
            "n_layers",
            "dropout",
            "training_config",
            "data_config",
        }
 
        base_kwargs = {k: v for k, v in kwargs.items() if k not in moirai_only_keys}
        super().__init__(**base_kwargs)
 
        self.model_type = "moirai"
 
        # Inference parameters
        self.patch_size = kwargs.get("patch_size", "auto")
        self.num_samples: int = kwargs.get("num_samples", 100)
        self.past_covariate_dim: int = kwargs.get("past_covariate_dim", 0)
        self.checkpoint_path: Optional[str] = kwargs.get("checkpoint_path", None)
 
        # Data helpers
        self.interval_mins: int = kwargs.get("interval_mins", 5)
        self.target_col: str = kwargs.get("target_col", "bg_mM")
        self.covariate_cols: List[str] = kwargs.get("covariate_cols", [])
 
        # Architecture (informational — actual values are in the HF checkpoint)
        self.d_model: int = kwargs.get("d_model", 768)
        self.n_heads: int = kwargs.get("n_heads", 12)
        self.n_layers: int = kwargs.get("n_layers", 12)
        self.dropout: float = kwargs.get("dropout", 0.1)
 
        # Sub-configs
        self.training_config: MoiraiTrainingConfig = kwargs.get(
            "training_config", MoiraiTrainingConfig()
        )
        self.data_config: MoiraiDataConfig = kwargs.get(
            "data_config", MoiraiDataConfig()
        )
 
    def supports_lora(self) -> bool:
        """Moirai is transformer-based and supports LoRA."""
        return True
 
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        base_dict = super().to_dict()
        base_dict.update(
            {
                "patch_size": self.patch_size,
                "num_samples": self.num_samples,
                "past_covariate_dim": self.past_covariate_dim,
                "checkpoint_path": self.checkpoint_path,
                "interval_mins": self.interval_mins,
                "target_col": self.target_col,
                "covariate_cols": self.covariate_cols,
                "d_model": self.d_model,
                "n_heads": self.n_heads,
                "n_layers": self.n_layers,
                "dropout": self.dropout,
            }
        )
        return base_dict