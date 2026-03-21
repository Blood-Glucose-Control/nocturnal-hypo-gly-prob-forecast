"""
Toto configuration classes.
"""

from src.models.base import ModelConfig, TrainingBackend


class TotoConfig(ModelConfig):
    """Configuration class for Toto models.

    Attributes:
        num_samples: Number of forecast samples (None = single mean prediction).
        samples_per_batch: Samples per batch during inference.
        max_steps: Number of fine-tuning gradient steps.
        lr: Learning rate for fine-tuning.
        min_lr: Minimum learning rate after decay.
        warmup_steps: LR warmup steps.
        stable_steps: Steps at peak LR before decay.
        decay_steps: Steps for LR decay.
        train_batch_size: Training batch size.
        val_batch_size: Validation batch size.
        val_prediction_len: Prediction length for validation loss.
    """

    def __init__(self, **kwargs):
        toto_specific_params = {
            "num_samples",
            "samples_per_batch",
            "max_steps",
            "num_epochs",
            "lr",
            "min_lr",
            "warmup_steps",
            "stable_steps",
            "decay_steps",
            "train_batch_size",
            "val_batch_size",
            "val_prediction_len",
            "covariate_cols",
        }

        # Filter out Toto-specific params from kwargs for parent class
        base_kwargs = {k: v for k, v in kwargs.items() if k not in toto_specific_params}

        # Call parent with filtered kwargs
        super().__init__(**base_kwargs)

        # Set Toto-specific defaults
        self.model_type = "toto"
        self.training_backend = TrainingBackend.CUSTOM

        # Inference
        self.num_samples = kwargs.get("num_samples", None)
        self.samples_per_batch = kwargs.get("samples_per_batch", 20)

        # Training — supports both max_steps and num_epochs.
        # num_epochs is used by the generic workflow; max_steps is Toto-native.
        # If neither is explicitly set, default to max_steps=3000.
        self.num_epochs = kwargs.get("num_epochs", None)
        self.max_steps = kwargs.get("max_steps", None if self.num_epochs else 3000)
        self.lr = kwargs.get("lr", 1e-4)
        self.min_lr = kwargs.get("min_lr", 1e-5)
        self.warmup_steps = kwargs.get("warmup_steps", 200)
        self.stable_steps = kwargs.get("stable_steps", 1000)
        self.decay_steps = kwargs.get("decay_steps", 1000)
        self.train_batch_size = kwargs.get("train_batch_size", 4)
        self.val_batch_size = kwargs.get("val_batch_size", 1)
        self.val_prediction_len = kwargs.get("val_prediction_len", 96)

        # Covariates — past-only exogenous variables (e.g., ["iob"])
        self.covariate_cols = kwargs.get("covariate_cols", None) or []
