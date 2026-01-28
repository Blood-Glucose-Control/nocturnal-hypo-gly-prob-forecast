"""Toto configuration class."""

from src.models.base import ModelConfig, TrainingStrategy


class TotoConfig(ModelConfig):
    """Configuration for Toto model."""

    def __init__(self, **kwargs):
        # Toto-specific params to extract before parent init
        toto_params = {
            "patch_size", "stride", "input_features", "target_feature",
            "use_nll_loss", "gradient_accumulation_steps", "mse_weight",
        }
        base_kwargs = {k: v for k, v in kwargs.items() if k not in toto_params}

        super().__init__(**base_kwargs)

        self.model_type = "toto"
        self.training_strategy = TrainingStrategy.TRANSFORMERS

        if self.model_path is None:
            self.model_path = "Datadog/Toto-Open-Base-1.0"

        # Architecture (from pretrained model)
        self.patch_size = kwargs.get("patch_size", 64)
        self.stride = kwargs.get("stride", 64)

        # Training
        self.freeze_backbone = kwargs.get("freeze_backbone", False)
        self.use_nll_loss = kwargs.get("use_nll_loss", True)
        self.gradient_accumulation_steps = kwargs.get("gradient_accumulation_steps", 1)
        self.mse_weight = kwargs.get("mse_weight", 0.1)

        # Data
        self.input_features = kwargs.get("input_features", ["bg_mM"])
        self.target_feature = kwargs.get("target_feature", "bg_mM")
