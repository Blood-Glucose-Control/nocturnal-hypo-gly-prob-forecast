"""
Toto configuration classes.
"""

from src.models.base import ModelConfig, TrainingBackend


class TotoConfig(ModelConfig):
    """Configuration class for Toto models."""

    def __init__(self, **kwargs):
        toto_specific_params = {"num_samples", "samples_per_batch"}

        # Filter out Toto-specific params from kwargs for parent class
        base_kwargs = {
            k: v for k, v in kwargs.items() if k not in toto_specific_params
        }

        # Call parent with filtered kwargs
        super().__init__(**base_kwargs)

        # Set Toto-specific defaults
        self.model_type = "toto"
        self.training_backend = TrainingBackend.CUSTOM
        self.num_samples = kwargs.get("num_samples", 20)
        self.samples_per_batch = kwargs.get("samples_per_batch", 20)
