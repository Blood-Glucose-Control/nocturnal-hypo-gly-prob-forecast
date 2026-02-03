"""
Sundial configuration classes.
"""

from src.models.base import ModelConfig, TrainingBackend

class SundialConfig(ModelConfig):
    """Configuration class for Sundial models."""
    def __init__(self, **kwargs):
        sundial_specific_params = {
        }

        # Filter out Sundial-specific params from kwargs for parent class
        base_kwargs = {k: v for k, v in kwargs.items() if k not in sundial_specific_params}

        # Call parent with filtered kwargs
        super().__init__(**base_kwargs)

        # Set Sundial-specific defaults
        self.model_type = "sundial"
        self.training_backend = TrainingBackend.TRANSFORMERS
        self.num_samples = kwargs.get("num_samples", 100)

