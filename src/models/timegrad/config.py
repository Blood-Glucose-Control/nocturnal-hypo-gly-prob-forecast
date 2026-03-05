"""TimeGrad configuration classes."""

from src.models.base import ModelConfig, TrainingBackend


class TimeGradConfig(ModelConfig):
    """Configuration class for TimeGrad models.

    TimeGrad is a diffusion-based probabilistic forecasting model that uses
    a GRU encoder with a denoising diffusion head. It trains from scratch
    (not a foundation model) and requires the pytorchts library.
    """

    # Parameters specific to TimeGrad that should not be passed to the parent
    # ModelConfig dataclass. This includes both TimeGrad-specific params and
    # data/feature config keys that are handled at the pipeline level (not stored
    # in ModelConfig) but may arrive via config.extra_config from a YAML file.
    TIMEGRAD_PARAMS = {
        # TimeGrad-specific
        "diff_steps",
        "beta_end",
        "beta_schedule",
        "loss_type",
        "cell_type",
        "num_cells",
        "num_layers",
        "num_samples",
        "scaling",
        "residual_layers",
        "residual_channels",
        "num_batches_per_epoch",
        "freq",
        # Data/feature config keys (handled by the data pipeline, not ModelConfig)
        "input_features",
        "target_features",
        "scaler_type",
        "resolution_min",
        "split_config",
        "fewshot_percent",
    }

    def __init__(self, **kwargs):
        # Filter out TimeGrad-specific params from kwargs for parent class
        base_kwargs = {k: v for k, v in kwargs.items() if k not in self.TIMEGRAD_PARAMS}

        # Call parent with filtered kwargs
        super().__init__(**base_kwargs)

        # Override base defaults for TimeGrad
        self.model_type = "timegrad"
        self.training_backend = TrainingBackend.CUSTOM
        self.training_mode = kwargs.get("training_mode", "from_scratch")

        # Diffusion parameters
        self.diff_steps = kwargs.get("diff_steps", 100)
        self.beta_end = kwargs.get("beta_end", 0.1)
        self.beta_schedule = kwargs.get("beta_schedule", "linear")
        self.loss_type = kwargs.get("loss_type", "l2")

        # RNN encoder parameters
        self.cell_type = kwargs.get("cell_type", "GRU")
        self.num_cells = kwargs.get("num_cells", 40)
        self.num_layers = kwargs.get("num_layers", 2)

        # Denoising network parameters
        self.residual_layers = kwargs.get("residual_layers", 8)
        self.residual_channels = kwargs.get("residual_channels", 8)

        # Inference parameters
        self.num_samples = kwargs.get("num_samples", 100)

        # Training parameters
        self.scaling = kwargs.get("scaling", True)
        self.num_batches_per_epoch = kwargs.get("num_batches_per_epoch", 100)

        # Time series frequency (GluonTS format, e.g. "5T" for 5-minute)
        self.freq = kwargs.get("freq", "5T")
