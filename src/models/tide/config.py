"""
TiDE Architecture configuration classes.

This module provides configuration classes specific to
the TiDE architecture, extending the base model
configuration with TiDE-specific parameters.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from src.models.base import ModelConfig, TrainingBackend

@dataclass
class TiDEConfig(ModelConfig):
    """Configuration class for TiDE model, extending the base ModelConfig.
    
    This configuration class specializes ModelConfig for the TiDE (Time-series Dense Encoder)
    architecture, adding TiDE-specific parameters while inheriting common training parameters.
    
    Attributes:
        target: Name of the target column to predict (e.g., blood glucose).
        seq_len: Input sequence length in time steps.
        lookback_len: Lookback length for decoder (usually same as seq_len).
        feat_size: Feature projection size in the encoder.
        hidden_size: Hidden layer dimension in dense layers.
        num_encoder_layers: Number of dense encoder layers.
        num_decoder_layers: Number of dense decoder layers.
        decoder_output_dim: Decoder output dimension per timestep.
        temporal_decoder_hidden: Hidden size for temporal decoder.
        drop_prob: Dropout probability (overrides base dropout).
    """
    # Data parameters
    target: str = "bg"  # Default target column
    
    # Sequence parameters (override base parameters)
    context_length: int = 72   # Input sequence length (e.g., 6 hours at 5-min intervals)
    forecast_length: int = 12  # Prediction horizon (e.g., 1 hour at 5-min intervals)
    
    # TiDE-specific model parameters
    seq_len: int = 72          # Alias for context_length for TiDE compatibility
    lookback_len: int = 72     # Lookback length for decoder
    feat_size: int = 4         # Feature projection size
    hidden_size: int = 128     # Hidden layer size
    num_encoder_layers: int = 2  # Number of encoder layers
    num_decoder_layers: int = 2  # Number of decoder layers
    decoder_output_dim: int = 8  # Decoder output dimension per timestep
    temporal_decoder_hidden: int = 64  # Temporal decoder hidden size
    
    # Override base parameters with TiDE defaults
    model_type: str = "tide"
    dropout: float = 0.3       # TiDE uses higher dropout
    batch_size: int = 32
    num_epochs: int = 50
    learning_rate: float = 1e-4
    early_stopping_patience: int = 10
    
    # Data split ratios
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Override base parameters that aren't used in TiDE
    d_model: Optional[int] = None  # Not used in TiDE (uses hidden_size instead)
    n_heads: Optional[int] = None  # Not used in TiDE (not transformer-based)
    n_layers: Optional[int] = None  # Not used in TiDE (uses num_encoder/decoder_layers)
    
    def __post_init__(self):
        """Ensure consistency between TiDE-specific and base parameters."""
        # Sync seq_len with context_length
        if self.seq_len != self.context_length:
            self.seq_len = self.context_length
        
        # Sync lookback_len with context_length if not explicitly set
        if self.lookback_len != self.context_length:
            self.lookback_len = self.context_length
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a dictionary, excluding None values.
        
        Returns:
            Dict[str, Any]: Dictionary containing all non-None configuration parameters.
        """
        result = {}
        for k, v in self.__dict__.items():
            # Skip None values (unused parameters)
            if v is None:
                continue
            # Handle enum values
            if hasattr(v, "value"):
                result[k] = v.value
            else:
                result[k] = v
        return result