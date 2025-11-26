"""
LoRA (Low-Rank Adaptation) utilities for efficient fine-tuning.

This module provides utilities for applying LoRA to time series foundation models
to enable memory-efficient fine-tuning of large pre-trained models.
"""

import json
import logging
import os
from typing import Dict, List, Optional, Union, Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class LoRALinear(nn.Module):
    """
    LoRA linear layer that adds trainable low-rank matrices to frozen linear layers.
    
    This implementation follows the LoRA paper: https://arxiv.org/abs/2106.09685
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: int = 32,
        dropout: float = 0.0,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """
        Initialize LoRA linear layer.
        
        Args:
            in_features: Number of input features
            out_features: Number of output features
            rank: LoRA rank (dimensionality of the low-rank adaptation)
            alpha: LoRA scaling parameter
            dropout: Dropout probability
            bias: Whether to include bias
            device: Device to place the layer on
            dtype: Data type for the layer
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA matrices
        self.lora_A = nn.Linear(in_features, rank, bias=False, device=device, dtype=dtype)
        self.lora_B = nn.Linear(rank, out_features, bias=False, device=device, dtype=dtype)
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        
        # Initialize LoRA weights
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize LoRA parameters following the original paper."""
        # Initialize A with Gaussian, B with zeros (so initial LoRA output is zero)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through LoRA adaptation."""
        # LoRA forward: x @ (A @ B) * scaling
        lora_output = self.lora_B(self.lora_A(self.dropout(x)))
        return lora_output * self.scaling


def apply_lora_to_model(
    model: nn.Module,
    target_modules: List[str],
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.0,
    lora_config: Optional[Dict[str, Any]] = None
) -> nn.Module:
    """
    Apply LoRA to specified modules in a model.
    
    Args:
        model: The model to apply LoRA to
        target_modules: List of module names/patterns to target
        rank: LoRA rank
        alpha: LoRA scaling parameter
        dropout: Dropout probability
        lora_config: Additional LoRA configuration
        
    Returns:
        Model with LoRA applied
    """
    if lora_config is not None:
        rank = lora_config.get("rank", rank)
        alpha = lora_config.get("alpha", alpha)
        dropout = lora_config.get("dropout", dropout)
        target_modules = lora_config.get("target_modules", target_modules)
    
    logger.info(f"Applying LoRA with rank={rank}, alpha={alpha}, dropout={dropout}")
    logger.info(f"Target modules: {target_modules}")
    
    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False
    
    # Apply LoRA to target modules
    lora_modules_added = 0
    for name, module in model.named_modules():
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                # Replace linear layer with LoRA-enhanced version
                lora_layer = LoRALinear(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                    bias=hasattr(module, 'bias') and module.bias is not None,
                    device=module.weight.device,
                    dtype=module.weight.dtype
                )
                
                # Replace the module
                _set_module_by_name(model, name, LoRAEnhancedLinear(module, lora_layer))
                lora_modules_added += 1
                logger.info(f"Applied LoRA to {name}")
    
    logger.info(f"LoRA applied to {lora_modules_added} modules")
    
    # Print parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Trainable percentage: {100 * trainable_params / total_params:.2f}%")
    
    return model


class LoRAEnhancedLinear(nn.Module):
    """
    Linear layer enhanced with LoRA adaptation.
    
    This combines the original frozen linear layer with LoRA adaptation.
    """
    
    def __init__(self, original_layer: nn.Linear, lora_layer: LoRALinear):
        """
        Initialize LoRA-enhanced linear layer.
        
        Args:
            original_layer: The original linear layer (will be frozen)
            lora_layer: The LoRA adaptation layer
        """
        super().__init__()
        
        self.original_layer = original_layer
        self.lora_layer = lora_layer
        
        # Freeze original layer
        for param in self.original_layer.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass combining original layer and LoRA adaptation."""
        original_output = self.original_layer(x)
        lora_output = self.lora_layer(x)
        return original_output + lora_output


def _set_module_by_name(model: nn.Module, name: str, module: nn.Module) -> None:
    """Set a module in the model by its dotted name."""
    parts = name.split('.')
    parent = model
    
    for part in parts[:-1]:
        parent = getattr(parent, part)
    
    setattr(parent, parts[-1], module)


def get_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Extract only the LoRA parameters from a model.
    
    Args:
        model: Model with LoRA layers
        
    Returns:
        State dict containing only LoRA parameters
    """
    lora_state_dict = {}
    
    for name, module in model.named_modules():
        if isinstance(module, LoRAEnhancedLinear):
            # Save LoRA parameters
            lora_prefix = f"{name}.lora_layer"
            for param_name, param in module.lora_layer.named_parameters():
                lora_state_dict[f"{lora_prefix}.{param_name}"] = param
    
    return lora_state_dict


def load_lora_state_dict(model: nn.Module, lora_state_dict: Dict[str, torch.Tensor]) -> None:
    """
    Load LoRA parameters into a model.
    
    Args:
        model: Model with LoRA layers
        lora_state_dict: State dict containing LoRA parameters
    """
    for name, param in lora_state_dict.items():
        # Navigate to the parameter in the model
        parts = name.split('.')
        module = model
        
        for part in parts[:-1]:
            module = getattr(module, part)
        
        # Set the parameter
        if hasattr(module, parts[-1]):
            target_param = getattr(module, parts[-1])
            target_param.data.copy_(param.data)


def save_lora_adapters(
    model: nn.Module,
    save_directory: str,
    config: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save LoRA adapters to a directory.
    
    Args:
        model: Model with LoRA layers
        save_directory: Directory to save adapters
        config: LoRA configuration to save
    """
    os.makedirs(save_directory, exist_ok=True)
    
    # Save LoRA parameters
    lora_state_dict = get_lora_state_dict(model)
    torch.save(lora_state_dict, os.path.join(save_directory, "adapter_model.bin"))
    
    # Save configuration
    if config is not None:
        with open(os.path.join(save_directory, "adapter_config.json"), "w") as f:
            json.dump(config, f, indent=2)
    
    logger.info(f"LoRA adapters saved to {save_directory}")


def load_lora_adapters(
    model: nn.Module,
    adapter_directory: str
) -> Dict[str, Any]:
    """
    Load LoRA adapters from a directory.
    
    Args:
        model: Model to load adapters into
        adapter_directory: Directory containing adapters
        
    Returns:
        Loaded configuration
    """
    # Load LoRA parameters
    adapter_path = os.path.join(adapter_directory, "adapter_model.bin")
    device = next(model.parameters()).device if hasattr(model, 'parameters') else 'cpu'
    lora_state_dict = torch.load(adapter_path, map_location=device)
    load_lora_state_dict(model, lora_state_dict)
    
    # Load configuration
    config_path = os.path.join(adapter_directory, "adapter_config.json")
    config = {}
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
    
    logger.info(f"LoRA adapters loaded from {adapter_directory}")
    return config


class LoRATrainer:
    """
    Specialized trainer for LoRA fine-tuning.
    
    This class provides utilities for training models with LoRA adaptations,
    including gradient scaling and adapter management.
    """
    
    def __init__(
        self,
        model: nn.Module,
        rank: int = 16,
        alpha: int = 32,
        target_modules: Optional[List[str]] = None
    ):
        """
        Initialize LoRA trainer.
        
        Args:
            model: Base model to apply LoRA to
            rank: LoRA rank
            alpha: LoRA scaling parameter
            target_modules: Modules to target for LoRA
        """
        self.model = model
        self.rank = rank
        self.alpha = alpha
        self.target_modules = target_modules or ["q_proj", "v_proj", "k_proj", "o_proj"]
        
        # Apply LoRA to model
        self.model = apply_lora_to_model(
            self.model,
            target_modules=self.target_modules,
            rank=rank,
            alpha=alpha
        )
    
    def get_trainable_parameters(self) -> List[torch.nn.Parameter]:
        """Get only the trainable LoRA parameters."""
        return [p for p in self.model.parameters() if p.requires_grad]
    
    def save_adapters(self, save_directory: str) -> None:
        """Save LoRA adapters."""
        config = {
            "rank": self.rank,
            "alpha": self.alpha,
            "target_modules": self.target_modules,
        }
        save_lora_adapters(self.model, save_directory, config)
    
    def load_adapters(self, adapter_directory: str) -> None:
        """Load LoRA adapters."""
        load_lora_adapters(self.model, adapter_directory)


# Helper functions for common LoRA configurations

def get_llama_lora_config(rank: int = 16, alpha: int = 32) -> Dict[str, Any]:
    """Get LoRA configuration for LLaMA-style models."""
    return {
        "rank": rank,
        "alpha": alpha,
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "dropout": 0.1,
        "bias": "none"
    }


def get_bert_lora_config(rank: int = 16, alpha: int = 32) -> Dict[str, Any]:
    """Get LoRA configuration for BERT-style models."""
    return {
        "rank": rank,
        "alpha": alpha,
        "target_modules": ["query", "value", "key", "dense"],
        "dropout": 0.1,
        "bias": "none"
    }


def get_time_series_lora_config(rank: int = 16, alpha: int = 32) -> Dict[str, Any]:
    """Get LoRA configuration for time series models."""
    return {
        "rank": rank,
        "alpha": alpha,
        "target_modules": [
            "q_proj", "v_proj", "k_proj", "o_proj",  # Attention layers
            "fc", "linear", "projection",  # Feed-forward layers
            "mixer", "temporal_mixer"  # Time series specific layers
        ],
        "dropout": 0.1,
        "bias": "none"
    }
