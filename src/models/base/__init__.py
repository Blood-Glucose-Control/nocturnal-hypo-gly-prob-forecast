"""
Base model framework for Time Series Foundation Models.

This module provides the foundational classes and utilities for implementing
time series foundation models in a unified, scalable framework.
"""

from .base_model import (
    BaseTSFM,
    ModelConfig,
    LoRAConfig,
    DistributedConfig,
    create_model_from_config,
    compare_models,
)

from .distributed import (
    DistributedManager,
    setup_deepspeed_config,
    setup_fsdp_config,
    GPUManager,
    distributed_manager,
)

from .lora_utils import (
    LoRALinear,
    LoRAEnhancedLinear,
    LoRATrainer,
    apply_lora_to_model,
    save_lora_adapters,
    load_lora_adapters,
    get_lora_state_dict,
    load_lora_state_dict,
    get_llama_lora_config,
    get_bert_lora_config,
    get_time_series_lora_config,
)

__all__ = [
    # Base model classes
    "BaseTSFM",
    "ModelConfig", 
    "LoRAConfig",
    "DistributedConfig",
    
    # Factory functions
    "create_model_from_config",
    "compare_models",
    
    # Distributed training
    "DistributedManager",
    "setup_deepspeed_config", 
    "setup_fsdp_config",
    "GPUManager",
    "distributed_manager",
    
    # LoRA utilities
    "LoRALinear",
    "LoRAEnhancedLinear", 
    "LoRATrainer",
    "apply_lora_to_model",
    "save_lora_adapters",
    "load_lora_adapters",
    "get_lora_state_dict",
    "load_lora_state_dict",
    "get_llama_lora_config",
    "get_bert_lora_config",
    "get_time_series_lora_config",
]