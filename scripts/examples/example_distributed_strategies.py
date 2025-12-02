#!/usr/bin/env python3
"""
Advanced distributed training configurations for TTM.

This shows different distributed strategies and how they map to your DistributedConfig.

How to run:
  # Just shows configuration examples (no actual training):
  python example_distributed_strategies.py
  
  # To actually train with these configs, copy the desired function
  # and use it in your training script like:
  #   model = example_ddp_config()
  #   model.fit("kaggle_brisT1D", output_dir="./output")

Strategies covered:
  1. DDP (Distributed Data Parallel) - Most common, works great with TTM
  2. DeepSpeed - Memory efficient, good for larger models  
  3. FSDP (Fully Sharded Data Parallel) - For very large models
  
Choose based on:
  - 2-8 GPUs: Use DDP
  - Memory constraints: Use DeepSpeed
  - Very large models (>1B params): Use FSDP
"""

from src.models.base import DistributedConfig
from src.models.ttm import TTMForecaster, TTMConfig
from src.utils.logging_helper import info_print

def example_ddp_config():
    """Standard DDP (Distributed Data Parallel) - Most common."""
    info_print("🔥 DDP Configuration (Most Stable)")
    
    distributed_config = DistributedConfig(
        enabled=True,
        strategy="ddp",
        world_size=4,  # Number of GPUs
        backend="nccl",  # Best for NVIDIA GPUs
        find_unused_parameters=False,  # TTM has static architecture - better performance
        gradient_as_bucket_view=True,  # Memory efficient gradient handling
    )
    
    config = TTMConfig(
        model_path="ibm-granite/granite-timeseries-ttm-r2",
        batch_size=16,  # Per-device batch size (total = 16 * 4 = 64)
        learning_rate=4e-4,  # Scale LR with world_size if needed
        use_cpu=False,
    )
    
    model = TTMForecaster(config, distributed_config=distributed_config)
    info_print("✅ DDP model ready - optimized for TTM performance!")
    info_print("   📈 find_unused_parameters=False eliminates extra graph traversal")
    return model

def example_deepspeed_config():
    """DeepSpeed for very large models or memory optimization."""
    info_print("⚡ DeepSpeed Configuration (Memory Efficient)")
    
    # DeepSpeed config (can also load from JSON file)
    deepspeed_config = {
        "train_batch_size": 64,  # Global batch size
        "train_micro_batch_size_per_gpu": 16,  # Per-GPU batch size
        "fp16": {"enabled": True},
        "zero_optimization": {
            "stage": 1,  # Stage 1 = optimizer state partitioning
        },
        "optimizer": {
            "type": "AdamW",
            "params": {"lr": 1e-4, "weight_decay": 0.01}
        }
    }
    
    distributed_config = DistributedConfig(
        enabled=True,
        strategy="deepspeed",
        deepspeed_config=deepspeed_config,
    )
    
    config = TTMConfig(
        model_path="ibm-granite/granite-timeseries-ttm-r2",
        # Note: batch_size ignored when using DeepSpeed config
        use_cpu=False,
    )
    
    model = TTMForecaster(config, distributed_config=distributed_config)
    info_print("✅ DeepSpeed model ready - great for large TTM models!")
    return model

def example_fsdp_config():
    """FSDP (Fully Sharded Data Parallel) for very large models."""
    info_print("🔄 FSDP Configuration (Ultra Large Models)")
    
    fsdp_config = {
        "fsdp_config": {
            "min_num_params": 1e6,  # Shard parameters > 1M
            "xla": False,
            "xla_fsdp_grad_ckpt": False,
        }
    }
    
    distributed_config = DistributedConfig(
        enabled=True,
        strategy="fsdp",
        fsdp_config=fsdp_config,
    )
    
    config = TTMConfig(
        model_path="ibm-granite/granite-timeseries-ttm-r2",
        batch_size=8,  # Smaller per-device for large models
        use_cpu=False,
    )
    
    model = TTMForecaster(config, distributed_config=distributed_config)
    info_print("✅ FSDP model ready - handles huge TTM variants!")
    return model

def main():
    """Show all distributed configurations."""
    info_print("🎯 TTM Distributed Training Strategies")
    info_print("=" * 50)
    
    try:
        example_ddp_config()
        info_print("")
        
        example_deepspeed_config()  
        info_print("")
        
        example_fsdp_config()
        info_print("")
        
        info_print("🎉 All configurations created successfully!")
        info_print("\nRecommendations:")
        info_print("• DDP: Best for most TTM training (2-8 GPUs)")
        info_print("• DeepSpeed: Good for memory optimization")
        info_print("• FSDP: Only needed for very large TTM variants")
        
    except Exception as e:
        info_print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()