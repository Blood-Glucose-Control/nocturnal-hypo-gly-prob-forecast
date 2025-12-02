#!/usr/bin/env python3
"""
Example demonstrating multi-GPU distributed training with TTM.

This script shows how to use your existing framework for distributed training
with minimal changes. TTM + transformers.Trainer makes it almost effortless!

How to run:
  # Single GPU or CPU (automatic detection):
  python example_distributed_ttm.py

  # Multi-GPU with torchrun (recommended):
  torchrun --nproc_per_node=2 --nnodes=1 example_distributed_ttm.py

  # Multi-GPU with environment variables:
  CUDA_VISIBLE_DEVICES=0,1 WORLD_SIZE=2 python example_distributed_ttm.py

  # Use the bash script for automatic GPU detection:
  bash run_distributed_ttm.sh

What this demonstrates:
  - Automatic GPU detection and configuration
  - DistributedConfig setup for different scenarios
  - Identical training code for single/multi-GPU
  - transformers.Trainer handling all distributed complexity
"""

import os
from src.models.base import DistributedConfig, GPUManager
from src.models.ttm import TTMForecaster, TTMConfig
from src.utils.logging_helper import info_print


def main():
    """Demonstrate distributed TTM training."""

    # 1. Detect available GPUs
    gpu_info = GPUManager.get_gpu_info()
    info_print("🔍 GPU Detection:")
    info_print(f"   Available: {gpu_info['gpu_available']}")
    info_print(f"   Count: {gpu_info['gpu_count']}")

    if gpu_info["gpu_count"] <= 1:
        info_print("⚠️  Single GPU or CPU detected - demonstrating config anyway")

    # 2. Create distributed configuration
    # Check if we're running with torchrun (has RANK environment variable)
    rank = int(os.environ.get("RANK", -1))
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))
    master_addr = os.environ.get("MASTER_ADDR", None)

    # Show rank information from each process (useful for debugging)
    if rank >= 0:
        info_print(
            f"Process info: rank {rank}, local_rank {local_rank}", rank_zero_only=False
        )
    else:
        info_print("No distributed environment detected (RANK not set)")

    # Determine if we can actually do distributed training
    can_do_distributed = (
        gpu_info["gpu_count"] > 1
        and rank >= 0
        and world_size > 1
        and master_addr is not None
    )

    # Provide clear guidance if user wants distributed but environment isn't set up
    if gpu_info["gpu_count"] > 1 and not can_do_distributed:
        info_print(
            "🚨 Multi-GPU Setup Detected but Distributed Training Not Configured!"
        )
        info_print("")
        info_print("To run distributed training, use one of these methods:")
        info_print("1. Use torchrun (recommended):")
        info_print(
            f"   torchrun --nproc_per_node={gpu_info['gpu_count']} --nnodes=1 scripts/examples/example_distributed_ttm.py"
        )
        info_print("")
        info_print("2. Use the bash script:")
        info_print("   bash scripts/examples/run_distributed_ttm.sh")
        info_print("")
        info_print("3. Set environment manually (advanced):")
        info_print("   export MASTER_ADDR=127.0.0.1")
        info_print("   export MASTER_PORT=29500")
        info_print(f"   export WORLD_SIZE={gpu_info['gpu_count']}")
        info_print("   # Then run with RANK=0,1,2,... for each process")
        info_print("")
        info_print("Falling back to single-GPU training...")
        info_print("")

    if can_do_distributed:
        distributed_config = DistributedConfig(
            enabled=True,
            strategy="ddp",  # DDP is most stable for TTM
            world_size=world_size,
            local_rank=local_rank,
            find_unused_parameters=False,  # TTM has static architecture - disable for performance
        )
        use_cpu = False
        info_print(
            f"🚀 Configured for {world_size} GPUs with DDP (rank {rank}/{world_size})"
        )
        info_print(f"   MASTER_ADDR: {master_addr}")
        info_print(f"   MASTER_PORT: {os.environ.get('MASTER_PORT', 'default')}")
        info_print("   📈 Performance: find_unused_parameters=False (faster training)")
    else:
        distributed_config = DistributedConfig(enabled=False)
        use_cpu = not gpu_info["gpu_available"]
        if gpu_info["gpu_count"] > 1:
            info_print(
                "⚠️  Multiple GPUs detected but distributed environment not set up"
            )
            info_print(
                "   Run with: torchrun --nproc_per_node=2 --nnodes=1 example_distributed_ttm.py"
            )
            info_print("   or: bash run_distributed_ttm.sh")
        info_print("📱 Configured for single device training")

    # 3. Create TTM configuration (same as before!)
    config = TTMConfig(
        model_path="ibm-granite/granite-timeseries-ttm-r2",
        context_length=512,
        forecast_length=96,
        batch_size=32,  # Per-device batch size
        learning_rate=1e-4,
        num_epochs=2,
        use_cpu=use_cpu,
        fp16=gpu_info["gpu_available"],  # Enable fp16 for GPU
    )

    # 4. Create model with distributed config - THAT'S IT!
    model = TTMForecaster(config, distributed_config=distributed_config)

    info_print("✅ TTM Model created with distributed support!")

    # 5. Check what got configured
    model_info = model.get_model_info()
    info_print("📊 Model Info:")
    info_print(f"   Distributed enabled: {model_info['distributed_enabled']}")
    info_print(f"   Training strategy: {model_info['training_strategy']}")

    # 6. Training works exactly the same!
    output_dir = "./trained_models/artifacts/_tsfm_testing/output_distributed_ttm"
    os.makedirs(output_dir, exist_ok=True)
    info_print(f"🏋️‍♂️ Training outputs to {output_dir}")

    info_print("🎯 Starting distributed training...")
    info_print("   (This will use transformers.Trainer with automatic DDP)")

    try:
        # This call is identical whether single or multi-GPU!
        results = model.fit(train_data="kaggle_brisT1D", output_dir=output_dir)
        return results

    except KeyboardInterrupt:
        info_print("🛑 Training interrupted by user")
        # Ensure cleanup happens even with Ctrl+C
        if hasattr(model, "_cleanup_distributed"):
            model._cleanup_distributed()
        raise
    except Exception as e:
        info_print(f"❌ Training failed: {e}")
        # Ensure cleanup happens even on error
        if hasattr(model, "_cleanup_distributed"):
            model._cleanup_distributed()
        raise


if __name__ == "__main__":
    main()
