#!/usr/bin/env python3
"""
Simple single-GPU version of TTM training for testing.

This script demonstrates TTM training without distributed complexity.
Use this to verify TTM training works before testing distributed.
"""

import os
from src.models.base import DistributedConfig, GPUManager
from src.models.ttm import TTMForecaster, TTMConfig
from src.utils.logging_helper import info_print


def main():
    """Simple single-GPU TTM training."""

    # Always use single device for simplicity
    gpu_info = GPUManager.get_gpu_info()
    info_print("🔍 GPU Detection:")
    info_print(f"   Available: {gpu_info['gpu_available']}")
    info_print(f"   Count: {gpu_info['gpu_count']}")

    # Force single device (no distributed)
    distributed_config = DistributedConfig(enabled=False)
    use_cpu = not gpu_info["gpu_available"]

    info_print("📱 Using single device training (no distributed)")

    # Simple TTM configuration
    config = TTMConfig(
        model_path="ibm-granite/granite-timeseries-ttm-r2",
        context_length=512,
        forecast_length=96,
        batch_size=16,  # Smaller batch for testing
        learning_rate=1e-4,
        num_epochs=1,  # Just 1 epoch for testing
        use_cpu=use_cpu,
        fp16=gpu_info["gpu_available"] and not use_cpu,
    )

    # Create model (no distributed config needed)
    model = TTMForecaster(config, distributed_config=distributed_config)
    info_print("✅ TTM Model created!")

    # Check model info
    model_info = model.get_model_info()
    info_print("📊 Model Info:")
    info_print(f"   Distributed enabled: {model_info['distributed_enabled']}")
    info_print(f"   Training strategy: {model_info['training_strategy']}")

    # Set up output directory
    output_dir = "./trained_models/artifacts/_tsfm_testing/output_single_gpu_ttm"
    os.makedirs(output_dir, exist_ok=True)
    info_print(f"🏋️‍♂️ Training outputs to {output_dir}")

    info_print("🎯 Starting single-GPU training...")

    try:
        # Train the model
        results = model.fit(train_data="eval", output_dir=output_dir)

        info_print("🏁 Training completed successfully!")
        info_print(f"📊 Results: {list(results.keys())}")

        return results

    except KeyboardInterrupt:
        info_print("🛑 Training interrupted by user")
        raise
    except Exception as e:
        info_print(f"❌ Training failed: {e}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
