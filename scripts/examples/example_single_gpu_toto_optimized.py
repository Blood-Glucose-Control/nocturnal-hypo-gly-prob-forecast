#!/usr/bin/env python3
"""
Optimized single-GPU Toto training with faster settings.
Uses larger batch size and gradient accumulation for speed.
"""

import os
import torch
from src.models.base import DistributedConfig, GPUManager
from src.models.toto import TotoForecaster, TotoConfig
from src.utils.logging_helper import info_print

def main():
    """Optimized single-GPU Toto training."""

    # 1. Hardware Detection
    gpu_info = GPUManager.get_gpu_info()
    info_print("🔍 GPU Detection:")
    info_print(f"   Available: {gpu_info['gpu_available']}")

    distributed_config = DistributedConfig(enabled=False)
    use_cpu = not gpu_info["gpu_available"]
    device = "cuda" if gpu_info["gpu_available"] else "cpu"

    info_print(f"📱 Using single device training: {device.upper()}")

    # 2. Optimized Toto Configuration
    # Matching zero-shot optimal: 42h context, 6h forecast
    config = TotoConfig(
        model_path="Datadog/Toto-Open-Base-1.0",
        context_length=504,      # 42 hours (matches optimal zero-shot)
        forecast_length=72,      # 6 hours (matches optimal zero-shot)
        batch_size=16,           # INCREASED from 8 for speed (if GPU memory allows)
        learning_rate=1e-5,
        num_epochs=10,
        use_cpu=use_cpu,
        fp16=False,              # Keep False for stability

        # Speed optimizations:
        gradient_accumulation_steps=1,  # Set to 2 if batch_size=8
        logging_steps=500,               # Log less frequently
        save_steps=5000,                 # Save less frequently
        dataloader_num_workers=4,        # More workers for faster data loading
    )

    # 3. Model Initialization
    model = TotoForecaster(config, distributed_config=distributed_config)
    info_print("✅ Toto Model created!")

    # 4. Output Setup
    output_dir = "./trained_models/artifacts/_tsfm_testing/output_single_gpu_toto_optimized"
    os.makedirs(output_dir, exist_ok=True)
    info_print(f"🏋️‍♂️ Training outputs to {output_dir}")

    # 5. Training Loop
    info_print("🎯 Starting optimized training...")
    info_print("   Context: 504 steps (42h), Forecast: 72 steps (6h)")
    info_print("   Epochs: 10, Batch size: 16, Workers: 4")
    info_print("   Estimated time: 3-4 hours")

    try:
        results = model.fit(
            train_data="kaggle_brisT1D",
            output_dir=output_dir
        )

        info_print("🏁 Training completed!")
        info_print(f"📊 Metrics: {list(results.keys())}")
        return results

    except Exception as e:
        info_print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
