#!/usr/bin/env python3
"""
Simple single-GPU version of Toto training for testing.
Follows the BaseTSFM pattern from the TTM framework.
"""

import os
import torch
from src.models.base import DistributedConfig, GPUManager
from src.models.toto import TotoForecaster, TotoConfig  # Assuming these wrappers exist
from src.utils.logging_helper import info_print

def main():
    """Simple single-GPU Toto training for quick iteration."""

    # 1. Hardware Detection
    gpu_info = GPUManager.get_gpu_info()
    info_print("🔍 GPU Detection:")
    info_print(f"   Available: {gpu_info['gpu_available']}")
    
    distributed_config = DistributedConfig(enabled=False)
    use_cpu = not gpu_info["gpu_available"]
    device = "cuda" if gpu_info["gpu_available"] else "cpu"

    info_print(f"📱 Using single device training: {device.upper()}")

    # 2. Toto Configuration (Adapted from TTM style)
    # Note: Toto uses 151M params, so we use a smaller batch and lower LR
    # IMPORTANT: Toto uses patch_size=64, so context+forecast must be divisible by 64
    # Matching zero-shot optimal configuration: ~42h context, 6h forecast
    config = TotoConfig(
        model_path="Datadog/Toto-Open-Base-1.0",
        context_length=504,      # 42 hours at 5-min resolution (7.875 patches of 64)
        forecast_length=72,      # 6 hours at 5-min resolution (1.125 patches of 64)
                                 # Total: 504 + 72 = 576 = 9 patches (divisible by 64)
        batch_size=8,            # Reduced for 151M param model memory safety
        learning_rate=1e-5,      # Lower LR for numerical stability
        num_epochs=10,           # Increased from 1 to 10 for better convergence
        use_cpu=use_cpu,
        fp16=False,              # Disabled - Toto's Student-T distribution is sensitive to fp16 NaNs
    )

    # 3. Model Initialization
    # Toto uses causal next-patch prediction with a patch size of 64
    model = TotoForecaster(config, distributed_config=distributed_config)
    info_print("✅ Toto Model created with Student-T Mixture head!")

    # 4. Output Setup
    output_dir = "./trained_models/artifacts/_tsfm_testing/output_single_gpu_toto"
    os.makedirs(output_dir, exist_ok=True)
    info_print(f"🏋️‍♂️ Training outputs to {output_dir}")

    # 5. Training Loop
    info_print("🎯 Starting single-GPU training iteration...")
    info_print("   Context: 504 steps (42h), Forecast: 72 steps (6h)")
    info_print("   Epochs: 10, Batch size: 8, Learning rate: 1e-5")
    try:
        # Toto handles multivariate inputs natively (CGM + carbs + insulin)
        # Note: The data loader automatically splits into train/validation
        results = model.fit(
            train_data="kaggle_brisT1D",
            output_dir=output_dir
        )

        info_print("🏁 Toto iteration completed!")
        info_print(f"📊 Metrics captured: {list(results.keys())}")
        return results

    except Exception as e:
        info_print(f"❌ Toto training failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()