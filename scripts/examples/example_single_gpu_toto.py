#!/usr/bin/env python3
"""
Single-GPU Toto fine-tuning example.

Demonstrates how to fine-tune the Toto time-series foundation model
on blood glucose forecasting data using the BaseTSFM framework.
"""

import os
from src.models.base import DistributedConfig, GPUManager
from src.models.toto import TotoForecaster, TotoConfig
from src.utils.logging_helper import info_print


def main():
    """Single-GPU Toto fine-tuning."""

    # 1. Hardware Detection
    gpu_info = GPUManager.get_gpu_info()
    info_print("GPU Detection:")
    info_print(f"   Available: {gpu_info['gpu_available']}")

    distributed_config = DistributedConfig(enabled=False)
    use_cpu = not gpu_info["gpu_available"]
    device = "cuda" if gpu_info["gpu_available"] else "cpu"

    info_print(f"Using device: {device.upper()}")

    # 2. Toto Configuration
    # Context: 42 hours, Forecast: 6 hours (matches optimal zero-shot config)
    # Total: 504 + 72 = 576 timesteps = 9 patches (divisible by patch_size=64)
    config = TotoConfig(
        model_path="Datadog/Toto-Open-Base-1.0",
        context_length=504,      # 42 hours at 5-min resolution
        forecast_length=72,      # 6 hours at 5-min resolution
        batch_size=16,
        learning_rate=1e-5,
        num_epochs=10,
        use_cpu=use_cpu,
        fp16=False,              # Keep False for numerical stability

        # Training settings
        gradient_accumulation_steps=1,
        logging_steps=500,
        save_steps=5000,
        dataloader_num_workers=4,

        # Early stopping
        early_stopping_patience=5,
    )

    # 3. Model Initialization
    model = TotoForecaster(config, distributed_config=distributed_config)
    info_print("Toto Model created!")

    # 4. Output Setup
    output_dir = "./trained_models/artifacts/_tsfm_testing/output_single_gpu_toto"
    os.makedirs(output_dir, exist_ok=True)
    info_print(f"Training outputs to {output_dir}")

    # 5. Training
    info_print("Starting training...")
    info_print("   Context: 504 steps (42h), Forecast: 72 steps (6h)")
    info_print(f"   Epochs: {config.num_epochs}, Batch size: {config.batch_size}")

    try:
        results = model.fit(
            train_data="kaggle_brisT1D",
            output_dir=output_dir,
        )

        info_print("Training completed!")
        info_print(f"Metrics: {list(results.keys())}")
        return results

    except Exception as e:
        info_print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
