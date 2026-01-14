#!/usr/bin/env python3
"""
LoRA (Low-Rank Adaptation) fine-tuning for Toto.

LoRA is more parameter-efficient than full fine-tuning:
- Only trains ~1-2% of parameters (adapter layers)
- Keeps pretrained knowledge intact
- Less prone to overfitting
- Faster training
"""

import os
from src.models.base import DistributedConfig, GPUManager, LoRAConfig
from src.models.toto import TotoForecaster, TotoConfig
from src.data.models import ColumnNames
from src.utils.logging_helper import info_print


def main():
    """LoRA fine-tuning for Toto."""

    # 1. Hardware Detection
    gpu_info = GPUManager.get_gpu_info()
    info_print("GPU Detection:")
    info_print(f"   Available: {gpu_info['gpu_available']}")

    distributed_config = DistributedConfig(enabled=False)
    use_cpu = not gpu_info["gpu_available"]
    device = "cuda" if gpu_info["gpu_available"] else "cpu"

    info_print(f"Using single device training: {device.upper()}")

    # 2. LoRA Configuration
    # LoRA adds small trainable adapters to attention layers
    # Toto uses wQKV (combined QKV) and wO (output projection) for attention
    lora_config = LoRAConfig(
        enabled=True,
        rank=16,  # Low rank (8-64 typical). Higher = more capacity but more params
        alpha=32,  # Scaling factor. Usually 2x rank
        dropout=0.1,  # Regularization
        target_modules=["wQKV", "wO"],  # Toto's attention layer names
        auto_detect_modules=False,  # Use explicit target modules
        bias="none",  # Don't train bias terms
    )

    info_print("LoRA Configuration:")
    info_print(f"   Rank: {lora_config.rank}")
    info_print(f"   Alpha: {lora_config.alpha}")
    info_print(f"   Dropout: {lora_config.dropout}")

    # 3. Toto Configuration
    # Multivariate input: BG + exogenous factors (insulin, carbs, activity)
    input_features = [
        ColumnNames.BG.value,
        ColumnNames.IOB.value,
        ColumnNames.COB.value,
        ColumnNames.STEPS.value,
        ColumnNames.CALS.value,
    ]

    config = TotoConfig(
        model_path="Datadog/Toto-Open-Base-1.0",
        context_length=504,  # 42 hours
        forecast_length=72,  # 6 hours
        batch_size=16,
        learning_rate=1e-5,  # Lower LR for more stable fine-tuning
        num_epochs=10,
        use_cpu=use_cpu,
        fp16=False,
        # Multivariate input features
        input_features=input_features,
        target_feature=ColumnNames.BG.value,  # Still predict BG only
        # Training settings
        gradient_accumulation_steps=1,
        logging_steps=500,
        save_steps=2000,
        dataloader_num_workers=4,
        # Regularization
        weight_decay=0.01,
        # Early stopping - higher patience to find better checkpoint
        early_stopping_patience=10,
        # Composite loss: NLL + MSE for better alignment with evaluation metric
        mse_weight=0.1,
    )

    # 4. Model Initialization with LoRA
    model = TotoForecaster(
        config, lora_config=lora_config, distributed_config=distributed_config
    )
    info_print("Toto Model with LoRA created!")

    # 5. Output Setup
    output_dir = "./trained_models/artifacts/_tsfm_testing/output_single_gpu_toto_lora"
    os.makedirs(output_dir, exist_ok=True)
    info_print(f"Training outputs to {output_dir}")

    # 6. Training Loop
    info_print("Starting LoRA fine-tuning...")
    info_print("   Context: 504 steps (42h), Forecast: 72 steps (6h)")
    info_print(f"   Features: {input_features}")
    info_print("   Epochs: 10, Batch size: 16, LR: 1e-5")
    info_print("   LoRA rank: 16, ~1-2% of params trainable")
    info_print("   Composite loss: NLL + 0.1*MSE, Patience: 10")

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
