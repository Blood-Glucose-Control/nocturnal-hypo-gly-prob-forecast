#!/usr/bin/env python3
"""
Example script demonstrating the use of the new base TSFM framework.

This script shows how to:
1. Configure and initialize models
2. Set up distributed training and LoRA
3. Train models using the unified interface
4. Save and load models
5. Make predictions

Run this script to test the base model framework implementation.
"""

import os
import json
import argparse
from datetime import datetime

# Import the base framework
from src.models.base import (
    LoRAConfig,
    DistributedConfig,
    GPUManager,
)

# Import TTM implementation
from src.models.ttm import TTMForecaster, TTMConfig

# Import utilities
from src.utils.logging_helper import info_print, error_print


def example_basic_usage():
    """Example 1: Basic model creation and usage."""
    info_print("=" * 60)
    info_print("Example 1: Basic TTM model creation")
    info_print("=" * 60)

    # Create a basic TTM configuration
    config = TTMConfig(
        model_path="ibm-granite/granite-timeseries-ttm-r2",
        context_length=512,
        forecast_length=96,
        batch_size=64,
        learning_rate=1e-4,
        num_epochs=5,
        use_cpu=True,  # Set to False if you have GPU
    )

    # Create the model
    model = TTMForecaster(config)

    # Print model information
    model_info = model.get_ttm_specific_info()
    info_print("Model created successfully!")
    info_print(f"Model type: {model_info['model_type']}")
    info_print(f"Total parameters: {model_info.get('total_parameters', 'N/A'):,}")
    info_print(f"Context length: {model_info['ttm_specific']['context_length']}")
    info_print(f"Forecast length: {model_info['ttm_specific']['forecast_length']}")

    return model


def example_lora_compatibility():
    """Example 2: Demonstrate LoRA compatibility across different architectures."""
    info_print("=" * 60)
    info_print("Example 2: LoRA compatibility demonstration")
    info_print("=" * 60)

    # LoRA configuration
    lora_config = LoRAConfig(
        enabled=True,
        rank=16,
        alpha=32,
        dropout=0.1,
        auto_detect_modules=True,  # Let the framework detect appropriate modules
    )

    info_print("Testing LoRA support across different model architectures:")
    
    # Test 1: TTM (MLP-Mixer based - should NOT support LoRA)
    info_print("\n1. Testing TTM (MLP-Mixer based model):")
    ttm_config = TTMConfig(
        model_path="ibm-granite/granite-timeseries-ttm-r2",
        context_length=512,
        forecast_length=96,
        use_cpu=True,
    )
    
    ttm_model = TTMForecaster(ttm_config, lora_config=lora_config)
    info_print(f"   TTM supports LoRA: {ttm_model.supports_lora()}")
    info_print("   LoRA was automatically disabled for TTM (MLP-Mixer architecture)")
    
    # Test 2: TSMixer (MLP-based - should NOT support LoRA)
    info_print("\n2. Testing TSMixer (MLP-based model):")
    try:
        from src.models.tsmixer import TSMixerForecaster, TSMixerConfig
        
        tsmixer_config = TSMixerConfig(
            context_length=512,
            forecast_length=96,
            d_model=128,
            n_blocks=4,
            use_cpu=True,
        )
        
        tsmixer_model = TSMixerForecaster(tsmixer_config, lora_config=lora_config)
        info_print(f"   TSMixer supports LoRA: {tsmixer_model.supports_lora()}")
        info_print("   LoRA was automatically disabled for TSMixer")
        
    except ImportError:
        info_print("   TSMixer not available - this is expected in current codebase")
    
    info_print("\n3. Testing automatic LoRA module detection:")
    # Since TTM doesn't support LoRA, we'd need a transformer model for this test
    # This would work if we had a Chronos or TimeGPT implementation
    info_print("   Auto-detection only works for transformer-based models")
    info_print("   TTM and TSMixer both use MLP architectures without attention")
    
    info_print("\nLoRA compatibility test complete!")
    info_print("Key benefits:")
    info_print("- Transformer models (Chronos, TimeGPT) support LoRA")  
    info_print("- MLP models (TTM, TSMixer) gracefully disable LoRA")
    info_print("- Framework automatically handles architecture compatibility")
    
    return ttm_model


def example_distributed_configuration():
    """Example 3: Distributed training configuration."""
    info_print("=" * 60)
    info_print("Example 3: Distributed training setup")
    info_print("=" * 60)

    # Get GPU information
    gpu_info = GPUManager.get_gpu_info()
    info_print(f"GPU available: {gpu_info['gpu_available']}")
    info_print(f"GPU count: {gpu_info['gpu_count']}")

    if gpu_info["gpu_count"] > 1:
        # Configure for multi-GPU training
        distributed_config = DistributedConfig(
            enabled=True,
            strategy="ddp",
            world_size=gpu_info["gpu_count"],
        )
        use_cpu = False
        info_print(f"Configured for {gpu_info['gpu_count']} GPUs with DDP")
    else:
        distributed_config = DistributedConfig(enabled=False)
        use_cpu = not gpu_info["gpu_available"]
        info_print("Configured for single device training")

    # Create model configuration
    config = TTMConfig(
        model_path="ibm-granite/granite-timeseries-ttm-r2",
        context_length=512,
        forecast_length=96,
        batch_size=64,
        learning_rate=1e-4,
        num_epochs=1,
        use_cpu=use_cpu,
        fp16=gpu_info["gpu_available"],  # Use fp16 if GPU available
    )
    info_print("\n Config: \n", config, "\n")
    # Create model with distributed configuration
    model = TTMForecaster(config, distributed_config=distributed_config)

    model_info = model.get_model_info()
    info_print("Distributed model created successfully!")
    info_print(f"Distributed training: {model_info['distributed_enabled']}")

    return model


def example_training_simulation():
    """Example 4: Simulate training workflow (without actual data)."""
    info_print("=" * 60)
    info_print("Example 4: Training workflow simulation")
    info_print("=" * 60)

    # Create a comprehensive configuration
    config = TTMConfig(
        model_path="ibm-granite/granite-timeseries-ttm-r2",
        context_length=512,
        forecast_length=96,
        batch_size=32,
        learning_rate=1e-4,
        num_epochs=1,  # Small for demo
        eval_strategy="steps",
        eval_steps=100,
        save_steps=200,
        early_stopping_patience=5,
        use_cpu=True,  # Set based on your hardware
        freeze_backbone=False,  # Explicitly ensure training is enabled
        fit_strategy="from_scratch",  # Use from_scratch to force full training
    )

    # LoRA for efficient training
    lora_config = LoRAConfig(
        enabled=True,
        rank=8,  # Smaller rank for demo
        alpha=16,
    )

    model = TTMForecaster(config, lora_config=lora_config)

    # Simulate training preparation
    info_print("Simulating training preparation...")
    info_print(f"Model initialized: {model.model is not None}")
    info_print(f"Configuration: {config.model_type}")

    dataset_name = "kaggle_brisT1D"
    # Save model configuration (useful for reproducibility)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"./trained_models/artifacts/_tsfm_testing/{timestamp}_{dataset_name}/"
    os.makedirs(output_dir, exist_ok=True)

    config_path = os.path.join(output_dir, "model_config.json")
    with open(config_path, "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    info_print(f"Configuration saved to: {config_path}")

    # In a real scenario, you would call:
    results = model.fit(train_data=dataset_name, output_dir=output_dir)
    print(results)
    info_print("Training simulation complete!")
    return model


def example_configuration_from_yaml():
    """Example 5: Load configuration from YAML (conceptual)."""
    info_print("=" * 60)
    info_print("Example 5: Configuration from external files")
    info_print("=" * 60)

    # This shows how you could load from external configuration
    # In practice, you'd load from configs/models/ttm/fine_tune.yaml

    example_config = {
        "model_type": "ttm",
        "model_path": "ibm-granite/granite-timeseries-ttm-r2",
        "context_length": 512,
        "forecast_length": 96,
        "batch_size": 64,
        "learning_rate": 1e-4,
        "num_epochs": 10,
        "fit_strategy": "fine_tune",
        "fp16": True,
        "early_stopping_patience": 5,
    }

    info_print("Example configuration:")
    info_print(json.dumps(example_config, indent=2))

    # Create config from dictionary
    config = TTMConfig(**example_config)
    model = TTMForecaster(config)

    info_print("Model created from external configuration!")
    return model


def main():
    """Run all examples."""
    parser = argparse.ArgumentParser(description="Test the base TSFM framework")
    parser.add_argument(
        "--example",
        type=int,
        choices=[1, 2, 3, 4, 5],
        help="Run specific example (1-5), or run all if not specified",
    )

    args = parser.parse_args()

    examples = {
        1: example_basic_usage,
        2: example_lora_compatibility,
        3: example_distributed_configuration,
        4: example_training_simulation,
        5: example_configuration_from_yaml,
    }

    try:
        if args.example:
            # Run specific example
            examples[args.example]()
        else:
            # Run all examples
            for i, example_func in examples.items():
                try:
                    example_func()
                    info_print("")  # Add spacing
                except Exception as e:
                    error_print(f"Example {i} failed: {str(e)}")
                    continue

        info_print("=" * 60)
        info_print("Framework testing complete!")
        info_print("=" * 60)

        info_print("Next steps:")
        info_print("1. Adapt the data preparation methods in TTMForecaster")
        info_print("2. Test with real data: model.fit('kaggle_brisT1D')")
        info_print("3. Implement other model types (Chronos, TimeGPT, etc.)")
        info_print("4. Add experiment management and model registry")

    except KeyboardInterrupt:
        info_print("Testing interrupted by user")
    except Exception as e:
        error_print(f"Framework testing failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
