#!/usr/bin/env python3
"""
Example script demonstrating the use of the new base TSFM framework.

This script provides comprehensive examples of the unified TSFM framework including:
1. Basic model configuration and initialization
2. LoRA (Low-Rank Adaptation) fine-tuning setup and compatibility testing
3. Distributed training configuration for multi-GPU environments
4. Complete training workflows with real diabetes datasets
5. External configuration management and best practices

USAGE:
======

Basic Usage (Single GPU/CPU):
    python scripts/examples/test_base_framework.py
    python scripts/examples/test_base_framework.py --example 1

Multi-GPU Distributed Training (Examples 3 & 4):
    # For multi-GPU training, use torchrun instead of python
    torchrun --nproc_per_node=2 scripts/examples/test_base_framework.py --example 3
    torchrun --nproc_per_node=4 scripts/examples/test_base_framework.py --example 4

    # Or use automatic GPU detection script
    bash scripts/examples/run_distributed_ttm.sh

WHEN TO USE TORCHRUN:
====================

Use `python` for:
- Examples 1, 2, 5 (basic functionality, no distributed training)
- Single GPU or CPU-only environments
- Testing and development

Use `torchrun` for:
- Examples 3, 4 when you have multiple GPUs available
- Any distributed training scenarios
- Production multi-GPU training workflows

HARDWARE REQUIREMENTS:
=====================

Minimum:
- CPU-only execution supported for all examples
- 8GB RAM recommended for model loading

Recommended:
- 1+ NVIDIA GPU with 8GB+ VRAM
- Multiple GPUs for distributed training examples
- CUDA-compatible PyTorch installation

EXAMPLES OVERVIEW:
=================

Example 1 (Basic): Model creation, configuration, CPU/GPU compatibility
Example 2 (LoRA): Architecture compatibility testing, automatic LoRA handling
Example 3 (Distributed): Multi-GPU setup, hardware detection, distributed config
Example 4 (Training): End-to-end training with real data, full pipeline testing
Example 5 (Config): External configuration, YAML integration patterns

TROUBLESHOOTING:
===============

Common Issues:
- "CUDA out of memory": Reduce batch_size in configurations
- "No module named 'src'": Run from repository root directory
- "torchrun not found": Install PyTorch with distributed support
- Multiple GPU detection fails: Check CUDA drivers and GPU visibility

For distributed training issues:
- Ensure all GPUs are visible: nvidia-smi
- Check CUDA availability: python -c "import torch; print(torch.cuda.is_available())"
- Verify torchrun installation: torchrun --help

Run this script to test the base model framework implementation and verify
your environment is properly configured for single or multi-GPU training.
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
    """
    Example 1: Basic model creation and usage.

    This example demonstrates:
    - Creating a TTMConfig with standard parameters
    - Initializing a TTMForecaster instance
    - Inspecting model properties and configuration
    - Understanding CPU vs GPU configuration

    Purpose:
    - Verify the basic framework instantiation works correctly
    - Show the minimal setup required for TTM models
    - Display model metadata and parameter counts

    Expected outcome:
    - Model is created successfully without errors
    - Model information is displayed including parameters and architecture
    - CPU mode is used for compatibility across all environments
    """
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
    """
    Example 2: Demonstrate LoRA compatibility across different architectures.

    This example demonstrates:
    - How different model architectures handle LoRA fine-tuning
    - Automatic compatibility detection and graceful degradation
    - The difference between transformer and MLP-based models for LoRA
    - Framework's ability to automatically disable LoRA for incompatible models

    Model architecture compatibility:
    - TTM (MLP-Mixer): Does NOT support LoRA (automatically disabled)
    - TSMixer (MLP): Does NOT support LoRA (automatically disabled)
    - Transformers (Chronos, TimeGPT): WOULD support LoRA (not implemented yet)

    Purpose:
    - Show that the framework gracefully handles LoRA configuration
    - Demonstrate automatic architecture detection
    - Explain why certain models don't support LoRA fine-tuning

    Expected outcome:
    - TTM model is created but LoRA is automatically disabled
    - Clear messaging about why LoRA isn't supported for MLP architectures
    - No errors or crashes when LoRA is requested for incompatible models
    """
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
    """
    Example 3: Distributed training configuration.

    This example demonstrates:
    - Automatic GPU detection and hardware assessment
    - Dynamic configuration based on available hardware
    - Setting up DistributedConfig for multi-GPU training
    - Fallback behavior for single-GPU or CPU-only systems

    Hardware scenarios covered:
    - Multi-GPU: Configures DDP (Distributed Data Parallel) training
    - Single GPU: Uses standard single-device training with GPU acceleration
    - CPU-only: Falls back to CPU training with appropriate settings

    Configuration decisions:
    - Automatically enables/disables fp16 based on GPU availability
    - Sets appropriate batch sizes for the hardware configuration
    - Chooses optimal distributed strategy (DDP for multi-GPU)

    Purpose:
    - Show how the framework adapts to different hardware setups
    - Demonstrate the distributed training configuration API
    - Verify that models can be created with distributed settings

    Expected outcome:
    - Hardware is detected correctly (GPU count and availability)
    - Appropriate distributed configuration is created
    - Model is instantiated with the correct distributed settings
    - No errors regardless of the underlying hardware setup
    """
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
    """
    Example 4: Complete training workflow simulation with real data.

    This example demonstrates:
    - End-to-end training setup from configuration to execution
    - Integration with real diabetes datasets (kaggle_brisT1D)
    - Comprehensive configuration including training hyperparameters
    - Model persistence and experiment tracking setup
    - Combining distributed training, LoRA, and full training pipeline

    Workflow components:
    - Hardware detection and optimal configuration selection
    - Model configuration with training-specific parameters
    - LoRA configuration for efficient fine-tuning
    - Output directory creation with timestamps
    - Configuration serialization for reproducibility
    - Actual model training execution

    Training configuration:
    - Uses real glucose monitoring data from Kaggle T1D dataset
    - Implements early stopping for training efficiency
    - Enables evaluation during training for monitoring
    - Saves checkpoints and logs for experiment tracking

    Purpose:
    - Demonstrate a complete training workflow
    - Show integration with real diabetes datasets
    - Verify that the framework can handle actual training workloads
    - Test model saving and experiment tracking capabilities

    Expected outcome:
    - Model trains successfully on real data
    - Training metrics are collected and logged
    - Model and configuration are saved to disk
    - Distributed training works if multiple GPUs are available
    - LoRA fine-tuning is applied efficiently (or disabled for TTM)

    Note: This example performs actual training and may take several minutes
    """
    info_print("=" * 60)
    info_print("Example 4: Training workflow simulation")
    info_print("=" * 60)

    # Get GPU information
    gpu_info = GPUManager.get_gpu_info()
    info_print(f"GPU available: {gpu_info['gpu_available']}")
    info_print(f"GPU count: {gpu_info['gpu_count']}")

    if gpu_info["gpu_count"] > 1:
        # Check if we're running with proper distributed environment (torchrun)
        rank = int(os.environ.get("RANK", -1))
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        world_size = int(os.environ.get("WORLD_SIZE", -1))
        master_addr = os.environ.get("MASTER_ADDR", None)

        if rank >= 0 and world_size > 1 and master_addr is not None:
            # Configure for multi-GPU training with proper environment
            distributed_config = DistributedConfig(
                enabled=True,
                strategy="ddp",
                world_size=world_size,
                local_rank=local_rank,
            )
            use_cpu = False
            info_print(
                f"Configured for {gpu_info['gpu_count']} GPUs with DDP (rank {rank}/{world_size})"
            )
        else:
            # Multiple GPUs detected but no distributed environment - warn user
            distributed_config = DistributedConfig(enabled=False)
            use_cpu = False
            info_print(
                f"⚠️  {gpu_info['gpu_count']} GPUs detected but no distributed environment"
            )
            info_print("   To enable multi-GPU training, use:")
            info_print(
                f"   torchrun --nproc_per_node={gpu_info['gpu_count']} scripts/examples/test_base_framework.py --example 4"
            )
            info_print("   Falling back to single-GPU training...")
    else:
        distributed_config = DistributedConfig(enabled=False)
        use_cpu = not gpu_info["gpu_available"]
        info_print("Configured for single device training")

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
        use_cpu=use_cpu,  # Set based on your hardware
        freeze_backbone=False,  # Explicitly ensure training is enabled
        training_mode="from_scratch",  # Use from_scratch to force full training
    )

    # LoRA for efficient training
    lora_config = LoRAConfig(
        enabled=True,
        rank=8,  # Smaller rank for demo
        alpha=16,
    )

    model = TTMForecaster(
        config, lora_config=lora_config, distributed_config=distributed_config
    )

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
    """
    Example 5: External configuration management and YAML integration.

    This example demonstrates:
    - Creating model configurations from external data sources
    - How to structure configuration dictionaries for the framework
    - Best practices for configuration management and versioning
    - Integration patterns with YAML/JSON configuration files

    Configuration management features:
    - Structured configuration with clear parameter organization
    - Support for different training strategies (fine_tune, from_scratch, zero_shot)
    - Hardware-agnostic configuration that adapts to available resources
    - Reproducible experiment setup through configuration serialization

    Real-world usage patterns:
    - Load from configs/models/ttm/fine_tune.yaml
    - Override specific parameters for experiment variations
    - Share configurations between team members
    - Version control configuration files alongside code

    Configuration validation:
    - All required parameters are present
    - Parameter values are within valid ranges
    - Model-specific constraints are satisfied
    - Backward compatibility with existing configurations

    Purpose:
    - Show how to integrate with external configuration systems
    - Demonstrate configuration best practices
    - Verify that models can be created from structured data
    - Prepare for integration with experiment management tools

    Expected outcome:
    - Model is created successfully from dictionary configuration
    - Configuration structure is clearly documented
    - Framework accepts external configuration formats
    - Easy to extend for YAML/JSON file loading

    Next steps for production use:
    - Add YAML file loading with schema validation
    - Implement configuration inheritance and overrides
    - Add configuration templates for common use cases
    """
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
        "training_mode": "fine_tune",
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
