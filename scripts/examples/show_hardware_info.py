#!/usr/bin/env python3
"""
Hardware Information Display Tool

This script provides comprehensive information about your system's hardware
capabilities for deep learning and time series forecasting. It shows:

- GPU specifications (memory, compute capability, multiprocessors)
- CUDA and PyTorch version compatibility
- Current GPU memory usage and availability
- CPU thread count and distributed training support
- Framework compatibility assessment

USAGE:
======

Display all hardware info:
    python scripts/examples/print_gpu_info.py

PURPOSE:
========

Use this script when:
- Setting up a new development environment
- Diagnosing hardware-related training issues
- Checking GPU memory before large model training
- Verifying CUDA/PyTorch installation
- Planning model configurations based on available resources
- Troubleshooting "CUDA out of memory" errors

OUTPUT INCLUDES:
===============

- GPU count and individual specifications
- Memory capacity and current usage per GPU
- CUDA compute capability for model compatibility
- PyTorch and CUDA version information
- Distributed training backend availability (NCCL, Gloo)

For distributed training setup validation, use check_distributed_training_setup.py instead.
"""

from src.models.base import GPUManager
from src.utils.logging_helper import info_print
import torch


def print_gpu_info():
    """
    Display comprehensive hardware information for deep learning.

    This function provides detailed information about the system's hardware
    capabilities, including:

    - GPU specifications (memory, compute capability, architecture)
    - PyTorch and CUDA version compatibility
    - Current memory usage and availability
    - Distributed training backend support
    - CPU information for fallback scenarios

    Useful for hardware diagnostics, environment setup validation,
    and planning model configurations based on available resources.
    """

    info_print("🖥️  GPU Information Summary")
    info_print("=" * 50)

    # Use our framework's GPU manager
    gpu_info = GPUManager.get_gpu_info()

    info_print(f"GPU Available: {gpu_info['gpu_available']}")
    info_print(f"GPU Count: {gpu_info['gpu_count']}")

    if gpu_info["gpu_available"]:
        info_print(f"Current GPU: {gpu_info['current_device']}")

        # Additional PyTorch GPU info
        info_print("\n🔧 PyTorch GPU Details:")
        info_print(f"CUDA Available: {torch.cuda.is_available()}")
        info_print(f"CUDA Version: {torch.version.cuda}")
        info_print(f"PyTorch Version: {torch.__version__}")

        if torch.cuda.is_available():
            info_print(f"CUDA Device Count: {torch.cuda.device_count()}")

            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / (1024**3)
                info_print(f"\nGPU {i}: {props.name}")
                info_print(f"  Memory: {memory_gb:.1f} GB")
                info_print(f"  Compute Capability: {props.major}.{props.minor}")
                info_print(f"  Multiprocessors: {props.multi_processor_count}")

                # Current memory usage
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated(i) / (1024**3)
                    reserved = torch.cuda.memory_reserved(i) / (1024**3)
                    info_print(f"  Memory Allocated: {allocated:.1f} GB")
                    info_print(f"  Memory Reserved: {reserved:.1f} GB")

        # Distributed training info
        info_print("\n🌐 Distributed Training Support:")
        info_print(f"Torch Distributed Available: {torch.distributed.is_available()}")
        if torch.distributed.is_available():
            info_print(f"NCCL Available: {torch.distributed.is_nccl_available()}")
            info_print(f"Gloo Available: {torch.distributed.is_gloo_available()}")
    else:
        info_print("No GPU available - using CPU")
        info_print(f"CPU Count: {torch.get_num_threads()} threads")

    info_print("\n🔗 Related Tools:")
    info_print(
        "   For distributed training setup: python scripts/examples/check_distributed_training_setup.py"
    )
    info_print(
        "   For framework testing: python scripts/examples/test_base_framework.py"
    )


if __name__ == "__main__":
    print_gpu_info()
