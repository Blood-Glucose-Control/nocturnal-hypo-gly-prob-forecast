#!/usr/bin/env python3
"""
Distributed Training Setup Validator

This script checks if your environment is properly configured for multi-GPU
distributed training with the TSFM framework. It validates:

- GPU availability and count
- Distributed training environment variables (RANK, LOCAL_RANK, etc.)
- torchrun installation and availability
- Provides specific commands for enabling distributed training

USAGE:
======

Check current setup:
    python scripts/examples/check_gpu_setup.py

After running torchrun:
    torchrun --nproc_per_node=2 scripts/examples/check_gpu_setup.py
    (Should show all environment variables as set)

PURPOSE:
========

Use this script when:
- Setting up multi-GPU training for the first time
- Debugging "Why isn't my distributed training working?"
- Verifying torchrun installation
- Getting exact commands for your hardware setup

For detailed hardware specifications, use show_hardware_info.py instead.
"""

import os
import sys

# Add src to path
sys.path.append('src')

from src.models.base import GPUManager
from src.utils.logging_helper import info_print

def check_multi_gpu_setup():
    """
    Validate distributed training environment setup.
    
    This function performs a comprehensive check of the system's readiness
    for multi-GPU distributed training, including:
    
    - Hardware detection (GPU count and availability)
    - Environment variable validation (RANK, LOCAL_RANK, WORLD_SIZE, etc.)
    - torchrun installation verification
    - Specific guidance for enabling distributed training
    
    Returns actionable guidance based on current system state.
    """
    
    info_print("🖥️  Multi-GPU Setup Check")
    info_print("=" * 40)
    
    # Get GPU info
    gpu_info = GPUManager.get_gpu_info()
    info_print(f"GPUs Available: {gpu_info['gpu_available']}")
    info_print(f"GPU Count: {gpu_info['gpu_count']}")
    
    if gpu_info['gpu_count'] > 1:
        info_print(f"✅ {gpu_info['gpu_count']} GPUs detected - multi-GPU training possible!")
        
        # Check environment
        env_vars = {
            'RANK': os.environ.get('RANK'),
            'LOCAL_RANK': os.environ.get('LOCAL_RANK'), 
            'WORLD_SIZE': os.environ.get('WORLD_SIZE'),
            'MASTER_ADDR': os.environ.get('MASTER_ADDR'),
            'MASTER_PORT': os.environ.get('MASTER_PORT')
        }
        
        info_print("\n🌐 Distributed Environment Check:")
        for var, value in env_vars.items():
            status = "✅" if value else "❌"
            info_print(f"   {status} {var}: {value or 'Not set'}")
        
        all_set = all(env_vars.values())
        if all_set:
            info_print("\n🎉 All distributed environment variables are set!")
            info_print("You can run distributed training directly.")
        else:
            info_print("\n⚠️  Distributed environment not configured.")
            info_print("\nTo enable multi-GPU training:")
            info_print(f"1. Use torchrun (easiest):")
            info_print(f"   torchrun --nproc_per_node={gpu_info['gpu_count']} scripts/examples/example_distributed_ttm.py")
            info_print("\n2. Use the bash script:")
            info_print("   bash scripts/examples/run_distributed_ttm.sh")
            info_print("\n3. Check if torchrun is available:")
            
            # Check if torchrun exists
            import subprocess
            try:
                result = subprocess.run(['torchrun', '--help'], capture_output=True, text=True)
                if result.returncode == 0:
                    info_print("   ✅ torchrun is available!")
                else:
                    info_print("   ❌ torchrun not found - check PyTorch installation")
            except FileNotFoundError:
                info_print("   ❌ torchrun not found in PATH")
                
    else:
        info_print("📱 Single GPU/CPU detected - distributed training not applicable")
    
    info_print("\n💡 Current Status:")
    if gpu_info['gpu_count'] > 1 and not os.environ.get('RANK'):
        info_print("   Running example_distributed_ttm.py now will use single-GPU fallback")
        info_print("   Use torchrun or bash script for multi-GPU training")
    else:
        info_print("   Ready for training!")
    
    info_print("\n🔗 Related Tools:")
    info_print("   For detailed hardware specs: python scripts/examples/print_gpu_info.py")

if __name__ == "__main__":
    check_multi_gpu_setup()