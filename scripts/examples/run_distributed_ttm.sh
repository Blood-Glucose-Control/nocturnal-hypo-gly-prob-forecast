#!/bin/bash
"""
Script to run distributed TTM training with proper process coordination.

This script automatically detects your GPU setup and launches training
with the optimal configuration.

How to run:
  # Make executable (first time only):
  chmod +x run_distributed_ttm.sh

  # Run with automatic GPU detection:
  ./run_distributed_ttm.sh
  # OR
  bash run_distributed_ttm.sh

  # Manually specify number of GPUs:
  NUM_GPUS=4 bash run_distributed_ttm.sh

  # Use specific GPUs only:
  CUDA_VISIBLE_DEVICES=0,2 bash run_distributed_ttm.sh

What this does:
  - Detects available GPUs automatically
  - Uses torchrun for proper multi-process coordination
  - Falls back to single GPU if only one available
  - Handles all the distributed training environment setup

Requirements:
  - PyTorch with distributed support
  - NVIDIA GPUs for multi-GPU (CUDA)
  - torchrun (included with PyTorch 1.9+)
"""

# Set number of GPUs to use
if [ -z "$NUM_GPUS" ]; then
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
fi

echo "🔍 Detected $NUM_GPUS GPUs"

if [ "$NUM_GPUS" -gt 1 ]; then
    echo "🚀 Running distributed training on $NUM_GPUS GPUs"
    echo "Using torchrun for proper process coordination..."

    # Method 1: Using torchrun (recommended - handles process coordination)
    torchrun \
        --nproc_per_node=$NUM_GPUS \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr="127.0.0.1" \
        --master_port=29500 \
        scripts/examples/example_distributed_ttm.py

else
    echo "📱 Running single GPU/CPU training"
    echo "No distributed setup needed..."
    python scripts/examples/example_distributed_ttm.py
fi

echo "✅ Distributed training complete!"
