# Example Scripts Guide

This guide explains the example scripts provided in `scripts/examples/` to help you get started with the TSFM (Time Series Foundation Model) framework. Each script serves a specific purpose in the development and deployment workflow.

## Overview of Example Scripts

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `show_hardware_info.py` | Hardware diagnostics | Environment setup, hardware troubleshooting |
| `check_distributed_training_setup.py` | Distributed training validation | Multi-GPU setup, distributed training issues |
| `test_base_framework.py` | Framework functionality testing | Development, integration validation |
| `example_distributed_ttm.py` | TTM distributed training demo | Production multi-GPU training |

---

## Hardware and Environment Scripts

### `show_hardware_info.py` - Hardware Information Display

**Purpose**: Provides comprehensive information about your system's hardware capabilities for deep learning.

**Usage**:
```bash
python scripts/examples/show_hardware_info.py
```

**What it shows**:
- GPU specifications (memory, compute capability, multiprocessors)
- CUDA and PyTorch version compatibility
- Current GPU memory usage and availability
- CPU thread count and distributed training support
- Framework compatibility assessment

**When to use**:
- Setting up a new development environment
- Diagnosing hardware-related training issues
- Checking GPU memory before large model training
- Verifying CUDA/PyTorch installation
- Planning model configurations based on available resources
- Troubleshooting "CUDA out of memory" errors

**Example output**:
```
🖥️  GPU Information Summary
==================================================
GPU Available: True
GPU Count: 2
Current GPU: 0

🔧 PyTorch GPU Details:
CUDA Available: True
CUDA Version: 11.8
PyTorch Version: 2.0.1

GPU 0: NVIDIA GeForce RTX 4090
  Memory: 24.0 GB
  Compute Capability: 8.9
  Multiprocessors: 128
  Memory Allocated: 0.0 GB
  Memory Reserved: 0.0 GB

🌐 Distributed Training Support:
Torch Distributed Available: True
NCCL Available: True
Gloo Available: True
```

---

### `check_distributed_training_setup.py` - Distributed Training Validator

**Purpose**: Validates if your environment is properly configured for multi-GPU distributed training.

**Usage**:
```bash
# Check current setup
python scripts/examples/check_distributed_training_setup.py

# Check setup after launching with torchrun
torchrun --nproc_per_node=2 scripts/examples/check_distributed_training_setup.py
```

**What it validates**:
- GPU availability and count
- Distributed training environment variables (RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT)
- torchrun installation and availability
- Provides specific commands for enabling distributed training

**When to use**:
- Setting up multi-GPU training for the first time
- Debugging "Why isn't my distributed training working?"
- Verifying torchrun installation
- Getting exact commands for your hardware setup

**Example output**:
```
🖥️  Multi-GPU Setup Check
========================================
GPUs Available: True
GPU Count: 2
✅ 2 GPUs detected - multi-GPU training possible!

🌐 Distributed Environment Check:
   ❌ RANK: Not set
   ❌ LOCAL_RANK: Not set
   ❌ WORLD_SIZE: Not set
   ❌ MASTER_ADDR: Not set
   ❌ MASTER_PORT: Not set

⚠️  Distributed environment not configured.

To enable multi-GPU training:
1. Use torchrun (easiest):
   torchrun --nproc_per_node=2 scripts/examples/example_distributed_ttm.py

2. Use the bash script:
   bash scripts/examples/run_distributed_ttm.sh

3. Check if torchrun is available:
   ✅ torchrun is available!
```

---

## Framework Testing and Development

### `test_base_framework.py` - Framework Functionality Testing

**Purpose**: Comprehensive test suite for the unified TSFM framework, validating all core functionality.

**Usage**:
```bash
# Run all examples
python scripts/examples/test_base_framework.py

# Run specific example
python scripts/examples/test_base_framework.py --example 1

# Run with distributed training (Examples 3 & 4)
torchrun --nproc_per_node=2 scripts/examples/test_base_framework.py --example 3
torchrun --nproc_per_node=4 scripts/examples/test_base_framework.py --example 4
```

**Examples included**:

#### Example 1: Basic Model Creation
- **Purpose**: Verify basic framework instantiation
- **What it tests**: TTMConfig creation, TTMForecaster initialization, model information display
- **Expected outcome**: Model created successfully, metadata displayed
- **Use case**: Development environment validation

#### Example 2: LoRA Compatibility Testing
- **Purpose**: Demonstrate LoRA (Low-Rank Adaptation) compatibility across architectures
- **What it tests**: Automatic LoRA detection, graceful degradation for incompatible models
- **Expected outcome**: TTM automatically disables LoRA (MLP architecture), no crashes
- **Use case**: Understanding fine-tuning capabilities

#### Example 3: Distributed Training Configuration
- **Purpose**: Test distributed training setup and configuration
- **What it tests**: Hardware detection, distributed configuration, model instantiation
- **Expected outcome**: Appropriate configuration for available hardware
- **Use case**: Multi-GPU setup validation

#### Example 4: Complete Training Workflow
- **Purpose**: End-to-end training with real diabetes data
- **What it tests**: Data loading, training execution, model persistence, experiment tracking
- **Expected outcome**: Model trains successfully, metrics collected, files saved
- **Use case**: Production workflow validation
- **Note**: This performs actual training and may take several minutes

#### Example 5: External Configuration Management
- **Purpose**: Configuration loading from external sources
- **What it tests**: Dictionary-based configuration, model creation from structured data
- **Expected outcome**: Model created from external configuration
- **Use case**: YAML/JSON integration patterns

**When to use**:
- Validating new framework installations
- Testing after framework modifications
- Ensuring compatibility across different environments
- Learning framework capabilities and APIs

**Hardware requirements**:
- **Minimum**: CPU-only execution supported, 8GB RAM recommended
- **Recommended**: 1+ NVIDIA GPU with 8GB+ VRAM, Multiple GPUs for distributed examples

---

## Production Training Scripts

### `example_distributed_ttm.py` - TTM Distributed Training Demo

**Purpose**: Production-ready example of distributed TTM training with comprehensive setup guidance.

**Usage**:
```bash
# Single GPU/CPU
python scripts/examples/example_distributed_ttm.py

# Multi-GPU distributed training
torchrun --nproc_per_node=2 scripts/examples/example_distributed_ttm.py

# Automated GPU detection
bash scripts/examples/run_distributed_ttm.sh
```

**What it demonstrates**:
- Automatic GPU detection and configuration
- DistributedConfig setup for different scenarios
- Identical training code for single/multi-GPU
- transformers.Trainer handling all distributed complexity

**When to use**:
- Production multi-GPU training workflows
- Learning distributed training best practices
- Testing distributed training before deploying larger experiments

---

## Workflow Recommendations

### For New Users

1. **Start with hardware validation**:
   ```bash
   python scripts/examples/show_hardware_info.py
   ```

2. **Test framework functionality**:
   ```bash
   python scripts/examples/test_base_framework.py --example 1
   ```

3. **If you have multiple GPUs, check distributed setup**:
   ```bash
   python scripts/examples/check_distributed_training_setup.py
   ```

### For Multi-GPU Setup

1. **Validate hardware**:
   ```bash
   python scripts/examples/show_hardware_info.py
   ```

2. **Check distributed environment**:
   ```bash
   python scripts/examples/check_distributed_training_setup.py
   ```

3. **Test distributed configuration**:
   ```bash
   torchrun --nproc_per_node=N scripts/examples/test_base_framework.py --example 3
   ```

4. **Run distributed training**:
   ```bash
   torchrun --nproc_per_node=N scripts/examples/example_distributed_ttm.py
   ```

### For Development and Debugging

1. **Framework validation**:
   ```bash
   python scripts/examples/test_base_framework.py
   ```

2. **Hardware troubleshooting**:
   ```bash
   python scripts/examples/show_hardware_info.py
   ```

3. **Distributed training issues**:
   ```bash
   python scripts/examples/check_distributed_training_setup.py
   ```

---

## Common Issues and Solutions

### "CUDA out of memory"
1. Check available memory: `python scripts/examples/show_hardware_info.py`
2. Reduce batch size in model configurations
3. Use gradient checkpointing or mixed precision training

### "No module named 'src'"
- Ensure you're running from the repository root directory
- Check that `src/` directory exists and contains `__init__.py`

### "torchrun not found"
1. Check torchrun installation: `torchrun --help`
2. Install PyTorch with distributed support
3. Verify PATH contains PyTorch installation

### Distributed training hangs
1. Validate environment: `python scripts/examples/check_distributed_training_setup.py`
2. Use torchrun instead of direct python execution
3. Check GPU visibility: `nvidia-smi`

### Multiple GPU detection fails
1. Check CUDA drivers and GPU visibility
2. Verify CUDA availability: `python -c "import torch; print(torch.cuda.is_available())"`
3. Validate PyTorch installation with CUDA support

---

## Integration with Other Tools

These example scripts integrate with other parts of the TSFM framework:

- **Model Registry**: Training results can be automatically registered
- **Experiment Management**: Configuration and metrics are saved for tracking
- **Data Pipeline**: Seamless integration with diabetes dataset loaders
- **Configuration System**: Support for YAML/JSON configuration files

For more detailed information about specific components, see:
- [Base Model Framework](../base_model_framework_README.md)
- [LoRA Compatibility](../base_model_lora_compatibility.md)
- [TTM YAML Configuration Guide](../ttm_yaml_config_guide.md)
