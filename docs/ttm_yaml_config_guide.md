# TTM Training with YAML Configuration Guide

## Overview
The TTM training system now supports YAML configuration files for systematic experiment tracking and reproducibility.

## Usage

### 1. Basic Usage (Default Parameters)
```bash
sbatch scripts/watgpu_slurm/ttm_finetune.sh
```

### 2. With Configuration File
```bash
Or with a specific config:
```bash
sbatch scripts/watgpu_slurm/ttm_finetune.sh models/configs/ttm_baseline_config.yaml
```
```

### 3. Available Configurations
- `ttm_baseline_config.yaml` - Standard baseline configuration
- `ttm_high_performance_config.yaml` - Optimized for L40S with larger batches
- `ttm_quick_test_config.yaml` - Fast testing with minimal epochs

## Creating Custom Configurations

Copy an existing config and modify parameters:
```bash
cp models/configs/ttm_baseline_config.yaml models/configs/my_experiment.yaml
# Edit my_experiment.yaml with your parameters

Run your experiment:
```bash
sbatch scripts/watgpu_slurm/ttm_finetune.sh models/configs/my_experiment.yaml
```
```

## Key Configuration Sections

### Experiment Metadata
```yaml
experiment:
  name: "my_experiment"
  description: "Description of what this experiment tests"
  tags: ["tag1", "tag2"]
  version: "1.0"
```

### Training Parameters
```yaml
training:
  batch_size: 128        # Adjust based on GPU memory
  learning_rate: 0.001   # null for auto-detection
  num_epochs: 10
  mixed_precision: true  # Enable for faster training
```

### Hardware Optimization
```yaml
hardware:
  expected_gpu_type: "L40S"
  expected_memory_gb: 48
  mixed_precision: true
```

## Output Organization

Each run creates:
```
results/runs/ttm_finetune/run_TIMESTAMP_jobID/
├── experiment_config.yaml     # Copy of your config
├── run_info.txt              # SLURM and hardware info
├── training.log              # Training output
├── gpu_monitoring.log        # GPU utilization
└── gpu_utilization.log       # Periodic GPU stats
```

## Model Registry

All runs are automatically tracked in a CSV registry:
```bash
python results/runs/view_registry.py
```

The registry includes:
- Experiment metadata and configuration
- Hardware specifications and SLURM settings
- Training results and performance metrics
- File paths for easy access to logs and outputs

## Performance Monitoring

Each run now captures:
- Detailed GPU utilization throughout training
- Memory usage patterns
- SLURM resource allocation vs actual usage
- Training performance metrics

## Best Practices

1. **Start with quick tests**: Use `ttm_quick_test_config.yaml` for initial experiments
2. **Optimize batch size**: Gradually increase until you hit memory limits
3. **Enable mixed precision**: Set `mixed_precision: true` for L40S
4. **Use descriptive names**: Clear experiment names help with organization
5. **Tag experiments**: Use tags to group related experiments

## Example Workflow

```bash
# 1. Quick test to verify setup
sbatch scripts/watgpu_slurm/ttm_finetune.sh models/configs/ttm_quick_test_config.yaml

# 2. Check the results
python results/runs/view_registry.py

# 3. Run high-performance version
sbatch scripts/watgpu_slurm/ttm_finetune.sh models/configs/ttm_high_performance_config.yaml

# 4. Compare results in registry
python results/runs/view_registry.py
```

## Troubleshooting

- **Config file not found**: Ensure the path is relative to the project root
- **YAML syntax errors**: Validate YAML syntax online or with `python -c "import yaml; yaml.safe_load(open('file.yaml'))"`
- **Memory errors**: Reduce batch_size in your config
- **Import errors**: Ensure PyYAML is installed: `pip install PyYAML`
