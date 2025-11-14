# TTM Training Pipeline Directory Structures

This document provides a clean overview of the proposed directory structures for the TTM training pipeline reorganization.

---

## Current File Structure (Problems)

```
src/train/
├── ttm.py                    # 765 lines - Most feature-complete, enhanced debugging
├── ttm_original.py           # 455 lines - Uses deprecated data loader, simple logging  
├── ttm_custom_metrics.py     # 590 lines - Duplicate of ttm.py with different docs
└── ttm_runner.py             # 171 lines - CLI wrapper, good pattern but isolated
```

**Issues:**
- 60%+ code duplication across 4 files (~2,000 lines total)
- Mixed concerns (data + training + metrics in single files)
- Inconsistent interfaces and return types

---

## Proposed New Structure (Research-Focused)

### Main TTM Module Structure

```
src/train/ttm/
├── __init__.py                   # Public API exports
├── core/
│   ├── __init__.py
│   ├── trainer.py               # Main TTM training logic with multi-GPU
│   ├── model_factory.py         # Model creation 
│   └── adapters.py              # LoRA and parameter-efficient methods
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py               # TTM-specific metrics (separate from legacy)
│   └── callbacks.py             # Training callbacks (checkpointing, logging)
├── config/
│   ├── __init__.py
│   ├── manager.py               # Simple YAML loading (no Hydra complexity)
│   └── defaults.py              # Default configurations
├── utils/
│   ├── __init__.py
│   ├── logging.py               # Research-friendly logging + optional Wandb
│   └── preprocessing.py         # TTM preprocessing using existing src/data/
└── cli/
    ├── __init__.py
    └── runner.py                # Simple CLI for experiments
```

### Experiment Output Structure

```
experiments/                         # Simple experiment tracking
├── ttm/
│   ├── kaggle_experiment_1/
│   │   ├── run_2025-11-12_14-30-45/     # Timestamped runs
│   │   │   ├── checkpoints/              # Model checkpoints
│   │   │   ├── logs/                     # Training logs
│   │   │   ├── config.yaml               # Full experiment config
│   │   │   ├── metrics.json              # Training metrics
│   │   │   └── results.txt               # Quick results summary
│   │   └── best_run -> run_2025-11-12_14-30-45/  # Symlink to best
│   ├── aleppo_baseline/
│   └── lora_experiments/
├── chronos/                         # Future models
└── shared/                          # Shared analysis scripts
    ├── compare_experiments.py
    └── plot_results.py
```

---

## Migration Path (Legacy → New Structure)

### File Migration Mapping

```
# Current files → New structure
ttm.py →
├── core/trainer.py              # Main training logic
├── utils/preprocessing.py       # TTM preprocessing (uses existing src/data/)
├── evaluation/metrics.py        # TTM metrics (separate from legacy)
└── utils/logging.py             # TTM-specific logging

ttm_custom_metrics.py →
├── evaluation/metrics.py        # Metrics and callbacks
└── core/trainer.py             # Training logic

ttm_original.py →
├── utils/preprocessing.py       # Adapter for legacy loader
└── core/trainer.py             # Training logic

ttm_runner.py →
└── cli/runner.py               # CLI interface
```

### Data Integration Strategy

```python
# TTM will use existing src/data/ infrastructure (no duplication)
from src.data.cache_manager import get_cache_manager
from src.tuning.benchmark import impute_missing_values

# Example: TTM preprocessing using existing utilities
class TTMPreprocessor:
    def __init__(self):
        self.cache_manager = get_cache_manager()
    
    def load_and_prepare(self, dataset_name: str):
        # Use existing cache system
        data = self.cache_manager.load_full_processed_data(dataset_name)
        # Use existing imputation utilities
        return impute_missing_values(data, ...)
```

---

## Key Research Features

### 1. Multi-GPU Training Structure
```python
# core/trainer.py
from accelerate import Accelerator

class TTMTrainer:
    def __init__(self, config):
        self.accelerator = Accelerator()  # Handles multi-GPU automatically
        self.device = self.accelerator.device
        
    def train(self, model, data):
        # Automatic multi-GPU handling
        model, optimizer, dataloader = self.accelerator.prepare(
            model, optimizer, dataloader
        )
```

### 2. Parameter-Efficient Fine-tuning Structure
```python
# core/adapters.py
from peft import LoraConfig, get_peft_model

def add_lora_adapters(model, rank=16):
    """Add LoRA adapters - reduces memory by 10x for fine-tuning"""
    lora_config = LoraConfig(r=rank, target_modules=["attention"])
    return get_peft_model(model, lora_config)
```

### 3. Simple Experiment Tracking Structure
```python
# utils/logging.py
class ExperimentLogger:
    def __init__(self, use_wandb=False):
        self.use_wandb = use_wandb
        if use_wandb:
            import wandb
            wandb.init()
    
    def log_metrics(self, metrics, step):
        # Log to both console and wandb if enabled
        pass
```

---

## Configuration Structure

### Simple YAML Configuration
```yaml
# configs/experiments/ttm_baseline.yaml
model:
  path: "ibm-granite/granite-timeseries-ttm-r2"
  context_length: 512
  forecast_length: 96

data:
  source_name: "kaggle_brisT1D"
  loader_type: "cache"
  y_feature: ["bg_mM"]
  x_features: ["steps", "cob", "carb_availability", "insulin_availability", "iob"]
  timestamp_column: "datetime"
  resolution_min: 5
  data_split: [0.9, 0.1]
  fewshot_percent: 100

training:
  batch_size: 128
  learning_rate: 0.001
  num_epochs: 10
  use_cpu: false
  loss: "mse"
  dataloader_num_workers: 2

logging:
  debug_mode: false
  log_level: "info"
  use_wandb: false

output:
  save_metrics: true
  metrics_format: "json"
```

---

## Implementation Phases

### Phase 1: Essential Structure Creation
```bash
mkdir -p src/train/ttm/{core,evaluation,config,utils,cli}
touch src/train/ttm/{__init__.py,core/__init__.py,evaluation/__init__.py,config/__init__.py,utils/__init__.py,cli/__init__.py}
```

### Key Components to Implement First:
1. **Multi-GPU Support** (`core/trainer.py`)
2. **LoRA Integration** (`core/adapters.py`) 
3. **Simple Logging** (`utils/logging.py`)
4. **Data Integration** (`utils/preprocessing.py`)

---

## Benefits Summary

### Code Quality
- **~60% code reduction**: From ~2,000 lines to ~800 lines
- **Zero duplication**: Each function implemented once
- **Clear separation**: Data, training, evaluation separated

### Research Impact  
- **4x faster experiments**: Multi-GPU support
- **10x memory reduction**: LoRA fine-tuning
- **Easy comparison**: Organized experiment structure
- **Quick iteration**: Simple config changes

### Skip for Research Phase
- ❌ Advanced Model Registry (too complex)
- ❌ Hydra Configuration (YAML + argparse is fine)
- ❌ Production Deployment (research focus first)
- ❌ Complex MLOps (Wandb integration sufficient)

---

This structure provides a clean, research-focused foundation for TTM training while eliminating code duplication and improving maintainability.