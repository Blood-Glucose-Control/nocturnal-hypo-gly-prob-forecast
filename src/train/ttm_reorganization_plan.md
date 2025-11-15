# TTM Training Pipeline Reorganization Plan

## Executive Summary

This document outlines a comprehensive reorganization plan for the Time Series Transformer Model (TTM) training pipeline to address significant code duplication, improve maintainability, and align with industry best practices for foundation model training.

**Current Issues:**
- 4 files with 60%+ code duplication (~2,000 lines total)
- Mixed concerns (data loading + training + metrics in single files)
- Inconsistent interfaces and return types
- Poor testability and reusability
- Anti-patterns not typical of production ML systems

**Target Outcome:**
- ~60% code reduction through deduplication
- Modular, testable, and maintainable architecture
- Industry-standard organization patterns
- Backward compatibility during migration

---

## Current State Analysis

### Existing File Structure Problems

| File | Size | Primary Issues |
|------|------|----------------|
| `ttm_original.py` | 455 lines | Uses deprecated data loader, simple logging |
| `ttm.py` | 765 lines | Most feature-complete, enhanced debugging |
| `ttm_custom_metrics.py` | 590 lines | Duplicate of ttm.py with different docs |
| `ttm_runner.py` | 171 lines | CLI wrapper, good pattern but isolated |

### Code Duplication Analysis

**Exact Duplicates:**
- ✅ `reduce_features_multi_patient()` - Identical across 3 files
- ✅ `load_processed_data_from_cache()` - Duplicated in 2 files
- ✅ `_get_finetune_trainer()` - 95%+ overlap across 3 files
- ✅ `CustomMetricsCallback` class - Nearly identical in 2 files

**Functional Differences:**
- **Data Loading**: `get_loader()` vs `load_processed_data_from_cache()`
- **Logging**: Simple `print()` vs sophisticated `debug_print()`/`info_print()`
- **Return Values**: `(trainer, test_dataset)` vs `metrics_summary` dict

---

## Industry Standard Comparison

### ❌ Current Anti-Patterns

```python
# Monolithic training function (NOT industry standard)
def finetune_ttm(model_path, data_source, ...):
    # 200+ lines mixing:
    # - Data loading
    # - Model setup
    # - Training logic
    # - Evaluation
    # - Checkpointing
    # - Metrics computation
```

### ✅ Industry Standard Patterns

**Examples from Leading Organizations:**

1. **HuggingFace Transformers:**
   - Single `Trainer` class handles all training
   - Modular components for models, data, metrics
   - Configuration-driven architecture

2. **PyTorch Lightning:**
   ```python
   class MyModel(LightningModule):
       def training_step(self, batch, batch_idx):
           return loss

   # Training script is minimal
   trainer = Trainer()
   trainer.fit(model, datamodule)
   ```

3. **OpenAI/Meta Approach:**
   - Thin training scripts (orchestration only)
   - Heavy business logic in reusable modules
   - Configuration management (Hydra, MLflow)

---

## Proposed Reorganization (Updated Based on Architecture Review)

**Key Principles from Architecture Review:**
- ✅ **Use existing `src/data/` infrastructure** via `CacheManager` - no duplicate data directories
- ✅ **Separate from legacy evaluation** - ignore `src/eval/` and traditional ML benchmarking
- ✅ **New output structure** for multiple TSFMs and experiments
- ✅ **Focus on TSFM-specific logic only** - leverage existing preprocessing utilities

**Modern Foundation Model Requirements:**
- 🔄 **Distributed Training**: Multi-GPU/multi-node support for large models
- 🔄 **Memory Optimization**: Gradient checkpointing, mixed precision, model sharding
- 🔄 **Advanced Training**: LoRA, QLoRA, parameter-efficient fine-tuning
- 🔄 **Model Versioning**: Proper model registry and experiment tracking
- 🔄 **Configuration Management**: Hydra-style hierarchical configs
- 🔄 **Cloud Integration**: Wandb, MLflow, HuggingFace Hub integration

### Updated Directory Structure

```
src/train/ttm/
├── __init__.py                   # Public API exports
├── core/
│   ├── __init__.py
│   ├── trainer.py               # TTMTrainer class - main orchestration
│   ├── model_factory.py         # Model creation and configuration
│   └── pipeline.py              # End-to-end training pipeline
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py               # TTM-specific metrics and callbacks (separate from legacy)
│   └── evaluator.py             # Evaluation pipeline
├── config/
│   ├── __init__.py
│   ├── manager.py               # Configuration loading and validation
│   └── defaults.py              # Default configurations
├── utils/
│   ├── __init__.py
│   ├── logging.py               # TTM-specific logging utilities
│   ├── checkpointing.py         # Model checkpointing
│   └── preprocessing.py         # TTM-specific preprocessing (uses src/data/ infrastructure)
└── cli/
    ├── __init__.py
    └── runner.py                # Command-line interface
```

### Data Integration Strategy

**Use Existing Infrastructure (No Duplicate Data Directory):**
```python
# TTM will use existing src/data/ infrastructure
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

### Legacy File Mapping

```
# Updated migration path - leverages existing infrastructure
ttm.py →
├── core/trainer.py (main training logic)
├── utils/preprocessing.py (TTM-specific preprocessing using src/data/)
├── evaluation/metrics.py (TTM metrics, separate from legacy)
└── utils/logging.py (TTM-specific logging)

ttm_custom_metrics.py →
├── evaluation/metrics.py (metrics and callbacks)
└── core/trainer.py (training logic)

ttm_original.py →
├── utils/preprocessing.py (adapter for legacy loader)
└── core/trainer.py (training logic)

ttm_runner.py →
└── cli/runner.py (CLI interface)
```

### Research-Focused Improvements (Phase 1)

**Goal**: Make the codebase cleaner and more maintainable for research, without over-engineering.

**Key Research Needs:**
1. **Easy experiment management** - Simple config changes, parameter sweeps
2. **Multi-GPU support** - Essential for foundation model research
3. **Better debugging** - Clear logging, metric tracking
4. **Quick iteration** - Fast setup of new experiments
5. **Reproducibility** - Consistent results across runs

**Phase 1 Priorities (Research-Appropriate):**

#### 1. **Multi-GPU Training Support (Critical for Foundation Models)**
```python
# Add to core/trainer.py - Simple but essential
import torch
from accelerate import Accelerator

class TTMTrainer:
    def __init__(self, config):
        # Simple multi-GPU setup
        self.accelerator = Accelerator()
        self.device = self.accelerator.device

    def train(self, model, data):
        # Automatic multi-GPU handling
        model, optimizer, dataloader = self.accelerator.prepare(
            model, optimizer, dataloader
        )
```

#### 2. **Parameter-Efficient Fine-tuning (LoRA)**
```python
# Add to core/adapters.py - Huge memory savings for research
from peft import LoraConfig, get_peft_model

def add_lora_adapters(model, rank=16):
    """Add LoRA adapters - reduces memory by 10x for fine-tuning"""
    lora_config = LoraConfig(r=rank, target_modules=["attention"])
    return get_peft_model(model, lora_config)
```

#### 3. **Simple Experiment Tracking**
```python
# Keep it simple - just structured logging + Wandb integration
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

### Proposed New Output Structure for Multiple TSFMs

**Problem**: Current `results/`, `models/`, and `scripts/` directories are organized for traditional ML experiments, not multiple foundation models with multiple experiments each.

**Solution**: Simple, TSFM research-focused output structure:

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

**Benefits for Research:**
- **Easy comparison**: All experiments organized by model and dataset
- **Quick iteration**: Simple directory structure, easy to navigate
- **Reproducibility**: Config and results stored together
- **No over-engineering**: Focused on research needs, not production complexity

### Updated Directory Structure (Research-Focused)

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
├── results/                         # Experiment results and analysis
│   ├── ttm/
│   │   ├── kaggle_experiment_1/
│   │   │   ├── evaluation_report.json
│   │   │   ├── comparison_plots/
│   │   │   └── run_history.csv
│   │   └── ...
│   └── chronos/
└── scripts/                         # TSFM-specific experiment scripts
    ├── ttm/
    │   ├── kaggle_fine_tune.sh
    │   ├── aleppo_baseline.sh
    │   └── production_deploy.py
    ├── chronos/
    └── shared/                      # Common TSFM utilities
        ├── slurm_templates/
        └── experiment_tracking.py
```

**Benefits:**
- **Multi-model support**: Each TSFM gets its own namespace
- **Experiment organization**: Clear separation of different experimental setups
- **Production tracking**: Dedicated space for production-ready models
- **Script organization**: TSFM-specific orchestration separate from legacy
- **Best model tracking**: Symlinks point to best performing runs

---

## Research-Focused Implementation Plan

### Phase 1: Essential Research Features (Week 1-2)

#### 1.1 Create Simple Structure
```bash
mkdir -p src/train/ttm/{core,evaluation,config,utils,cli}
touch src/train/ttm/{__init__.py,core/__init__.py,evaluation/__init__.py,config/__init__.py,utils/__init__.py,cli/__init__.py}
```

#### 1.2 Multi-GPU Support (Critical for Foundation Models)
- **`core/trainer.py`**: Add Accelerate library integration for easy multi-GPU
  ```python
  from accelerate import Accelerator

  class TTMTrainer:
      def __init__(self):
          self.accelerator = Accelerator()  # Handles multi-GPU automatically
  ```

#### 1.3 Parameter-Efficient Fine-tuning (Huge Research Value)
- **`core/adapters.py`**: Add LoRA support for memory-efficient training
  ```python
  from peft import LoraConfig, get_peft_model

  def add_lora_to_model(model, rank=16):
      # Reduces memory usage by ~10x for fine-tuning
      return get_peft_model(model, LoraConfig(r=rank))
  ```

#### 1.4 Simple Experiment Tracking
- **`utils/logging.py`**: Enhanced logging with optional Wandb integration
- **`evaluation/callbacks.py`**: Research-friendly callbacks (checkpointing, metrics)

#### 1.5 Data Integration (Use Existing Infrastructure)
- **`utils/preprocessing.py`**: TTM preprocessing using existing CacheManager
- Remove duplicate functions, use `src/data/` infrastructure

### Why These Changes Matter for Research

#### **Multi-GPU Support = Faster Experiments**
```python
# Before: Single GPU, slow training
trainer = TTMTrainer()  # Takes 8 hours on single GPU

# After: Multi-GPU, faster iteration
trainer = TTMTrainer()  # Takes 2 hours on 4 GPUs
```
**Research Impact**: 4x faster experiments = 4x more iterations per day

#### **LoRA = More Experiments with Less Memory**
```python
# Before: Full fine-tuning
model = load_ttm_model()  # Requires 80GB VRAM, limits experiments

# After: LoRA fine-tuning
model = add_lora_to_model(load_ttm_model())  # Requires 8GB VRAM
```
**Research Impact**: Run 10x more experiments on same hardware

#### **Better Organization = Easier Comparison**
```
# Before: Files scattered everywhere
ttm.py, ttm_custom_metrics.py, ttm_original.py...

# After: Organized experiments
experiments/ttm/lora_vs_full_finetune/
├── lora_rank_8/
├── lora_rank_16/
└── full_finetune/
```
**Research Impact**: Easy to compare different approaches, reproduce results

### Skip These (Too Complex for Research Phase)
- ❌ **Advanced Model Registry** - Simple file organization is fine
- ❌ **Hydra Configuration** - YAML + argparse works well
- ❌ **Production Deployment** - Focus on research first
- ❌ **Complex MLOps** - Wandb integration is enough

### What to Build Next (Phase 2)
1. **Hyperparameter sweeps** - Simple scripts to test different configs
2. **Result comparison tools** - Scripts to compare experiment results
3. **Better visualization** - Plot training curves, metrics across experiments
  ```python
  class TTMMetricsCallback(TrainerCallback):
      # TTM-specific metrics, completely separate from src/eval/
      pass
  ```

#### 1.2 Extract Utilities First
- **`utils/logging.py`**: Move debug_print/info_print from ttm.py
- **`config/manager.py`**: Extract YAML loading from ttm_runner.py
- **`config/defaults.py`**: Define default configurations

#### 1.3 Data Layer
- **`data/loaders.py`**:
  ```python
  class CacheDataLoader:
      def load_processed_data_from_cache(self, data_source_name): ...

  class LegacyDataLoader:
      def load_via_get_loader(self, data_source_name): ...
  ```
- **`data/preprocessing.py`**: Move `reduce_features_multi_patient()`

#### 1.4 Evaluation Layer
- **`evaluation/metrics.py`**:
  ```python
  class CustomMetricsCallback(TrainerCallback): ...
  def compute_custom_metrics(eval_pred): ...
  ```

### Phase 2: Core Training Logic (Week 3-4)

#### 2.1 Model Factory
- **`core/model_factory.py`**:
  ```python
  class TTMModelFactory:
      def create_model(self, config):
          return get_model(
              config.model_path,
              context_length=config.context_length,
              # ... other parameters
          )
  ```

#### 2.2 Main Trainer
- **`core/trainer.py`**:
  ```python
  class TTMTrainer:
      def __init__(self, config):
          self.config = config
          self.data_loader = self._create_data_loader()
          self.model_factory = TTMModelFactory()
          self.evaluator = TTMEvaluator()

      def train(self):
          # Orchestrate training pipeline
          data = self.data_loader.load(self.config.data_source_name)
          model = self.model_factory.create_model(self.config)
          trainer = self._create_hf_trainer(model, data)

          # Training
          trainer.train()

          # Evaluation
          metrics = self.evaluator.evaluate(trainer, data)
          return metrics
  ```

#### 2.3 Pipeline Integration
- **`core/pipeline.py`**: High-level pipeline orchestration
- **`cli/runner.py`**: Updated CLI interface using new modules

### Phase 3: Testing and Validation (Week 5)

#### 3.1 Unit Tests
```python
# tests/test_data_loaders.py
def test_cache_data_loader():
    loader = CacheDataLoader()
    data = loader.load_processed_data_from_cache("kaggle_brisT1D")
    assert data is not None

# tests/test_trainer.py
def test_ttm_trainer():
    config = TTMConfig(...)
    trainer = TTMTrainer(config)
    metrics = trainer.train()
    assert "eval_loss" in metrics
```

#### 3.2 Integration Tests
- End-to-end training with small dataset
- Configuration validation
- Backward compatibility verification

### Phase 4: Migration and Deprecation (Week 6)

#### 4.1 Backward Compatibility Layer
```python
# src/train/ttm/__init__.py
from .core.trainer import TTMTrainer
from .legacy import finetune_ttm  # Compatibility wrapper

def finetune_ttm(*args, **kwargs):
    """Legacy function - use TTMTrainer class instead."""
    warnings.warn("finetune_ttm is deprecated, use TTMTrainer", DeprecationWarning)
    # Convert old interface to new
    config = convert_legacy_args(*args, **kwargs)
    trainer = TTMTrainer(config)
    return trainer.train()
```

#### 4.2 Update Existing Scripts
- Update imports in existing notebooks/scripts
- Add migration guides to documentation
- Update configuration files

#### 4.3 Deprecation Warnings
```python
# Add to old files
import warnings
warnings.warn(
    "ttm.py is deprecated. Use src.train.ttm.TTMTrainer instead.",
    DeprecationWarning,
    stacklevel=2
)
```

---

## New Public API

### Main Training Interface

```python
from src.train.ttm import TTMTrainer, TTMConfig

# Configuration-driven approach
config = TTMConfig.from_yaml("configs/experiments/ttm_baseline.yaml")
trainer = TTMTrainer(config)
metrics = trainer.train()

# Or programmatic approach
config = TTMConfig(
    model_path="ibm-granite/granite-timeseries-ttm-r2",
    data_source_name="kaggle_brisT1D",
    batch_size=128,
    # ... other parameters
)
trainer = TTMTrainer(config)
metrics = trainer.train()
```

### CLI Interface

```bash
# Simple training
python -m src.train.ttm.cli --config configs/experiments/ttm_baseline.yaml

# With overrides
python -m src.train.ttm.cli --config configs/experiments/ttm_baseline.yaml \
    --batch-size 256 --num-epochs 20

# Legacy compatibility
python -m src.train.ttm.cli --legacy-mode ttm_original
```

### Configuration Structure

```yaml
# configs/experiments/ttm_baseline.yaml
model:
  path: "ibm-granite/granite-timeseries-ttm-r2"
  context_length: 512
  forecast_length: 96

data:
  source_name: "kaggle_brisT1D"
  loader_type: "cache"  # or "legacy"
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

output:
  save_metrics: true
  metrics_format: "json"
```

---

## Benefits and Impact

### Code Quality Improvements

1. **~60% Code Reduction**: From ~2,000 lines to ~800 lines
2. **Single Source of Truth**: Each function implemented once
3. **Clear Separation of Concerns**: Data, training, evaluation, config separated
4. **Better Error Handling**: Centralized error handling and logging
5. **Improved Documentation**: Each module has clear purpose and API

### Maintainability Benefits

1. **Easier Bug Fixes**: Changes only need to happen in one place
2. **Simpler Testing**: Each component can be unit tested independently
3. **Better Team Collaboration**: Clear ownership boundaries
4. **Easier Feature Addition**: New data sources, models, metrics easy to add

### Development Workflow Improvements

1. **Faster Development**: Reusable components speed up experimentation
2. **Better Debugging**: Centralized logging and error handling
3. **Configuration Management**: YAML-based experiment configuration
4. **Reproducibility**: Clear configuration tracking and versioning

### Alignment with Industry Standards

1. **Modular Architecture**: Matches HuggingFace, PyTorch Lightning patterns
2. **Configuration-Driven**: Standard ML operations practice
3. **Testable Components**: Industry-standard testing approaches
4. **Scalable Design**: Easy to extend for production use

---

## Risk Mitigation

### Backward Compatibility

- Legacy wrapper functions maintain existing interfaces
- Gradual migration path with deprecation warnings
- Comprehensive testing of compatibility layer

### Testing Strategy

- Unit tests for each new component
- Integration tests for full pipeline
- Regression tests against existing results
- Performance benchmarking

### Rollback Plan

- Keep existing files during transition period
- Feature flag to switch between old/new implementations
- Comprehensive documentation of changes

---

## Success Metrics

### Code Quality Metrics
- [ ] Lines of code reduced by 50%+
- [ ] Code duplication eliminated (0% duplication)
- [ ] Test coverage > 80%
- [ ] Documentation coverage > 90%

### Performance Metrics
- [ ] Training time unchanged (±5%)
- [ ] Memory usage unchanged (±10%)
- [ ] Model performance unchanged
- [ ] Reproducible results

### Developer Experience Metrics
- [ ] Setup time for new experiments < 5 minutes
- [ ] New data source integration < 1 hour
- [ ] Bug fix time reduced by 50%
- [ ] Team onboarding time reduced

---

## Timeline Summary

| Week | Phase | Deliverables |
|------|-------|-------------|
| 1-2 | Core Module Creation | Utils, data, evaluation modules |
| 3-4 | Training Logic | Trainer class, model factory, pipeline |
| 5 | Testing | Unit tests, integration tests |
| 6 | Migration | Compatibility layer, documentation, deprecation |

**Total Estimated Effort**: 6 weeks (1 developer)

---

## Conclusion

This reorganization plan transforms the TTM training pipeline from a collection of duplicated scripts into a professional, maintainable, and scalable foundation model training system. The modular architecture follows industry best practices and provides a solid foundation for future development and research.

The phased approach ensures minimal disruption to ongoing work while delivering immediate benefits in code quality and maintainability. The backward compatibility layer ensures existing experiments and workflows continue to function during the transition period.
