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

## Proposed Reorganization

### New Directory Structure

```
src/train/ttm/
├── __init__.py                   # Public API exports
├── core/
│   ├── __init__.py
│   ├── trainer.py               # TTMTrainer class - main orchestration
│   ├── model_factory.py         # Model creation and configuration
│   └── pipeline.py              # End-to-end training pipeline
            ├── data/
            │   ├── __init__.py
            │   ├── loaders.py               # Data loading strategies
            │   ├── preprocessing.py         # Feature reduction, imputation
            │   └── validation.py            # Data validation utilities
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py               # Custom metrics and callbacks
│   └── evaluator.py             # Evaluation pipeline
├── config/
│   ├── __init__.py
│   ├── manager.py               # Configuration loading and validation
│   └── defaults.py              # Default configurations
├── utils/
│   ├── __init__.py
│   ├── logging.py               # Debug/info logging utilities
│   ├── checkpointing.py         # Model checkpointing
│   └── io.py                    # File I/O utilities
└── cli/
    ├── __init__.py
    └── runner.py                # Command-line interface
```

### Legacy File Mapping

```
# Migration path for existing functionality
ttm.py →
├── core/trainer.py (main training logic)
├── data/loaders.py (cache loading)
├── data/preprocessing.py (feature reduction)
├── evaluation/metrics.py (custom metrics)
└── utils/logging.py (debug/info print)

ttm_custom_metrics.py →
├── evaluation/metrics.py (metrics and callbacks)
└── core/trainer.py (training logic)

ttm_original.py →
├── data/loaders.py (legacy loader support)
└── core/trainer.py (training logic)

ttm_runner.py →
└── cli/runner.py (CLI interface)
```

---

## Implementation Plan

### Phase 1: Core Module Creation (Week 1-2)

#### 1.1 Create Base Structure
```bash
mkdir -p src/train/ttm/{core,data,evaluation,config,utils,cli}
touch src/train/ttm/{__init__.py,core/__init__.py,data/__init__.py,evaluation/__init__.py,config/__init__.py,utils/__init__.py,cli/__init__.py}
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
