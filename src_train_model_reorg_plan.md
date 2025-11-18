# Training Pipeline Reorganization Plan

## Executive Summary

This document outlines a comprehensive reorganization plan for the nocturnal-hypo-gly-prob-forecast repository to support modern Time Series Foundation Model (TSFM) research. The plan emphasizes scalability, reproducibility, and maintainability while following best practices from top academic and industry research labs.

## Current State Analysis

### Issues Identified
1. **Training Code Duplication**: TTM modules contain significant code duplication
2. **Scattered Configurations**: Model configs spread across multiple directories
3. **Inconsistent Experiment Structure**: Mixed organization between `experiments/`, `results/`, and `models/`
4. **Limited Distributed Training Support**: No unified multi-GPU framework
5. **Manual Experiment Tracking**: Limited automation for reproducibility
6. **Fragmented Evaluation**: No centralized evaluation pipeline

## Proposed Directory Structure

```
nocturnal/
├── src/
│   ├── data/                           # Keep existing, enhance with versioning
│   │   ├── versioning/
│   │   │   ├── __init__.py
│   │   │   ├── data_versioning.py      # Simple data version tracking
│   │   │   └── dataset_registry.py     # Track dataset configs used in experiments
│   │   └── [existing structure...]
│   │
│   ├── models/                         # NEW: Unified model architectures
│   │   ├── __init__.py
│   │   ├── base/
│   │   │   ├── __init__.py
│   │   │   ├── base_model.py          # Abstract base for all TSFMs
│   │   │   ├── distributed.py         # Multi-GPU training utilities
│   │   │   └── lora_utils.py          # LoRA integration
│   │   ├── ttm/
│   │   │   ├── __init__.py
│   │   │   ├── model.py               # Refactored TTM implementation
│   │   │   ├── config.py              # TTM-specific configs
│   │   │   └── trainer.py             # TTM trainer with distributed support
│   │   ├── chronos/
│   │   │   ├── __init__.py
│   │   │   ├── model.py
│   │   │   ├── config.py
│   │   │   └── trainer.py
│   │   ├── timegpt/
│   │   ├── ts2vec/
│   │   ├── timellm/
│   │   ├── tide/
│   │   └── tsmixer/
│   │
│   ├── training/                       # NEW: Unified training pipeline
│   │   ├── __init__.py
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── trainer_factory.py     # Create trainers for different models
│   │   │   ├── distributed_launcher.py # Handle multi-GPU setup
│   │   │   ├── checkpoint_manager.py   # Model saving/loading
│   │   │   └── training_loop.py       # Common training logic
│   │   ├── strategies/
│   │   │   ├── __init__.py
│   │   │   ├── fine_tuning.py         # Fine-tuning strategies
│   │   │   ├── from_scratch.py        # Training from scratch
│   │   │   └── zero_shot.py           # Zero-shot evaluation
│   │   └── utils/
│   │       ├── __init__.py
│   │       ├── reproducibility.py     # Seed setting, version tracking
│   │       └── memory_optimization.py # LoRA, gradient checkpointing
│   │
│   ├── evaluation/                     # Enhanced evaluation pipeline
│   │   ├── __init__.py
│   │   ├── metrics/
│   │   │   ├── __init__.py
│   │   │   ├── regression.py          # RMSE, MSE, MAE, MAPE
│   │   │   ├── clinical.py            # Clarke Error Grid Analysis
│   │   │   └── probabilistic.py       # Metrics for probabilistic forecasts
│   │   ├── evaluator.py               # Main evaluation orchestrator
│   │   └── report_generator.py        # Automated report generation
│   │
│   ├── experiments/                    # NEW: Experiment management
│   │   ├── __init__.py
│   │   ├── base/
│   │   │   ├── __init__.py
│   │   │   ├── experiment.py          # Base experiment class
│   │   │   └── registry.py            # Experiment tracking
│   │   ├── nocturnal/
│   │   │   ├── __init__.py
│   │   │   └── experiments.py         # Nocturnal-specific experiments
│   │   ├── prandial/
│   │   │   ├── __init__.py
│   │   │   └── experiments.py         # Prandial-specific experiments
│   │   ├── ablation/
│   │   │   ├── __init__.py
│   │   │   └── experiments.py         # Data ablation experiments
│   │   ├── personalization/
│   │   │   ├── __init__.py
│   │   │   └── experiments.py         # Personalization experiments
│   │   └── transfer_learning/
│   │       ├── __init__.py
│   │       └── experiments.py         # Cross-dataset/patient transfer
│   │
│   ├── registry/                       # NEW: Model and experiment registries
│   │   ├── __init__.py
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── model_registry.py      # Enhanced model tracking
│   │   │   └── version_manager.py     # Model versioning
│   │   └── experiments/
│   │       ├── __init__.py
│   │       ├── experiment_registry.py  # Experiment result tracking
│   │       └── result_aggregator.py   # Cross-experiment analysis
│   │
│   └── [existing utils/, tuning/...]
│
├── configs/                            # NEW: Centralized configuration management
│   ├── models/
│   │   ├── ttm/
│   │   │   ├── base.yaml              # Base TTM configuration
│   │   │   ├── fine_tune.yaml         # Fine-tuning specific
│   │   │   └── from_scratch.yaml      # Training from scratch
│   │   ├── chronos/
│   │   ├── timegpt/
│   │   └── [other models...]
│   ├── experiments/
│   │   ├── nocturnal/
│   │   │   ├── point_forecast.yaml
│   │   │   └── probabilistic_forecast.yaml
│   │   ├── prandial/
│   │   ├── ablation/
│   │   └── [other experiment types...]
│   ├── data/
│   │   ├── kaggle_bris_t1d.yaml
│   │   ├── aleppo.yaml
│   │   ├── gluroo.yaml
│   │   └── multi_dataset.yaml         # Multi-dataset training configs
│   ├── training/
│   │   ├── single_gpu.yaml
│   │   ├── multi_gpu_ddp.yaml
│   │   ├── multi_gpu_deepspeed.yaml
│   │   └── slurm_cluster.yaml
│   └── evaluation/
│       ├── standard_metrics.yaml
│       ├── clinical_metrics.yaml
│       └── multi_testset.yaml
│
├── experiments/                        # REORGANIZED: Experiment results storage
│   ├── nocturnal_forecast/
│   │   ├── point_forecast/
│   │   │   ├── ttm_vs_chronos_2025_11_17/
│   │   │   │   ├── experiment_config.yaml
│   │   │   │   ├── results/
│   │   │   │   │   ├── metrics_summary.json
│   │   │   │   │   ├── detailed_results.csv
│   │   │   │   │   └── figures/
│   │   │   │   ├── logs/
│   │   │   │   └── metadata.json      # Git commit, environment, etc.
│   │   │   └── [other point forecast experiments...]
│   │   └── probabilistic_forecast/
│   ├── prandial_forecast/
│   ├── data_ablation/
│   ├── personalization/
│   └── transfer_learning/
│
├── trained_models/                     # NEW: Centralized model artifacts
│   ├── registry.db                     # SQLite for complex queries (future)
│   ├── registry.csv                    # Simple CSV for now
│   └── artifacts/
│       ├── ttm/
│       │   ├── 20251117_142301_kaggle_ft/  # timestamp_dataset_strategy
│       │   │   ├── model.pt
│       │   │   ├── config.yaml
│       │   │   ├── training_log.json
│       │   │   ├── metadata.json
│       │   │   └── checkpoints/
│       │   └── [other TTM models...]
│       ├── chronos/
│       └── [other model types...]
│
└── scripts/
    ├── training/                       # ENHANCED: Training entry points
    │   ├── train_model.py             # Unified training script
    │   ├── slurm/
    │   │   ├── single_gpu.sh
    │   │   ├── multi_gpu.sh
    │   │   └── adaptive_resources.sh   # Auto-select GPU config
    │   └── [existing scripts...]
    ├── experiments/                    # NEW: Experiment runners
    │   ├── run_experiment.py          # Main experiment runner
    │   ├── nocturnal_experiments.py
    │   ├── prandial_experiments.py
    │   └── ablation_experiments.py
    └── [existing directories...]
```

## Key Components

### 1. Unified Model Architecture (`src/models/`)

#### Base Model Framework
```python
# src/models/base/base_model.py
class BaseTSFM(ABC):
    """Abstract base class for all Time Series Foundation Models"""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.distributed_strategy = None
        self.lora_config = None

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def setup_distributed(self, strategy: str = "ddp"):
        """Configure distributed training strategy"""
        pass

    def enable_lora(self, lora_config: LoRAConfig):
        """Enable LoRA for memory-efficient fine-tuning"""
        pass
```

#### Model-Specific Implementations
Each model (TTM, Chronos, etc.) inherits from `BaseTSFM` and implements:
- Model-specific architecture
- Custom training procedures
- Distributed training optimizations
- LoRA integration points

### 2. Training Pipeline (`src/training/`)

#### Unified Training Interface
```python
# src/training/core/trainer_factory.py
class TrainerFactory:
    @staticmethod
    def create_trainer(model_type: str, config: TrainingConfig):
        """Factory method to create appropriate trainer"""
        trainers = {
            "ttm": TTMTrainer,
            "chronos": ChronosTrainer,
            "timegpt": TimeGPTTrainer,
            # ... other models
        }
        return trainers[model_type](config)
```

#### Distributed Training Support
- **PyTorch DDP**: For most models, configurable per model type
- **DeepSpeed**: For large models requiring advanced memory optimization
- **Automatic GPU Detection**: Scripts auto-detect available GPUs and configure accordingly

### 3. Configuration Management (`configs/`)

#### Hierarchical Configuration Structure
- **Model Configs**: Model-specific parameters, architecture settings
- **Training Configs**: Distributed training settings, optimization parameters
- **Data Configs**: Dataset specifications, preprocessing parameters
- **Experiment Configs**: End-to-end experiment definitions

#### Configuration Composition
```yaml
# configs/experiments/nocturnal/ttm_point_forecast.yaml
model:
  type: ttm
  config: !include ../models/ttm/fine_tune.yaml

training:
  config: !include ../training/multi_gpu_ddp.yaml

data:
  datasets: !include ../data/multi_dataset.yaml

evaluation:
  config: !include ../evaluation/standard_metrics.yaml
```

### 4. Experiment Management (`src/experiments/`)

#### Experiment Framework
```python
# src/experiments/base/experiment.py
class BaseExperiment:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.registry = ExperimentRegistry()

    def setup_data(self):
        """Setup data pipelines with proper train/test splits"""
        pass

    def setup_models(self):
        """Initialize models for comparison"""
        pass

    def run_training(self):
        """Execute training phase"""
        pass

    def run_evaluation(self):
        """Execute evaluation on holdout sets"""
        pass

    def generate_report(self):
        """Generate experiment report"""
        pass
```

#### Experiment Types
- **Nocturnal Experiments**: Night-time glucose forecasting
- **Prandial Experiments**: Meal-time glucose forecasting
- **Ablation Studies**: Data ablation experiments
- **Personalization**: Patient-specific model adaptation
- **Transfer Learning**: Cross-dataset, cross-CGM, cross-patient studies

### 5. Model Registry (`src/registry/`)

#### Simple Registry (Current Implementation)
```python
# src/registry/models/model_registry.py
class ModelRegistry:
    def __init__(self, registry_path: str = "trained_models/registry.csv"):
        self.registry_path = registry_path
        self.df = self._load_registry()

    def register_model(self, model_info: ModelInfo):
        """Register a newly trained model"""
        entry = {
            "model_id": f"{model_info.type}_{model_info.timestamp}",
            "model_type": model_info.type,
            "training_strategy": model_info.strategy,  # fine_tune, from_scratch, etc.
            "dataset": model_info.dataset,
            "timestamp": model_info.timestamp,
            "artifact_path": model_info.artifact_path,
            "config_hash": model_info.config_hash,
            "git_commit": model_info.git_commit,
            "performance_metrics": json.dumps(model_info.metrics),
            "training_time_hours": model_info.training_time,
            "gpu_config": model_info.gpu_config
        }
        # Add to CSV and save
        pass

    def get_model(self, model_id: str) -> ModelInfo:
        """Retrieve model information"""
        pass

    def find_models(self, **filters) -> List[ModelInfo]:
        """Find models by criteria"""
        pass
```

#### Future MLflow Integration Points
- Registry designed to be easily migrated to MLflow
- Consistent metadata structure
- Artifact tracking preparation

### 6. Evaluation Pipeline (`src/evaluation/`)

#### Automated Evaluation
```python
# src/evaluation/evaluator.py
class ModelEvaluator:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.metrics = self._load_metrics()

    def evaluate_model(self, model, test_datasets):
        """Evaluate model on multiple test sets"""
        results = {}
        for dataset_name, dataset in test_datasets.items():
            predictions = model.predict(dataset)
            dataset_results = {}

            for metric_name, metric in self.metrics.items():
                score = metric.compute(predictions, dataset.targets)
                dataset_results[metric_name] = score

            results[dataset_name] = dataset_results

        return results
```

#### Metrics Implementation
- **Regression Metrics**: RMSE, MSE, MAE, MAPE
- **Clinical Metrics**: Clarke Error Grid Analysis
- **Probabilistic Metrics**: For uncertainty quantification
- **Custom Metrics**: Domain-specific glucose forecasting metrics

### 7. Data Versioning (`src/data/versioning/`)

#### Simple Data Tracking
```python
# src/data/versioning/data_versioning.py
class DataVersionManager:
    def __init__(self):
        self.version_registry = {}

    def create_data_snapshot(self, dataset_config: dict) -> str:
        """Create a versioned snapshot of data configuration"""
        config_hash = hashlib.md5(
            json.dumps(dataset_config, sort_keys=True).encode()
        ).hexdigest()[:8]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_id = f"{timestamp}_{config_hash}"

        self.version_registry[version_id] = {
            "config": dataset_config,
            "timestamp": timestamp,
            "hash": config_hash
        }

        return version_id
```

## Migration Plan for TTM Code

### Phase 1: Extract Common Functionality
1. **Identify Shared Code**: Analyze `src/train/ttm/` for reusable components
2. **Create Base Classes**: Move common training logic to `src/training/core/`
3. **Extract TTM Model**: Refactor `ttm.py` to `src/models/ttm/model.py`

### Phase 2: Implement New Structure
1. **Create TTM Trainer**: Implement `src/models/ttm/trainer.py` with distributed support
2. **Add Configuration**: Create hierarchical configs in `configs/models/ttm/`
3. **Implement Registry**: Set up model registration for TTM

### Phase 3: Testing and Validation
1. **Unit Tests**: Test individual components
2. **Integration Tests**: Test full training pipeline
3. **Performance Validation**: Ensure no performance regression

### Phase 4: Migration Scripts
```bash
# scripts/migration/migrate_ttm.py
# Automated migration script to move existing TTM models to new registry
```

## Implementation Roadmap

### Week 1: Foundation Setup
- [ ] Create new directory structure
- [ ] Implement base model classes
- [ ] Set up configuration management system

### Week 2: Training Pipeline
- [ ] Implement unified training framework
- [ ] Add distributed training support
- [ ] Create SLURM integration scripts

### Week 3: Model Implementation
- [ ] Migrate TTM to new structure
- [ ] Implement Chronos integration
- [ ] Add LoRA support

### Week 4: Experiment Framework
- [ ] Create experiment management system
- [ ] Implement evaluation pipeline
- [ ] Set up model registry

### Week 5: Integration and Testing
- [ ] End-to-end testing
- [ ] Performance optimization
- [ ] Documentation

## Entry Points

### Training Script
```bash
# scripts/training/train_model.py
python scripts/training/train_model.py \
    --model ttm \
    --experiment nocturnal_point_forecast \
    --config configs/experiments/nocturnal/ttm_point_forecast.yaml \
    --distributed multi_gpu
```

### SLURM Integration
```bash
# scripts/training/slurm/multi_gpu.sh
#!/bin/bash
#SBATCH --job-name=ttm_training
#SBATCH --gres=gpu:2
#SBATCH --mem=32G

python scripts/training/train_model.py \
    --model $MODEL_TYPE \
    --experiment $EXPERIMENT_TYPE \
    --config $CONFIG_PATH \
    --distributed multi_gpu
```

### Experiment Runner
```bash
# scripts/experiments/run_experiment.py
python scripts/experiments/run_experiment.py \
    --experiment_type nocturnal \
    --models ttm,chronos,timegpt \
    --datasets kaggle_bris_t1d,aleppo \
    --output_dir experiments/nocturnal_forecast/model_comparison_2025_11_17
```

## Benefits of This Reorganization

### 1. **Scalability**
- Modular architecture supports easy addition of new models
- Distributed training ready for large-scale experiments
- Configuration system scales to complex multi-model studies

### 2. **Reproducibility**
- Comprehensive experiment tracking
- Version control for data, models, and configurations
- Automated environment and dependency tracking

### 3. **Maintainability**
- Clear separation of concerns
- Reduced code duplication
- Consistent interfaces across models

### 4. **Research Productivity**
- Automated experiment pipeline
- Easy model comparison
- Standardized evaluation metrics

### 5. **Future-Proofing**
- MLflow integration ready
- Model serving preparation
- Extensible registry system

## Configuration Examples

### Model Configuration
```yaml
# configs/models/ttm/fine_tune.yaml
model_type: ttm
architecture:
  d_model: 512
  n_heads: 8
  n_layers: 6
  context_length: 512
  prediction_length: 96

training:
  strategy: fine_tuning
  base_model_path: "huggingface/ttm-1.5B"

lora:
  enabled: true
  rank: 16
  alpha: 32
  target_modules: ["q_proj", "v_proj"]

optimization:
  optimizer: adamw
  learning_rate: 1e-4
  weight_decay: 0.01
  warmup_steps: 1000
```

### Experiment Configuration
```yaml
# configs/experiments/nocturnal/ttm_vs_chronos.yaml
experiment_name: "TTM vs Chronos Nocturnal Forecasting"
experiment_type: nocturnal

models:
  - type: ttm
    config: !include ../../models/ttm/fine_tune.yaml
  - type: chronos
    config: !include ../../models/chronos/base.yaml

data:
  datasets:
    - kaggle_bris_t1d
    - aleppo
  splits:
    temporal_holdout: 0.2  # Last 20% of data
    patient_holdout: 0.15  # 15% of patients

training:
  distributed: true
  gpu_strategy: ddp
  max_epochs: 50
  early_stopping:
    patience: 10
    metric: val_rmse

evaluation:
  metrics:
    - rmse
    - mae
    - mape
    - clarke_error_grid
  test_sets:
    - temporal_holdout
    - patient_holdout
```

This reorganization plan provides a solid foundation for scalable TSFM research while maintaining the flexibility needed for academic experimentation. The structure follows modern ML engineering practices and can easily be extended as the project grows.
