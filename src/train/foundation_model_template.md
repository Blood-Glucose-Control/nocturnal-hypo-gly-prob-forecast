# Time Series Foundation Model (TSFM) Training Pipeline Template

## Overview

This document provides a standardized template for organizing **Time Series Foundation Model** training pipelines in the nocturnal project. This infrastructure is designed specifically for modern foundation models (TTM, Chronos, TimesFM, etc.) and is completely separate from the legacy traditional ML benchmarking system.

**Key Principles:**
- Use existing `src/data/` infrastructure via `CacheManager`
- Focus on TSFM-specific training logic only  
- Independent evaluation system (separate from legacy `src/eval/`)
- Organized output structure for multiple foundation models and experiments
- Industry-standard modular architecture

---

## Standard Directory Structure Template

```
src/train/{model_name}/
├── __init__.py                   # Public API exports
├── core/
│   ├── __init__.py
│   ├── trainer.py               # {ModelName}Trainer class
│   ├── model_factory.py         # Model creation and configuration
│   ├── pipeline.py              # End-to-end training pipeline
│   └── scheduler.py             # Learning rate schedulers (if custom)
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py               # TSFM-specific metrics and callbacks
│   ├── evaluator.py             # Evaluation pipeline (separate from legacy)
│   └── visualization.py         # Result visualization (optional)
├── losses/
│   ├── __init__.py
│   └── custom.py                # Custom loss functions
├── config/
│   ├── __init__.py
│   ├── manager.py               # Configuration loading and validation
│   ├── defaults.py              # Default configurations
│   └── schema.py                # Configuration validation schema
├── utils/
│   ├── __init__.py
│   ├── logging.py               # TSFM-specific logging utilities
│   ├── checkpointing.py         # Model checkpointing
│   ├── preprocessing.py         # TSFM-specific preprocessing (uses src/data/)
│   └── helpers.py               # Model-specific helper functions
└── cli/
    ├── __init__.py
    ├── runner.py                # Main CLI entry point
    └── commands.py              # CLI subcommands
```

---

## Proposed Output Structure for Multiple TSFMs

To handle multiple time series foundation models with multiple experiments each, we propose a new organized output structure:

```
tsfm_results/
├── models/                          # Trained model artifacts
│   ├── ttm/
│   │   ├── kaggle_experiment_1/
│   │   │   ├── 2025-11-12_14-30-45/     # Timestamped runs
│   │   │   │   ├── model/                # Model checkpoints
│   │   │   │   ├── logs/                 # Training logs
│   │   │   │   └── metrics.json          # Training metrics
│   │   │   └── best_model -> 2025-11-12_14-30-45/  # Symlink to best
│   │   ├── aleppo_experiment_1/
│   │   └── production_v1/               # Production-ready models
│   ├── chronos/
│   │   ├── fine_tune_experiment_1/
│   │   └── ...
│   └── timesfm/
│       └── ...
├── results/                         # Experiment results and analysis
│   ├── ttm/
│   │   ├── kaggle_experiment_1/
│   │   │   ├── evaluation_report.json
│   │   │   ├── comparison_plots/
│   │   │   └── run_history.csv
│   │   └── ...
│   └── chronos/
│       └── ...
└── scripts/                         # TSFM-specific experiment scripts
    ├── ttm/
    │   ├── kaggle_fine_tune.sh
    │   ├── aleppo_baseline.sh
    │   └── production_deploy.py
    ├── chronos/
    └── shared/                      # Common utilities
        ├── slurm_templates/
        └── experiment_tracking.py
```

**Key Benefits:**
- **Multi-model support**: Each TSFM gets its own namespace
- **Experiment organization**: Clear separation of different experimental setups
- **Production tracking**: Dedicated space for production-ready models
- **Script organization**: TSFM-specific orchestration separate from legacy scripts
- **Best model tracking**: Symlinks point to best performing runs

---

## Data Integration with Existing Infrastructure

### Using CacheManager for Data Loading

```python
# Example: TTM data loading using existing infrastructure
from src.data.cache_manager import get_cache_manager

class TTMDataLoader:
    def __init__(self, dataset_name: str):
        self.cache_manager = get_cache_manager()
        self.dataset_name = dataset_name
    
    def load_processed_data(self) -> dict[str, pd.DataFrame]:
        """Load processed data (ready for imputation/scaling)"""
        return self.cache_manager.load_full_processed_data(self.dataset_name)
    
    def prepare_for_training(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Apply TSFM-specific preprocessing (imputation, scaling, etc.)"""
        # Use existing preprocessing utilities from src/data/
        from src.tuning.benchmark import impute_missing_values
        from src.utils.time_series_helper import get_interval_minutes
        
        # Apply preprocessing and return training-ready data
        pass
```

---

## Template Implementation Guide

### 1. Core Trainer Class Pattern

**File: `src/train/{model_name}/core/trainer.py`**

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from transformers import Trainer, TrainingArguments

class BaseFoundationTrainer(ABC):
    """Base class for all foundation model trainers."""

    def __init__(self, config: 'ModelConfig'):
        self.config = config
        self.data_loader = self._create_data_loader()
        self.model_factory = self._create_model_factory()
        self.evaluator = self._create_evaluator()
        self.logger = self._create_logger()

    @abstractmethod
    def _create_data_loader(self):
        """Create appropriate data loader for this model type."""
        pass

    @abstractmethod
    def _create_model_factory(self):
        """Create model factory for this architecture."""
        pass

    @abstractmethod
    def _create_evaluator(self):
        """Create evaluator for this model type."""
        pass

    def _create_logger(self):
        """Create logger (can be overridden for custom logging)."""
        from ..utils.logging import create_logger
        return create_logger(self.config.logging)

    def train(self) -> Dict[str, Any]:
        """Main training orchestration method."""
        self.logger.info(f"Starting {self.__class__.__name__} training")

        # Load and prepare data
        data = self.data_loader.load(self.config.data)

        # Create model
        model = self.model_factory.create_model(self.config.model)

        # Create trainer
        hf_trainer = self._create_hf_trainer(model, data)

        # Training
        self._pre_training_hook(hf_trainer, data)
        training_result = self._execute_training(hf_trainer)
        self._post_training_hook(hf_trainer, training_result)

        # Evaluation
        metrics = self.evaluator.evaluate(hf_trainer, data)

        # Save results
        self._save_results(metrics, training_result)

        return metrics

    @abstractmethod
    def _create_hf_trainer(self, model, data) -> Trainer:
        """Create HuggingFace trainer with model-specific configuration."""
        pass

    def _pre_training_hook(self, trainer: Trainer, data: Any) -> None:
        """Hook called before training starts."""
        pass

    def _execute_training(self, trainer: Trainer) -> Any:
        """Execute the actual training."""
        if self.config.training.resume_from_checkpoint:
            return trainer.train(resume_from_checkpoint=True)
        else:
            return trainer.train()

    def _post_training_hook(self, trainer: Trainer, result: Any) -> None:
        """Hook called after training completes."""
        pass

    def _save_results(self, metrics: Dict[str, Any], training_result: Any) -> None:
        """Save training results and metrics."""
        # Implementation depends on output configuration
        pass

# Concrete implementation for specific model
class {ModelName}Trainer(BaseFoundationTrainer):
    """Trainer for {ModelName} architecture."""

    def _create_data_loader(self):
        from ..data.loaders import {ModelName}DataLoader
        return {ModelName}DataLoader(self.config.data)

    def _create_model_factory(self):
        from ..core.model_factory import {ModelName}ModelFactory
        return {ModelName}ModelFactory()

    def _create_evaluator(self):
        from ..evaluation.evaluator import {ModelName}Evaluator
        return {ModelName}Evaluator(self.config.evaluation)

    def _create_hf_trainer(self, model, data) -> Trainer:
        # Model-specific trainer creation
        training_args = TrainingArguments(
            # Standard HF arguments based on config
            **self.config.training.to_hf_args()
        )

        return Trainer(
            model=model,
            args=training_args,
            train_dataset=data.train_dataset,
            eval_dataset=data.eval_dataset,
            compute_metrics=self.evaluator.compute_metrics,
            callbacks=self._create_callbacks(),
            optimizers=self._create_optimizers(model),
        )
```

### 2. Configuration Management Pattern

**File: `src/train/{model_name}/config/schema.py`**

```python
from dataclasses import dataclass
from typing import List, Optional, Union, Any, Dict
from omegaconf import OmegaConf

@dataclass
class BaseModelConfig:
    """Base configuration for all foundation models."""
    model_type: str
    model_path: str
    device: str = "auto"
    precision: str = "fp16"  # fp16, fp32, bf16

    def validate(self):
        """Validate configuration parameters."""
        if self.model_type not in self.supported_model_types():
            raise ValueError(f"Unsupported model type: {self.model_type}")

@dataclass
class BaseDataConfig:
    """Base data configuration."""
    source_name: str
    loader_type: str = "default"
    batch_size: int = 32
    num_workers: int = 4
    data_split: List[float] = None

    def __post_init__(self):
        if self.data_split is None:
            self.data_split = [0.8, 0.1, 0.1]  # train, val, test

@dataclass
class BaseTrainingConfig:
    """Base training configuration."""
    num_epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    save_strategy: str = "steps"
    save_steps: int = 500
    eval_strategy: str = "steps"
    eval_steps: int = 500
    logging_steps: int = 100
    resume_from_checkpoint: bool = False
    gradient_checkpointing: bool = False

    def to_hf_args(self) -> Dict[str, Any]:
        """Convert to HuggingFace TrainingArguments format."""
        return {
            "num_train_epochs": self.num_epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_steps": self.warmup_steps,
            "save_strategy": self.save_strategy,
            "save_steps": self.save_steps,
            "eval_strategy": self.eval_strategy,
            "eval_steps": self.eval_steps,
            "logging_steps": self.logging_steps,
            "gradient_checkpointing": self.gradient_checkpointing,
        }

@dataclass
class BaseEvaluationConfig:
    """Base evaluation configuration."""
    metrics: List[str] = None
    eval_batch_size: Optional[int] = None
    save_predictions: bool = True

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = ["loss"]

@dataclass
class BaseLoggingConfig:
    """Base logging configuration."""
    level: str = "INFO"
    debug_mode: bool = False
    log_file: Optional[str] = None
    wandb_project: Optional[str] = None
    mlflow_experiment: Optional[str] = None

@dataclass
class BaseOutputConfig:
    """Base output configuration."""
    save_dir: str = "./outputs"
    experiment_name: Optional[str] = None
    save_metrics: bool = True
    save_model: bool = True
    metrics_format: str = "json"  # json, yaml, csv

# Model-specific configuration
@dataclass
class {ModelName}Config:
    """Configuration for {ModelName} training."""
    model: BaseModelConfig
    data: BaseDataConfig  # Or {ModelName}DataConfig if custom needed
    training: BaseTrainingConfig
    evaluation: BaseEvaluationConfig
    logging: BaseLoggingConfig
    output: BaseOutputConfig

    @classmethod
    def from_yaml(cls, yaml_path: str) -> '{ModelName}Config':
        """Load configuration from YAML file."""
        conf = OmegaConf.load(yaml_path)
        return cls(**conf)

    def validate(self):
        """Validate entire configuration."""
        self.model.validate()
        # Add model-specific validation
```

### 3. Data Integration Pattern (Using Existing Infrastructure)

**File: `src/train/{model_name}/utils/preprocessing.py`**

```python
from typing import Dict, List, Tuple
import pandas as pd
from src.data.cache_manager import get_cache_manager
from src.tuning.benchmark import impute_missing_values

class TSFMDataPreprocessor:
    """TSFM-specific data preprocessing using existing infrastructure."""

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.cache_manager = get_cache_manager()

    def load_and_prepare(
        self, 
        x_features: List[str],
        y_features: List[str],
        imputation_config: Dict[str, str] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Load processed data from cache and apply TSFM-specific preprocessing.
        
        Uses existing src/data/ infrastructure:
        - CacheManager for data loading
        - Existing imputation utilities
        - Existing preprocessing functions
        """
        # Load processed data (ready for imputation/scaling)
        processed_data = self.cache_manager.load_full_processed_data(self.dataset_name)
        
        if processed_data is None:
            raise ValueError(f"No processed data found for {self.dataset_name}")
        
        # Apply imputation using existing utilities
        if imputation_config:
            for patient_id, df in processed_data.items():
                processed_data[patient_id] = impute_missing_values(
                    df,
                    columns=x_features + y_features,
                    **imputation_config
                )
        
        return processed_data

    def prepare_for_tsfm_training(
        self, 
        data: Dict[str, pd.DataFrame],
        **model_specific_params
    ) -> pd.DataFrame:
        """Apply TSFM-specific transformations (feature reduction, etc.)"""
    def _validate_data(self, data: Any) -> bool:
        """Validate loaded data meets requirements."""
        pass

class DataContainer:
    """Container for train/val/test datasets."""

    def __init__(self,
                 train_dataset: Dataset,
                 val_dataset: Dataset,
                 test_dataset: Dataset,
                 metadata: Dict[str, Any] = None):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.metadata = metadata or {}

    @property
    def num_samples(self) -> Dict[str, int]:
        return {
            "train": len(self.train_dataset),
            "val": len(self.val_dataset),
            "test": len(self.test_dataset)
        }

# Model-specific implementation
class {ModelName}DataLoader(BaseDataLoader):
    """Data loader for {ModelName} architecture."""

    def load(self, **kwargs) -> DataContainer:
        # Load raw data
        raw_data = self._load_raw_data()

        # Preprocess
        processed_data = self._preprocess_data(raw_data)

        # Split
        train_data, val_data, test_data = self._split_data(processed_data)

        # Create datasets
        train_dataset = self._create_dataset(train_data)
        val_dataset = self._create_dataset(val_data)
        test_dataset = self._create_dataset(test_data)

        # Validate
        assert self._validate_data(train_dataset)

        return DataContainer(train_dataset, val_dataset, test_dataset)

    def _load_raw_data(self):
        """Load raw data based on source_name."""
        # Implementation specific to model and data source
        pass

    def _preprocess_data(self, raw_data):
        """Apply model-specific preprocessing."""
        from ..data.preprocessing import {ModelName}Preprocessor
        preprocessor = {ModelName}Preprocessor(self.config)
        return preprocessor.process(raw_data)

    def _split_data(self, data):
        """Split data according to config."""
        # Standard splitting logic
        pass

    def _create_dataset(self, data) -> Dataset:
        """Create PyTorch Dataset."""
        from ..data.datasets import {ModelName}Dataset
        return {ModelName}Dataset(data, self.config)

    def _validate_data(self, dataset: Dataset) -> bool:
        """Validate dataset meets model requirements."""
        # Model-specific validation
        return True
```

### 4. TSFM Evaluation Pattern (Separate from Legacy)

**Note: This evaluation system is completely separate from the legacy `src/eval/` system and traditional ML benchmarking. Focus on TSFM-specific metrics and evaluation needs.**

**File: `src/train/{model_name}/evaluation/metrics.py`**

```python
from typing import Dict, Any, List, Callable
import numpy as np
import time
from transformers import TrainerCallback
from sklearn.metrics import mean_squared_error, mean_absolute_error

class TSFMMetricsCallback(TrainerCallback):
    """TSFM-specific callback for metrics collection."""

    def __init__(self, metric_functions: Dict[str, Callable] = None):
        self.metric_functions = metric_functions or {}
        self.metrics_history = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when logging occurs."""
        if logs is not None:
            # Add custom metrics to logs
            for name, func in self.metric_functions.items():
                if f"eval_{name}" not in logs:
                    # Only add if not already computed
                    continue

            # Add metadata
            logs["timestamp"] = time.time()
            logs["step"] = state.global_step
            logs["epoch"] = state.epoch

            self.metrics_history.append(logs.copy())

        return control

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all collected metrics."""
        if not self.metrics_history:
            return {}

        return {
            "final_metrics": self.metrics_history[-1],
            "best_metrics": self._get_best_metrics(),
            "metrics_history": self.metrics_history,
            "training_summary": self._get_training_summary()
        }

    def _get_best_metrics(self) -> Dict[str, Any]:
        """Get best metrics across training."""
        # Implementation depends on what "best" means for each metric
        pass

    def _get_training_summary(self) -> Dict[str, Any]:
        """Get training summary statistics."""
        pass

def compute_base_metrics(eval_pred) -> Dict[str, float]:
    """Base metrics computation function."""
    predictions, labels = eval_pred

    # Ensure correct shapes
    if len(predictions.shape) > 2:
        predictions = predictions.reshape(-1, predictions.shape[-1])
        labels = labels.reshape(-1, labels.shape[-1])

    # Convert to numpy if needed
    if hasattr(predictions, 'cpu'):
        predictions = predictions.cpu().numpy()
    if hasattr(labels, 'cpu'):
        labels = labels.cpu().numpy()

    # Base metrics that work for most models
    metrics = {}

    try:
        # Mean squared error (for regression tasks)
        mse = np.mean((predictions - labels) ** 2)
        metrics["mse"] = float(mse)
        metrics["rmse"] = float(np.sqrt(mse))
    except:
        pass

    try:
        # Mean absolute error
        mae = np.mean(np.abs(predictions - labels))
        metrics["mae"] = float(mae)
    except:
        pass

    return metrics

# Model-specific metrics
class {ModelName}MetricsCallback(BaseMetricsCallback):
    """Metrics callback for {ModelName}."""

    def __init__(self):
        super().__init__(metric_functions={
            "custom_metric_1": self._compute_custom_metric_1,
            "custom_metric_2": self._compute_custom_metric_2,
        })

    def _compute_custom_metric_1(self, predictions, labels):
        """Compute model-specific metric 1."""
        # Implementation specific to model requirements
        pass

def compute_{model_name}_metrics(eval_pred) -> Dict[str, float]:
    """Compute {ModelName}-specific metrics."""
    base_metrics = compute_base_metrics(eval_pred)

    predictions, labels = eval_pred

    # Add model-specific metrics
    custom_metrics = {
        "model_specific_metric": 0.0,  # Replace with actual computation
    }

    return {**base_metrics, **custom_metrics}
```

### 5. CLI Interface Pattern

**File: `src/train/{model_name}/cli/runner.py`**

```python
import argparse
import sys
from pathlib import Path
from typing import Optional

def create_base_parser() -> argparse.ArgumentParser:
    """Create base argument parser for all models."""
    parser = argparse.ArgumentParser(
        description=f"{ModelName} Training Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        help="Directory to save run outputs and logs"
    )

    # Model overrides
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument("--model-path", type=str, help="Model path or HF identifier")
    model_group.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], help="Device to use")

    # Training overrides
    train_group = parser.add_argument_group("Training Configuration")
    train_group.add_argument("--batch-size", type=int, help="Training batch size")
    train_group.add_argument("--learning-rate", type=float, help="Learning rate")
    train_group.add_argument("--num-epochs", type=int, help="Number of training epochs")
    train_group.add_argument("--resume", action="store_true", help="Resume from checkpoint")

    # Data overrides
    data_group = parser.add_argument_group("Data Configuration")
    data_group.add_argument("--data-source", type=str, help="Data source name")
    data_group.add_argument("--data-split", nargs="+", type=float, help="Train/val/test split")

    # Utilities
    util_group = parser.add_argument_group("Utilities")
    util_group.add_argument("--debug", action="store_true", help="Enable debug mode")
    util_group.add_argument("--dry-run", action="store_true", help="Dry run (validate config only)")
    util_group.add_argument("--validate-config", action="store_true", help="Validate configuration and exit")

    return parser

def apply_cli_overrides(config: '{ModelName}Config', args: argparse.Namespace) -> '{ModelName}Config':
    """Apply command-line overrides to configuration."""
    if args.model_path:
        config.model.model_path = args.model_path
    if args.device:
        config.model.device = args.device
    if args.batch_size:
        config.data.batch_size = args.batch_size
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    if args.num_epochs:
        config.training.num_epochs = args.num_epochs
    if args.resume:
        config.training.resume_from_checkpoint = True
    if args.data_source:
        config.data.source_name = args.data_source
    if args.data_split:
        config.data.data_split = args.data_split
    if args.debug:
        config.logging.debug_mode = True
        config.logging.level = "DEBUG"

    return config

def main():
    """Main CLI entry point."""
    parser = create_base_parser()
    args = parser.parse_args()

    # Load configuration
    if args.config:
        from ..config.manager import load_config
        config = load_config(args.config)
    else:
        from ..config.defaults import get_default_config
        config = get_default_config()

    # Apply CLI overrides
    config = apply_cli_overrides(config, args)

    # Validate configuration
    try:
        config.validate()
    except Exception as e:
        print(f"Configuration validation failed: {e}", file=sys.stderr)
        sys.exit(1)

    if args.validate_config:
        print("Configuration is valid!")
        sys.exit(0)

    if args.dry_run:
        print("Dry run mode - configuration loaded successfully")
        print(f"Would train {config.model.model_type} for {config.training.num_epochs} epochs")
        sys.exit(0)

    # Setup output directory
    if args.run_dir:
        output_dir = Path(args.run_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        config.output.save_dir = str(output_dir)

    # Start training
    from ..core.trainer import {ModelName}Trainer

    print(f"Starting {ModelName} training...")
    print(f"Model: {config.model.model_path}")
    print(f"Data: {config.data.source_name}")
    print(f"Epochs: {config.training.num_epochs}")

    trainer = {ModelName}Trainer(config)
    metrics = trainer.train()

    print("Training completed successfully!")
    print(f"Final metrics: {metrics.get('final_metrics', {})}")

    # Save metrics if requested
    if config.output.save_metrics and args.run_dir:
        import json
        metrics_file = output_dir / "training_metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to: {metrics_file}")

if __name__ == "__main__":
    main()
```

---

## Configuration Templates

### Default Configuration Template

**File: `configs/models/{model_name}_default.yaml`**

```yaml
# {ModelName} Default Configuration Template

model:
  model_type: "{model_name}"
  model_path: "path/to/pretrained/model"
  device: "auto"
  precision: "fp16"
  # Model-specific parameters
  context_length: 512
  hidden_size: 768

data:
  source_name: "default_dataset"
  loader_type: "default"
  batch_size: 32
  num_workers: 4
  data_split: [0.8, 0.1, 0.1]  # train, val, test
  # Model-specific data parameters
  max_sequence_length: 512
  preprocessing:
    normalize: true
    tokenize: true

training:
  num_epochs: 10
  learning_rate: 1e-4
  weight_decay: 0.01
  warmup_steps: 1000
  save_strategy: "steps"
  save_steps: 500
  eval_strategy: "steps"
  eval_steps: 500
  logging_steps: 100
  resume_from_checkpoint: false
  gradient_checkpointing: false
  # Model-specific training parameters
  freeze_backbone: false
  use_lora: false

evaluation:
  metrics: ["loss", "accuracy"]
  eval_batch_size: null  # Use same as training if null
  save_predictions: true

logging:
  level: "INFO"
  debug_mode: false
  log_file: null
  wandb_project: null
  mlflow_experiment: null

output:
  save_dir: "./outputs/{model_name}"
  experiment_name: null  # Auto-generated if null
  save_metrics: true
  save_model: true
  metrics_format: "json"
```

### Experiment Configuration Template

**File: `configs/experiments/{model_name}_experiment.yaml`**

```yaml
# {ModelName} Experiment Configuration Template
# Inherits from default configuration and overrides specific parameters

defaults:
  - models/{model_name}_default

# Override specific parameters for this experiment
model:
  model_path: "specific/model/for/experiment"

data:
  source_name: "experiment_dataset"
  batch_size: 64

training:
  num_epochs: 50
  learning_rate: 5e-5

evaluation:
  metrics: ["loss", "accuracy", "f1", "precision", "recall"]

logging:
  wandb_project: "{model_name}_experiments"

output:
  experiment_name: "{model_name}_baseline_experiment"
```

---

## Testing Template

### Unit Test Template

**File: `tests/test_{model_name}/test_trainer.py`**

```python
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from src.train.{model_name}.core.trainer import {ModelName}Trainer
from src.train.{model_name}.config.schema import {ModelName}Config

class Test{ModelName}Trainer:
    """Test suite for {ModelName}Trainer."""

    @pytest.fixture
    def sample_config(self):
        """Create a sample configuration for testing."""
        return {ModelName}Config(
            model=BaseModelConfig(
                model_type="{model_name}",
                model_path="test/model/path"
            ),
            data=BaseDataConfig(
                source_name="test_dataset",
                batch_size=2
            ),
            training=BaseTrainingConfig(
                num_epochs=1,
                learning_rate=1e-4
            ),
            evaluation=BaseEvaluationConfig(),
            logging=BaseLoggingConfig(),
            output=BaseOutputConfig()
        )

    @pytest.fixture
    def trainer(self, sample_config):
        """Create trainer instance for testing."""
        return {ModelName}Trainer(sample_config)

    def test_trainer_initialization(self, trainer):
        """Test trainer initializes correctly."""
        assert trainer.config is not None
        assert trainer.data_loader is not None
        assert trainer.model_factory is not None
        assert trainer.evaluator is not None

    @patch('src.train.{model_name}.data.loaders.{ModelName}DataLoader.load')
    @patch('src.train.{model_name}.core.model_factory.{ModelName}ModelFactory.create_model')
    def test_train_method(self, mock_create_model, mock_load_data, trainer):
        """Test the main training method."""
        # Mock data loading
        mock_data = Mock()
        mock_load_data.return_value = mock_data

        # Mock model creation
        mock_model = Mock()
        mock_create_model.return_value = mock_model

        # Mock HF trainer
        with patch.object(trainer, '_create_hf_trainer') as mock_create_trainer:
            mock_hf_trainer = Mock()
            mock_create_trainer.return_value = mock_hf_trainer

            with patch.object(trainer.evaluator, 'evaluate') as mock_evaluate:
                mock_evaluate.return_value = {"test_loss": 0.5}

                # Test training
                result = trainer.train()

                # Assertions
                mock_load_data.assert_called_once()
                mock_create_model.assert_called_once()
                mock_create_trainer.assert_called_once_with(mock_model, mock_data)
                mock_hf_trainer.train.assert_called_once()
                assert "test_loss" in result

# Integration test template
class TestIntegration{ModelName}:
    """Integration tests for {ModelName} pipeline."""

    @pytest.mark.slow
    def test_end_to_end_training(self, tmp_path):
        """Test complete training pipeline with small dataset."""
        # Create minimal config
        config = create_minimal_config(output_dir=tmp_path)

        # Run training
        trainer = {ModelName}Trainer(config)
        metrics = trainer.train()

        # Check outputs
        assert metrics is not None
        assert "final_metrics" in metrics
        assert (tmp_path / "training_metrics.json").exists()
```

---

## Implementation Checklist

When implementing a new foundation model architecture, follow this checklist:

### Phase 1: Setup Structure
- [ ] Create directory structure following template
- [ ] Copy template files and rename appropriately
- [ ] Update all `{model_name}` and `{ModelName}` placeholders
- [ ] Create `__init__.py` files with proper exports

### Phase 2: Configuration
- [ ] Define model-specific configuration schema
- [ ] Create default configuration YAML
- [ ] Implement configuration validation
- [ ] Test configuration loading and validation

### Phase 3: Data Pipeline
- [ ] Implement model-specific data loader
- [ ] Create preprocessing functions
- [ ] Define dataset class
- [ ] Test data loading and preprocessing

### Phase 4: Model Integration
- [ ] Implement model factory
- [ ] Define model-specific trainer class
- [ ] Configure HuggingFace trainer integration
- [ ] Test model creation and training setup

### Phase 5: Evaluation
- [ ] Implement model-specific metrics
- [ ] Create evaluation pipeline
- [ ] Add custom callbacks if needed
- [ ] Test metric computation

### Phase 6: CLI and Integration
- [ ] Implement CLI interface
- [ ] Test command-line usage
- [ ] Create integration tests
- [ ] Document usage examples

### Phase 7: Testing and Documentation
- [ ] Write comprehensive unit tests
- [ ] Add integration tests
- [ ] Create documentation
- [ ] Validate against existing models

### Phase 8: Validation
- [ ] Run end-to-end training test
- [ ] Compare with reference implementation
- [ ] Performance benchmarking
- [ ] Code review and cleanup

---

## Best Practices and Guidelines

### 1. Code Organization
- **Single Responsibility**: Each module should have one clear purpose
- **Dependency Injection**: Use constructor injection for dependencies
- **Interface Segregation**: Create focused interfaces, not monolithic ones
- **Configuration-Driven**: All behavior should be configurable

### 2. Error Handling
- **Fail Fast**: Validate configurations and inputs early
- **Meaningful Errors**: Provide clear, actionable error messages
- **Graceful Degradation**: Handle missing optional features gracefully
- **Logging**: Log important decisions and state changes

### 3. Testing Strategy
- **Unit Tests**: Test each component in isolation
- **Integration Tests**: Test component interactions
- **End-to-End Tests**: Test complete workflows
- **Performance Tests**: Benchmark critical paths

### 4. Documentation
- **API Documentation**: Document all public interfaces
- **Configuration Documentation**: Explain all configuration options
- **Usage Examples**: Provide working examples
- **Migration Guides**: Help users upgrade between versions

### 5. Performance Considerations
- **Memory Efficiency**: Minimize memory usage, especially for large models
- **Lazy Loading**: Load components only when needed
- **Caching**: Cache expensive operations when appropriate
- **Profiling**: Profile critical paths regularly

---

## Migration from Existing Code

When migrating existing model training code to this template:

### 1. Identify Components
- Extract data loading logic
- Identify model-specific code
- Separate training orchestration
- Extract evaluation metrics

### 2. Map to Template Structure
- Move data code to `data/` modules
- Move model code to `core/model_factory.py`
- Move training logic to `core/trainer.py`
- Move metrics to `evaluation/`

### 3. Gradual Migration
- Start with configuration management
- Migrate data pipeline first
- Then move training logic
- Finally add CLI and testing

### 4. Maintain Compatibility
- Create wrapper functions for old interfaces
- Add deprecation warnings
- Provide migration documentation
- Support both old and new interfaces temporarily

---

## Conclusion

This template provides a standardized, scalable foundation for implementing training pipelines for any foundation model architecture. By following this structure, new models will be:

- **Consistent** with existing implementations
- **Maintainable** through clear separation of concerns
- **Testable** with comprehensive testing support
- **Configurable** with YAML-based configuration management
- **Professional** following industry best practices

Use this template as a starting point and adapt it to the specific needs of each model architecture while maintaining the overall organizational principles.
