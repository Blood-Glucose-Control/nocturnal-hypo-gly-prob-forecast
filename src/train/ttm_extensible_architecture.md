# TTM Extensible Architecture for MLflow & Hydra

This document outlines how to design the TTM architecture to seamlessly integrate MLflow and Hydra later.

---

## Core Design Principles

### 1. Abstract Interfaces
Use abstract base classes that can be swapped out without changing core logic.

### 2. Dependency Injection
Components receive their dependencies (logger, config manager) rather than creating them internally.

### 3. Plugin-Style Architecture
Configuration and experiment tracking are pluggable systems.

---

## Current Simple Implementation → Future Extensions

### Configuration Management Evolution

#### Phase 1: Simple YAML (Current)
```python
# config/manager.py
class SimpleConfigManager:
    def load_config(self, path: str) -> dict:
        with open(path, 'r') as f:
            return yaml.safe_load(f)

# Usage
config_manager = SimpleConfigManager()
config = config_manager.load_config("config.yaml")
```

#### Phase 2: Hydra Integration (Future)
```python
# config/manager.py
class HydraConfigManager:
    @hydra.main(config_path="configs", config_name="default")
    def load_config(self, cfg: DictConfig) -> dict:
        return OmegaConf.to_container(cfg, resolve=True)

# Same usage - just swap out the manager!
config_manager = HydraConfigManager()
config = config_manager.load_config("config.yaml")
```

### Experiment Tracking Evolution

#### Phase 1: Simple Logging (Current)
```python
# utils/logging.py
class SimpleExperimentLogger:
    def __init__(self, experiment_dir: str):
        self.experiment_dir = Path(experiment_dir)
        
    def log_metric(self, key: str, value: float, step: int = None):
        # Write to JSON file
        pass
        
    def log_params(self, params: dict):
        # Write to JSON file  
        pass
```

#### Phase 2: MLflow Integration (Future)
```python
# utils/logging.py  
class MLflowExperimentLogger:
    def __init__(self, experiment_name: str):
        mlflow.set_experiment(experiment_name)
        
    def log_metric(self, key: str, value: float, step: int = None):
        mlflow.log_metric(key, value, step)
        
    def log_params(self, params: dict):
        mlflow.log_params(params)
```

---

## Extensible Directory Structure

```
src/train/ttm/
├── __init__.py                   # Public API exports
├── core/
│   ├── __init__.py
│   ├── trainer.py               # Uses abstract interfaces
│   ├── model_factory.py         
│   └── adapters.py              # LoRA implementation
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py               
│   └── callbacks.py             
├── config/
│   ├── __init__.py
│   ├── interfaces.py            # Abstract config interface
│   ├── simple_manager.py        # Simple YAML implementation
│   ├── hydra_manager.py         # Future: Hydra implementation  
│   └── defaults.py              
├── experiment/
│   ├── __init__.py
│   ├── interfaces.py            # Abstract experiment tracking interface
│   ├── simple_logger.py         # Simple file-based logging
│   ├── mlflow_logger.py         # Future: MLflow implementation
│   └── wandb_logger.py          # Future: Wandb implementation
├── utils/
│   ├── __init__.py
│   ├── preprocessing.py         
│   └── registry.py              # Model versioning (works with any logger)
└── cli/
    ├── __init__.py
    └── runner.py                # CLI that works with any config system
```

---

## Abstract Interfaces

### Configuration Interface

```python
# config/interfaces.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    """Standardized training configuration"""
    batch_size: int
    learning_rate: float
    num_epochs: int
    mixed_precision: bool = True
    lora_rank: Optional[int] = None
    
@dataclass  
class TTMExperimentConfig:
    """Complete TTM experiment configuration"""
    model_path: str
    data_source: str
    training: TrainingConfig
    experiment_name: str
    output_dir: str

class ConfigManager(ABC):
    """Abstract configuration manager"""
    
    @abstractmethod
    def load_config(self, config_path: str) -> TTMExperimentConfig:
        """Load configuration from file/system"""
        pass
        
    @abstractmethod
    def validate_config(self, config: TTMExperimentConfig) -> bool:
        """Validate configuration is complete and correct"""
        pass
        
    @abstractmethod
    def save_config(self, config: TTMExperimentConfig, output_path: str) -> None:
        """Save configuration (for reproducibility)"""
        pass
```

### Experiment Tracking Interface

```python
# experiment/interfaces.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pathlib import Path

class ExperimentTracker(ABC):
    """Abstract experiment tracking interface"""
    
    @abstractmethod
    def start_experiment(self, experiment_name: str, config: dict) -> str:
        """Start new experiment, return run_id"""
        pass
        
    @abstractmethod
    def log_params(self, params: dict) -> None:
        """Log hyperparameters"""
        pass
        
    @abstractmethod
    def log_metrics(self, metrics: dict, step: Optional[int] = None) -> None:
        """Log training metrics"""
        pass
        
    @abstractmethod
    def log_model(self, model_path: Path, model_name: str) -> None:
        """Log trained model"""
        pass
        
    @abstractmethod
    def end_experiment(self) -> str:
        """End experiment, return final run_id"""
        pass
```

---

## Implementation: Phase 1 (Simple)

### Simple Configuration Manager

```python
# config/simple_manager.py
import yaml
from pathlib import Path
from .interfaces import ConfigManager, TTMExperimentConfig, TrainingConfig

class SimpleConfigManager(ConfigManager):
    """Simple YAML-based configuration manager"""
    
    def load_config(self, config_path: str) -> TTMExperimentConfig:
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)
            
        training_config = TrainingConfig(**data['training'])
        
        return TTMExperimentConfig(
            model_path=data['model']['path'],
            data_source=data['data']['source_name'],
            training=training_config,
            experiment_name=data.get('experiment_name', 'default'),
            output_dir=data.get('output_dir', './experiments')
        )
    
    def validate_config(self, config: TTMExperimentConfig) -> bool:
        # Simple validation
        return all([
            config.model_path,
            config.data_source,
            config.training.batch_size > 0,
            config.training.num_epochs > 0
        ])
    
    def save_config(self, config: TTMExperimentConfig, output_path: str) -> None:
        # Convert back to dict and save as YAML
        config_dict = {
            'model': {'path': config.model_path},
            'data': {'source_name': config.data_source},
            'training': {
                'batch_size': config.training.batch_size,
                'learning_rate': config.training.learning_rate,
                'num_epochs': config.training.num_epochs,
                'mixed_precision': config.training.mixed_precision,
                'lora_rank': config.training.lora_rank
            },
            'experiment_name': config.experiment_name,
            'output_dir': config.output_dir
        }
        
        with open(output_path, 'w') as f:
            yaml.dump(config_dict, f)
```

### Simple Experiment Tracker

```python
# experiment/simple_logger.py
import json
from pathlib import Path
from datetime import datetime
from .interfaces import ExperimentTracker

class SimpleExperimentTracker(ExperimentTracker):
    """Simple file-based experiment tracking"""
    
    def __init__(self, base_dir: str = "./experiments"):
        self.base_dir = Path(base_dir)
        self.current_run_dir = None
        self.run_id = None
        
    def start_experiment(self, experiment_name: str, config: dict) -> str:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_id = f"{experiment_name}_{timestamp}"
        
        self.current_run_dir = self.base_dir / experiment_name / f"run_{timestamp}"
        self.current_run_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        config_path = self.current_run_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
            
        return self.run_id
    
    def log_params(self, params: dict) -> None:
        params_path = self.current_run_dir / "params.json"
        with open(params_path, 'w') as f:
            json.dump(params, f, indent=2)
    
    def log_metrics(self, metrics: dict, step: Optional[int] = None) -> None:
        metrics_path = self.current_run_dir / "metrics.jsonl"
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'step': step,
            **metrics
        }
        
        with open(metrics_path, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def log_model(self, model_path: Path, model_name: str) -> None:
        model_info = {
            'model_name': model_name,
            'model_path': str(model_path),
            'logged_at': datetime.now().isoformat()
        }
        
        model_info_path = self.current_run_dir / "model_info.json"
        with open(model_info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
    
    def end_experiment(self) -> str:
        summary = {
            'run_id': self.run_id,
            'ended_at': datetime.now().isoformat(),
            'status': 'completed'
        }
        
        summary_path = self.current_run_dir / "run_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
            
        return self.run_id
```

---

## Core Training Logic (Interface-Based)

### TTM Trainer with Dependency Injection

```python
# core/trainer.py
from typing import Optional
from ..config.interfaces import ConfigManager, TTMExperimentConfig
from ..experiment.interfaces import ExperimentTracker
from .adapters import add_lora_adapters

class TTMTrainer:
    """Main TTM trainer that works with any config manager and experiment tracker"""
    
    def __init__(
        self,
        config_manager: ConfigManager,
        experiment_tracker: ExperimentTracker,
        config_path: str
    ):
        self.config_manager = config_manager
        self.experiment_tracker = experiment_tracker
        
        # Load and validate configuration
        self.config = config_manager.load_config(config_path)
        if not config_manager.validate_config(self.config):
            raise ValueError("Invalid configuration")
    
    def train(self) -> dict:
        """Train TTM model"""
        
        # Start experiment tracking
        run_id = self.experiment_tracker.start_experiment(
            self.config.experiment_name,
            self._config_to_dict()
        )
        
        try:
            # Log training parameters
            self.experiment_tracker.log_params(self._get_training_params())
            
            # Load model and apply LoRA if configured
            model = self._load_model()
            if self.config.training.lora_rank:
                model = add_lora_adapters(model, rank=self.config.training.lora_rank)
                self.experiment_tracker.log_params({
                    'lora_enabled': True,
                    'lora_rank': self.config.training.lora_rank
                })
            
            # Load data
            train_data, eval_data = self._load_data()
            
            # Create trainer
            trainer = self._create_hf_trainer(model, train_data, eval_data)
            
            # Training loop with logging
            trainer.train()
            
            # Final evaluation
            metrics = trainer.evaluate()
            self.experiment_tracker.log_metrics(metrics)
            
            # Save model
            model_path = self._save_model(trainer.model)
            self.experiment_tracker.log_model(model_path, "ttm_finetuned")
            
            return metrics
            
        finally:
            # Always end experiment tracking
            final_run_id = self.experiment_tracker.end_experiment()
            return {'run_id': final_run_id, 'metrics': metrics}
    
    def _config_to_dict(self) -> dict:
        """Convert config to dict for logging"""
        return {
            'model_path': self.config.model_path,
            'data_source': self.config.data_source,
            'batch_size': self.config.training.batch_size,
            'learning_rate': self.config.training.learning_rate,
            'num_epochs': self.config.training.num_epochs,
            'lora_rank': self.config.training.lora_rank
        }
    
    def _get_training_params(self) -> dict:
        """Extract hyperparameters for logging"""
        return {
            'learning_rate': self.config.training.learning_rate,
            'batch_size': self.config.training.batch_size,
            'num_epochs': self.config.training.num_epochs,
            'mixed_precision': self.config.training.mixed_precision,
        }
    
    # ... other helper methods
```

---

## CLI That Works with Any System

```python
# cli/runner.py
import argparse
from pathlib import Path
from ..config.simple_manager import SimpleConfigManager
from ..experiment.simple_logger import SimpleExperimentTracker
from ..core.trainer import TTMTrainer

def main():
    parser = argparse.ArgumentParser(description="TTM Training CLI")
    parser.add_argument("--config", required=True, help="Path to configuration file")
    parser.add_argument("--experiment-dir", default="./experiments", 
                       help="Base directory for experiments")
    
    args = parser.parse_args()
    
    # Use simple implementations (easily swappable later)
    config_manager = SimpleConfigManager()
    experiment_tracker = SimpleExperimentTracker(args.experiment_dir)
    
    # Create trainer
    trainer = TTMTrainer(
        config_manager=config_manager,
        experiment_tracker=experiment_tracker,
        config_path=args.config
    )
    
    # Train
    results = trainer.train()
    print(f"Training completed. Run ID: {results['run_id']}")

if __name__ == "__main__":
    main()
```

---

## Future Extensions (Phase 2)

### MLflow Integration

```python
# experiment/mlflow_logger.py
import mlflow
from .interfaces import ExperimentTracker

class MLflowExperimentTracker(ExperimentTracker):
    """MLflow-based experiment tracking"""
    
    def __init__(self, tracking_uri: str = None):
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        self.run = None
        
    def start_experiment(self, experiment_name: str, config: dict) -> str:
        mlflow.set_experiment(experiment_name)
        self.run = mlflow.start_run()
        return self.run.info.run_id
    
    def log_params(self, params: dict) -> None:
        mlflow.log_params(params)
    
    def log_metrics(self, metrics: dict, step: Optional[int] = None) -> None:
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step)
    
    def log_model(self, model_path: Path, model_name: str) -> None:
        mlflow.pytorch.log_model(model_path, model_name)
    
    def end_experiment(self) -> str:
        run_id = self.run.info.run_id
        mlflow.end_run()
        return run_id
```

### Hydra Integration

```python
# config/hydra_manager.py
import hydra
from omegaconf import DictConfig, OmegaConf
from .interfaces import ConfigManager, TTMExperimentConfig, TrainingConfig

class HydraConfigManager(ConfigManager):
    """Hydra-based configuration manager"""
    
    def load_config(self, config_path: str = None) -> TTMExperimentConfig:
        # If used with @hydra.main decorator, cfg is automatically loaded
        # This is more complex but allows for composition and overrides
        pass
    
    @hydra.main(config_path="../configs", config_name="ttm_default")  
    def load_with_hydra(self, cfg: DictConfig) -> TTMExperimentConfig:
        """Load configuration using Hydra"""
        config_dict = OmegaConf.to_container(cfg, resolve=True)
        
        training_config = TrainingConfig(**config_dict['training'])
        
        return TTMExperimentConfig(
            model_path=config_dict['model']['path'],
            data_source=config_dict['data']['source_name'],
            training=training_config,
            experiment_name=config_dict.get('experiment_name', 'default'),
            output_dir=config_dict.get('output_dir', './experiments')
        )
```

---

## Easy Migration Path

### Step 1: Start with Simple Implementation
```python
# Simple setup (current)
config_manager = SimpleConfigManager()
experiment_tracker = SimpleExperimentTracker("./experiments")
```

### Step 2: Swap to MLflow (just change 2 lines!)
```python
# MLflow setup (future)
config_manager = SimpleConfigManager()  # Keep simple config
experiment_tracker = MLflowExperimentTracker("http://localhost:5556")  # Add MLflow
```

### Step 3: Add Hydra (just change 2 lines!)
```python
# Hydra + MLflow setup (future)
config_manager = HydraConfigManager()  # Add Hydra
experiment_tracker = MLflowExperimentTracker("http://localhost:5556")  # Keep MLflow
```

### Step 4: Full Production Setup
```python
# Production setup with both
config_manager = HydraConfigManager()
experiment_tracker = MLflowExperimentTracker("http://production-mlflow:5556")

# Optional: Composite tracker for multiple backends
experiment_tracker = CompositeTracker([
    MLflowExperimentTracker("http://mlflow:5556"),
    WandbExperimentTracker(project="ttm-experiments")
])
```

---

## LoRA Integration (Ready for Extension)

```python
# core/adapters.py
from typing import Optional, Dict, Any
from peft import LoraConfig, get_peft_model, TaskType

class LoRAManager:
    """Manages LoRA adapter configurations and applications"""
    
    def __init__(self):
        self.supported_models = {
            'ttm': {
                'target_modules': ['q_proj', 'v_proj', 'k_proj', 'out_proj'],
                'task_type': TaskType.CAUSAL_LM
            },
            'chronos': {
                'target_modules': ['q_proj', 'v_proj'],
                'task_type': TaskType.CAUSAL_LM  
            }
        }
    
    def add_lora_to_model(
        self,
        model,
        model_type: str = 'ttm',
        rank: int = 16,
        alpha: int = 32,
        dropout: float = 0.1,
        bias: str = "none",
        **kwargs
    ):
        """Add LoRA adapters to model"""
        
        if model_type not in self.supported_models:
            raise ValueError(f"Model type {model_type} not supported")
        
        model_config = self.supported_models[model_type]
        
        # Create LoRA configuration
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            target_modules=model_config['target_modules'],
            lora_dropout=dropout,
            bias=bias,
            task_type=model_config['task_type'],
            **kwargs
        )
        
        # Apply LoRA to model
        lora_model = get_peft_model(model, lora_config)
        
        return lora_model
    
    def get_lora_params_count(self, model) -> Dict[str, int]:
        """Get parameter counts for LoRA model"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'percentage_trainable': (trainable_params / total_params) * 100
        }

# Simple function for backward compatibility
def add_lora_adapters(model, rank: int = 16, model_type: str = 'ttm'):
    """Simple LoRA addition function"""
    manager = LoRAManager()
    return manager.add_lora_to_model(model, model_type=model_type, rank=rank)
```

---

## Benefits of This Architecture

### 1. **Zero Breaking Changes**
- Swap out config/tracking systems without changing training logic
- Start simple, upgrade incrementally

### 2. **Easy A/B Testing**
```python
# Test different experiment trackers easily
trackers = [
    SimpleExperimentTracker("./simple_experiments"),
    MLflowExperimentTracker("http://localhost:5556"),
    WandbExperimentTracker(project="ttm-comparison")
]

for tracker in trackers:
    trainer = TTMTrainer(config_manager, tracker, "config.yaml")
    trainer.train()
```

### 3. **Research-Friendly Extensions**
```python
# Add new experiment tracking easily
class CustomResearchTracker(ExperimentTracker):
    def log_gradients(self, model):
        # Custom gradient logging
        pass
    
    def log_attention_weights(self, model):
        # Custom attention analysis
        pass
```

### 4. **Production-Ready Scaling**
- Use simple file-based tracking for development
- Switch to MLflow for team collaboration
- Add Hydra for complex hyperparameter sweeps
- No code changes to core training logic

---

This architecture lets you start simple and grow incrementally, making MLflow and Hydra integration completely seamless when you're ready!