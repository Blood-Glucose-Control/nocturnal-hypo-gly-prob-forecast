"""Core training pipeline that orchestrates model training workflows."""

import yaml
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from src.data.datasets.data_loader import get_loader
from .trainers.base_trainer import BaseTrainer
from .trainers.statistical_trainer import StatisticalTrainer

# TODO: Import other trainers when they are implemented
# from .trainers.ml_trainer import MLTrainer
# from .trainers.dl_trainer import DeepLearningTrainer
# from .trainers.foundation_trainer import FoundationTrainer
from .checkpointing.checkpoint_manager import CheckpointManager
from .evaluation.evaluator import ModelEvaluator


@dataclass
class TrainingConfig:
    """Configuration for training pipeline."""

    model_type: str  # 'statistical', 'ml', 'dl', 'foundation'
    model_name: str
    dataset_name: str
    epochs: int
    checkpoint_frequency: int = 10
    output_dir: str = "models"
    results_dir: str = "results"
    use_gpu: bool = True
    slurm_job: bool = False
    model_params: Optional[Dict[str, Any]] = None
    data_config: Optional[Dict[str, Any]] = None


class TrainingPipeline:
    """Main training pipeline that trains models on all available patients by default."""

    def __init__(self, config: TrainingConfig, config_file_path: Optional[str] = None):
        self.config = config
        self.config_file_path = config_file_path
        self.trainer = self._get_trainer()
        self.checkpoint_manager = CheckpointManager(
            output_dir=config.output_dir,
            checkpoint_frequency=config.checkpoint_frequency,
        )
        self.evaluator = ModelEvaluator(results_dir=config.results_dir)

    def _get_trainer(self) -> BaseTrainer:
        """Get the appropriate trainer based on model type."""
        trainer_map = {
            "statistical": StatisticalTrainer,
            # TODO: Add other trainers when implemented
            # 'ml': MLTrainer,
            # 'dl': DeepLearningTrainer,
            # 'foundation': FoundationTrainer
        }

        if self.config.model_type not in trainer_map:
            raise ValueError(
                f"Unknown model type: {self.config.model_type}. "
                f"Available types: {list(trainer_map.keys())}"
            )

        return trainer_map[self.config.model_type](
            model_name=self.config.model_name, use_gpu=self.config.use_gpu
        )

    def train(self, patient_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Train model on all patients (default) or specified subset."""
        if patient_ids is None:
            patient_ids = self._get_all_patient_ids()

        print(f"Training on {len(patient_ids)} patients...")

        # Load all patient data at once for models that can handle it
        data_config = self.config.data_config or {}
        loader = get_loader(
            data_source_name=self.config.dataset_name,  # type: ignore
            dataset_type=data_config.get("dataset_type", "train"),
            use_cached=data_config.get("use_cached", True),  # Default to cached
            keep_columns=data_config.get("keep_columns"),
            num_validation_days=data_config.get("num_validation_days", 20),
        )

        # Train model on combined patient data
        model_results = self.trainer.train(
            data=loader,
            patient_ids=patient_ids,
            epochs=self.config.epochs,
            checkpoint_manager=self.checkpoint_manager,
            model_params=self.config.model_params,
        )

        return model_results

    def _get_all_patient_ids(self) -> List[str]:
        """Get all available patient IDs from the dataset."""
        data_config = self.config.data_config or {}
        loader = get_loader(
            data_source_name=self.config.dataset_name,  # type: ignore
            dataset_type=data_config.get("dataset_type", "train"),
            use_cached=data_config.get("use_cached", True),  # Default to cached
            keep_columns=data_config.get("keep_columns"),
            num_validation_days=data_config.get("num_validation_days", 20),
        )
        return (
            list(loader.processed_data.keys())
            if hasattr(loader, "processed_data")
            else []
        )

    def save_model(self, model, model_info: Dict[str, Any]) -> str:
        """Save trained model with descriptive naming and config."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir_name = (
            f"{timestamp}_{self.config.model_name}_{self.config.dataset_name}_v1"
        )
        model_path = Path(self.config.output_dir) / model_dir_name
        model_path.mkdir(parents=True, exist_ok=True)

        # Copy original config file to model directory
        if self.config_file_path:
            config_filename = f"config_{timestamp}.yaml"
            shutil.copy2(self.config_file_path, model_path / config_filename)

        # Save model and metadata
        self.trainer.save_model(model, model_path)
        self._save_model_metadata(model_info, model_path)

        return str(model_path)

    def _save_model_metadata(self, model_info: Dict[str, Any], model_path: Path):
        """Save model training metadata."""
        metadata = {
            "model_type": self.config.model_type,
            "model_name": self.config.model_name,
            "dataset_name": self.config.dataset_name,
            "training_date": datetime.now().isoformat(),
            "epochs": self.config.epochs,
            "config_file": self.config_file_path,
            **model_info,
        }

        import json

        with open(model_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    @classmethod
    def from_config_file(cls, config_file_path: str) -> "TrainingPipeline":
        """Create pipeline from YAML config file."""
        with open(config_file_path, "r") as f:
            config_dict = yaml.safe_load(f)

        config = TrainingConfig(**config_dict)
        return cls(config, config_file_path)
