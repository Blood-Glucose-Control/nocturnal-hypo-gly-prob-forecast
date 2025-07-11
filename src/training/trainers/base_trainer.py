"""Base trainer class that all model trainers should inherit from."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from pathlib import Path

from ..checkpointing.checkpoint_manager import CheckpointManager


class BaseTrainer(ABC):
    """Abstract base class for all model trainers."""

    def __init__(self, model_name: str, use_gpu: bool = True):
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.model = None

    @abstractmethod
    def train(
        self,
        data: Any,
        patient_ids: Optional[List[str]] = None,
        epochs: int = 100,
        checkpoint_manager: Optional[CheckpointManager] = None,
        model_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Train the model on the provided data.

        Args:
            data: Training data (format depends on the model type)
            patient_ids: List of patient IDs to train on (if None, use all)
            epochs: Number of training epochs
            checkpoint_manager: Optional checkpoint manager for saving checkpoints
            model_params: Optional parameters specific to the model type

        Returns:
            Dictionary of training results/metrics
        """
        pass

    @abstractmethod
    def save_model(self, model: Any, path: Path) -> None:
        """Save model to disk.

        Args:
            model: Trained model to save
            path: Directory path to save the model
        """
        pass

    @abstractmethod
    def load_model(self, path: Path) -> Any:
        """Load model from disk.

        Args:
            path: Directory path where model is saved

        Returns:
            Loaded model
        """
        pass

    @abstractmethod
    def predict(self, model: Any, data: Any) -> Any:
        """Make predictions with the model.

        Args:
            model: Trained model
            data: Input data for prediction

        Returns:
            Model predictions
        """
        pass
