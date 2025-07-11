"""Trainer modules for different model types."""

from .base_trainer import BaseTrainer
from .statistical_trainer import StatisticalTrainer
# TODO: Import other trainers when they are implemented
# from .ml_trainer import MLTrainer
# from .dl_trainer import DeepLearningTrainer
# from .foundation_trainer import FoundationTrainer

__all__ = [
    "BaseTrainer",
    "StatisticalTrainer",
    # TODO: Add other trainers when implemented
    # "MLTrainer",
    # "DeepLearningTrainer",
    # "FoundationTrainer"
]
