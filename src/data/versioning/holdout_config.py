# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)
# For commercial licensing, contact: christopher/cjrisi AT gluroo/uwaterloo DOT com/ca

"""
Holdout configuration system for reproducible train/test splits.

This module defines holdout strategies for creating consistent train/holdout splits
across all experiments. Supports both temporal splits (holding out end of time series)
and patient-based splits (holding out specific patients).
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional, Union
import yaml
import logging

logger = logging.getLogger(__name__)


class HoldoutType(Enum):
    """Types of holdout strategies."""

    TEMPORAL = "temporal"
    PATIENT_BASED = "patient_based"
    HYBRID = "hybrid"  # Both temporal and patient-based


@dataclass
class TemporalHoldoutConfig:
    """Configuration for temporal holdout strategy.

    Holds out a percentage of data at the end of each patient's time series.
    """

    holdout_percentage: float = 0.2  # 20% of data at the end
    min_train_samples: int = 608  # Minimum samples required in training set
    min_holdout_samples: int = 608  # Minimum samples required in holdout set

    def __post_init__(self):
        """Validate configuration."""
        if not 0.0 < self.holdout_percentage < 1.0:
            raise ValueError(
                f"holdout_percentage must be between 0 and 1, got {self.holdout_percentage}"
            )
        if self.min_train_samples < 1:
            raise ValueError(
                f"min_train_samples must be positive, got {self.min_train_samples}"
            )
        if self.min_holdout_samples < 1:
            raise ValueError(
                f"min_holdout_samples must be positive, got {self.min_holdout_samples}"
            )


@dataclass
class PatientHoldoutConfig:
    """Configuration for patient-based holdout strategy.

    Holds out specific patients that are never used for training.
    """

    holdout_patients: List[str] = field(
        default_factory=list
    )  # List of patient IDs to hold out
    holdout_percentage: Optional[float] = (
        None  # Alternative: percentage of patients to hold out
    )
    min_train_patients: int = 10  # Minimum patients required for training
    min_holdout_patients: int = 10  # Minimum patients required for holdout
    random_seed: int = 42  # Seed for reproducible patient selection

    def __post_init__(self):
        """Validate configuration."""
        if self.holdout_percentage is not None:
            if not 0.0 < self.holdout_percentage < 1.0:
                raise ValueError(
                    f"holdout_percentage must be between 0 and 1, got {self.holdout_percentage}"
                )
        if self.min_train_patients < 1:
            raise ValueError(
                f"min_train_patients must be positive, got {self.min_train_patients}"
            )
        if self.min_holdout_patients < 1:
            raise ValueError(
                f"min_holdout_patients must be positive, got {self.min_holdout_patients}"
            )

    def has_predefined_patients(self) -> bool:
        """Check if specific patients are predefined."""
        return len(self.holdout_patients) > 0


@dataclass
class HoldoutConfig:
    """Complete holdout configuration for a dataset.

    Defines how to split data into training and holdout sets.
    Can use temporal splits, patient-based splits, or both.
    """

    dataset_name: str
    holdout_type: HoldoutType
    temporal_config: Optional[TemporalHoldoutConfig] = None
    patient_config: Optional[PatientHoldoutConfig] = None
    description: str = ""
    created_date: Optional[str] = None
    version: str = "1.0"

    def __post_init__(self):
        """Validate configuration consistency."""
        if self.holdout_type == HoldoutType.TEMPORAL:
            if self.temporal_config is None:
                raise ValueError("temporal_config required for TEMPORAL holdout type")
        elif self.holdout_type == HoldoutType.PATIENT_BASED:
            if self.patient_config is None:
                raise ValueError(
                    "patient_config required for PATIENT_BASED holdout type"
                )
        elif self.holdout_type == HoldoutType.HYBRID:
            if self.temporal_config is None or self.patient_config is None:
                raise ValueError(
                    "Both temporal_config and patient_config required for HYBRID holdout type"
                )

    def to_dict(self) -> dict:
        """Convert configuration to dictionary for serialization."""
        result = {
            "dataset_name": self.dataset_name,
            "holdout_type": self.holdout_type.value,
            "description": self.description,
            "created_date": self.created_date,
            "version": self.version,
        }

        if self.temporal_config:
            result["temporal_config"] = {
                "holdout_percentage": self.temporal_config.holdout_percentage,
                "min_train_samples": self.temporal_config.min_train_samples,
                "min_holdout_samples": self.temporal_config.min_holdout_samples,
            }

        if self.patient_config:
            result["patient_config"] = {
                "holdout_patients": self.patient_config.holdout_patients,
                "holdout_percentage": self.patient_config.holdout_percentage,
                "min_train_patients": self.patient_config.min_train_patients,
                "min_holdout_patients": self.patient_config.min_holdout_patients,
                "random_seed": self.patient_config.random_seed,
            }

        return result

    @classmethod
    def from_dict(cls, data: dict) -> "HoldoutConfig":
        """Create configuration from dictionary."""
        holdout_type = HoldoutType(data["holdout_type"])

        temporal_config = None
        if "temporal_config" in data:
            temporal_config = TemporalHoldoutConfig(**data["temporal_config"])

        patient_config = None
        if "patient_config" in data:
            patient_config = PatientHoldoutConfig(**data["patient_config"])

        return cls(
            dataset_name=data["dataset_name"],
            holdout_type=holdout_type,
            temporal_config=temporal_config,
            patient_config=patient_config,
            description=data.get("description", ""),
            created_date=data.get("created_date"),
            version=data.get("version", "1.0"),
        )

    def save(self, path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

        logger.info(f"Saved holdout config for {self.dataset_name} to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "HoldoutConfig":
        """Load configuration from YAML file."""
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Holdout config not found: {path}")

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        config = cls.from_dict(data)
        logger.info(f"Loaded holdout config for {config.dataset_name} from {path}")

        return config


# Predefined holdout configurations for common use cases
def get_default_temporal_config(
    holdout_percentage: float = 0.2,
) -> TemporalHoldoutConfig:
    """Get default temporal holdout configuration."""
    return TemporalHoldoutConfig(
        holdout_percentage=holdout_percentage,
        min_train_samples=100,
        min_holdout_samples=20,
    )


def get_default_patient_config(
    holdout_percentage: float = 0.2, random_seed: int = 42
) -> PatientHoldoutConfig:
    """Get default patient-based holdout configuration."""
    return PatientHoldoutConfig(
        holdout_patients=[],  # Will be filled in by generate_patient_holdout
        holdout_percentage=holdout_percentage,
        min_train_patients=3,
        min_holdout_patients=1,
        random_seed=random_seed,
    )


def get_default_hybrid_config(
    temporal_percentage: float = 0.2,
    patient_percentage: float = 0.2,
    random_seed: int = 42,
) -> tuple[TemporalHoldoutConfig, PatientHoldoutConfig]:
    """Get default hybrid holdout configuration."""
    return (
        get_default_temporal_config(temporal_percentage),
        get_default_patient_config(patient_percentage, random_seed),
    )
