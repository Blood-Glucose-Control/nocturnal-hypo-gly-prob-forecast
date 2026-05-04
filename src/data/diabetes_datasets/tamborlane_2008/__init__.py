"""
Tamborlane 2008 CGM Dataset Module.

This module provides data loading and processing capabilities for the Tamborlane 2008
continuous glucose monitoring dataset from pediatric Type 1 diabetes patients.
"""

from .tamborlane_2008 import Tamborlane2008DataLoader
from .data_cleaner import (
    clean_tamborlane_2008_data,
    process_single_patient_tamborlane,
    extract_cgm_features,
    validate_tamborlane_data,
    prepare_for_modeling,
)

__all__ = [
    "Tamborlane2008DataLoader",
    "clean_tamborlane_2008_data",
    "process_single_patient_tamborlane",
    "extract_cgm_features",
    "validate_tamborlane_data",
    "prepare_for_modeling",
]
