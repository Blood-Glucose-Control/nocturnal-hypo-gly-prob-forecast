# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)
# For commercial licensing, contact: [Add your contact information]

"""
Base transformer interfaces and abstract classes.

This module provides the base interfaces for all data transformers in the system,
ensuring consistent APIs across different transformation methods.
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import Union


class BaseTransformer(ABC):
    """
    Abstract base class for all data transformers.

    Transformers convert input data (Series or DataFrame) into transformed output
    while maintaining consistent interfaces for fitting and transforming data.
    """

    @abstractmethod
    def fit(self, X: Union[pd.Series, pd.DataFrame], y=None) -> "BaseTransformer":
        """
        Fit the transformer to the data.

        Args:
            X: Input data to fit the transformer on
            y: Optional target data (unused in most transformers)

        Returns:
            Self, for method chaining
        """
        pass

    @abstractmethod
    def transform(
        self, X: Union[pd.Series, pd.DataFrame]
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Transform the input data using the fitted transformer.

        Args:
            X: Input data to transform

        Returns:
            Transformed data, same type as input
        """
        pass

    def fit_transform(
        self, X: Union[pd.Series, pd.DataFrame], y=None
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Fit the transformer to the data and transform in one step.

        Args:
            X: Input data to fit and transform
            y: Optional target data (unused in most transformers)

        Returns:
            Transformed data
        """
        return self.fit(X, y).transform(X)


class PatientGroupTransformer(BaseTransformer):
    """
    Applies a transformer to data grouped by patient ID.

    This transformer wraps another transformer and applies it separately
    to each patient's data subset, then recombines the results.
    """

    def __init__(self, transformer: BaseTransformer, patient_col: str = "p_num"):
        """
        Args:
            transformer: A transformer with fit_transform method
            patient_col: Column name containing patient identifiers
        """
        self.transformer = transformer
        self.patient_col = patient_col
        self.fitted_transformers = {}  # Store fitted transformers by patient ID

    def fit(self, X: pd.DataFrame, y=None) -> "PatientGroupTransformer":
        """
        Fit the wrapped transformer to each patient group.

        Args:
            X: DataFrame containing patient data
            y: Ignored

        Returns:
            Self for method chaining
        """
        # Clear any previously fitted transformers
        self.fitted_transformers = {}

        # Fit a separate transformer for each patient
        for patient_id, patient_data in X.groupby(self.patient_col):
            # Create a fresh copy of the transformer for each patient
            patient_transformer = type(self.transformer)(**self.transformer.__dict__)
            self.fitted_transformers[patient_id] = patient_transformer.fit(patient_data)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the wrapped transformer to each patient group.

        Args:
            X: DataFrame containing patient data

        Returns:
            DataFrame with transformed values
        """
        transformed_data = []

        for patient_id, patient_data in X.groupby(self.patient_col):
            # Get the fitted transformer for this patient
            if patient_id in self.fitted_transformers:
                transformer = self.fitted_transformers[patient_id]
            else:
                # If we don't have a fitted transformer for this patient, use the default
                transformer = self.transformer

            transformed = transformer.transform(patient_data)
            transformed_data.append(transformed)

        if not transformed_data:
            return (
                X.copy()
            )  # Return a copy of the original if no transformations were done

        result_df = pd.concat(transformed_data, axis=0)
        return result_df.sort_index()  # Sort to preserve original order
