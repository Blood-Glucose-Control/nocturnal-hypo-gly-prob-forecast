import unittest
import pandas as pd
import numpy as np

from src.tuning.benchmark import impute_missing_values


class TestImputeMissingValues(unittest.TestCase):
    def setUp(self):
        # Create a test DataFrame with various types of columns and missing values
        self.df = pd.DataFrame(
            {
                "datetime": pd.date_range(start="2023-01-01", periods=6, freq="15min"),
                "bg-0:00": [100, np.nan, 110, 105, np.nan, 120],
                "hr-0:00": [70, 72, np.nan, 75, np.nan, 80],
                "steps-0:00": [0, 100, 200, np.nan, np.nan, 300],
                "cals-0:00": [1200, 1250, np.nan, 1300, np.nan, 1400],
            }
        )
        self.df.set_index("datetime", inplace=True)

    def test_impute_all_columns(self):
        """Test imputation for all column types with default methods"""
        columns = ["bg-0:00", "hr-0:00", "steps-0:00", "cals-0:00"]
        df_imputed = impute_missing_values(self.df, columns)

        # Check that the original DataFrame wasn't modified
        self.assertTrue(self.df["bg-0:00"].isna().any())

        # Check that all NaN values were imputed
        self.assertFalse(df_imputed["bg-0:00"].isna().any())
        self.assertFalse(df_imputed["hr-0:00"].isna().any())
        self.assertFalse(df_imputed["steps-0:00"].isna().any())
        self.assertFalse(df_imputed["cals-0:00"].isna().any())

        # Check that steps-0:00 was imputed with constant 0
        step_nan_indices = self.df[self.df["steps-0:00"].isna()].index
        for idx in step_nan_indices:
            self.assertEqual(df_imputed.loc[idx, "steps-0:00"], 0)

    def test_impute_bg_only(self):
        """Test imputation for just blood glucose data"""
        columns = ["bg-0:00"]
        df_imputed = impute_missing_values(self.df, columns)

        # Check that bg values were imputed
        self.assertFalse(df_imputed["bg-0:00"].isna().any())

        # Check that other columns weren't touched
        self.assertTrue(df_imputed["hr-0:00"].isna().any())
        self.assertTrue(df_imputed["steps-0:00"].isna().any())
        self.assertTrue(df_imputed["cals-0:00"].isna().any())

        # Verify the blood glucose values were interpolated
        # For linear interpolation, we can check if values are between the known points
        bg_nan_indices = self.df[self.df["bg-0:00"].isna()].index
        for idx in bg_nan_indices:
            # Get surrounding values to verify interpolation is reasonable
            nearby_values = self.df["bg-0:00"].dropna()
            self.assertTrue(
                nearby_values.min()
                <= df_imputed.loc[idx, "bg-0:00"]
                <= nearby_values.max()
            )

    def test_impute_with_custom_methods(self):
        """Test imputation with non-default methods"""
        columns = ["bg-0:00", "hr-0:00"]
        df_imputed = impute_missing_values(
            self.df, columns, bg_method="nearest", hr_method="nearest"
        )

        # Check imputation worked
        self.assertFalse(df_imputed["bg-0:00"].isna().any())
        self.assertFalse(df_imputed["hr-0:00"].isna().any())

        # With nearest method, check if values match one of the known values
        known_bg_values = set(self.df["bg-0:00"].dropna().values)
        known_hr_values = set(self.df["hr-0:00"].dropna().values)

        bg_nan_indices = self.df[self.df["bg-0:00"].isna()].index
        hr_nan_indices = self.df[self.df["hr-0:00"].isna()].index

        for idx in bg_nan_indices:
            self.assertIn(df_imputed.loc[idx, "bg-0:00"], known_bg_values)

        for idx in hr_nan_indices:
            self.assertIn(df_imputed.loc[idx, "hr-0:00"], known_hr_values)

    def test_nonexistent_column(self):
        """Test that the function handles columns that don't exist in the DataFrame"""
        columns = ["bg-0:00", "nonexistent_column"]
        # This should not raise an error
        df_imputed = impute_missing_values(self.df, columns)

        # Check that bg values were still imputed properly
        self.assertFalse(df_imputed["bg-0:00"].isna().any())

    def test_cals_min_value_imputation(self):
        """Test that cals-0:00 use the min value for constant imputation"""
        columns = ["cals-0:00"]
        df_imputed = impute_missing_values(self.df, columns)

        # Get the minimum value in the original cals-0:00 column
        min_val = self.df["cals-0:00"].min()

        # Check that NaN values were replaced with the minimum value
        cal_nan_indices = self.df[self.df["cals-0:00"].isna()].index
        for idx in cal_nan_indices:
            self.assertEqual(df_imputed.loc[idx, "cals-0:00"], min_val)
