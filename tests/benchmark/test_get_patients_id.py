import unittest
import pandas as pd
from src.tuning.benchmark import get_patient_ids


class TestGetPatientIds(unittest.TestCase):
    def setUp(self):
        times_5min = pd.date_range(start="2023-01-01", periods=10, freq="5min")
        times_15min = pd.date_range(start="2023-01-01", periods=10, freq="15min")

        data = []

        # Patient 1: 5-minute intervals
        for time in times_5min:
            data.append({"p_num": "p01", "time": time})

        # Patient 2: 15-minute intervals
        for time in times_15min:
            data.append({"p_num": "p02", "time": time})

        # Patient 3: 5-minute intervals
        for time in times_5min:
            data.append({"p_num": "p03", "time": time})

        # Patient 4: 15-minute intervals
        for time in times_15min:
            data.append({"p_num": "p04", "time": time})

        # Patient 5: irregular intervals (should be excluded)
        irregular_times = [
            pd.Timestamp("2023-01-01 00:00:00"),
            pd.Timestamp("2023-01-01 00:07:00"),  # Not 5 or 15 min
            pd.Timestamp("2023-01-01 00:14:00"),
            pd.Timestamp("2023-01-01 00:21:00"),
        ]
        for time in irregular_times:
            data.append({"p_num": "p05", "time": time})

        self.df = pd.DataFrame(data)

    def test_get_5min_patients(self):
        """Test getting patients with 5-minute intervals"""
        patients = get_patient_ids(self.df, is_5min=True)

        # Should return P01 and P03
        self.assertEqual(len(patients), 2)
        self.assertIn("p01", patients)
        self.assertIn("p03", patients)
        self.assertNotIn("p02", patients)
        self.assertNotIn("p04", patients)
        self.assertNotIn("p05", patients)

    def test_get_15min_patients(self):
        """Test getting patients with 15-minute intervals"""
        patients = get_patient_ids(self.df, is_5min=False)

        # Should return P002 and P004
        self.assertEqual(len(patients), 2)
        self.assertIn("p02", patients)
        self.assertIn("p04", patients)
        self.assertNotIn("p01", patients)
        self.assertNotIn("p03", patients)
        self.assertNotIn("p05", patients)

    def test_limit_number_of_patients(self):
        """Test limiting the number of returned patients"""
        patients = get_patient_ids(self.df, is_5min=True, n_patients=1)
        self.assertEqual(len(patients), 1)
        self.assertTrue(patients[0] in ["p01", "p03"])

    def test_n_patients_greater_than_available(self):
        """Test when n_patients is greater than available patients"""
        patients = get_patient_ids(self.df, is_5min=True, n_patients=10)
        self.assertEqual(len(patients), 2)

    def test_n_patients_negative(self):
        """Test with n_patients=-1, which should return all matching patients"""
        patients = get_patient_ids(self.df, is_5min=True, n_patients=-1)
        self.assertEqual(len(patients), 2)

    def test_no_matching_patients(self):
        """Test when no patients match the criteria"""
        # Create a DataFrame with only 10-min interval patients
        times_10min = pd.date_range(start="2023-01-01", periods=10, freq="10min")
        data = []
        for time in times_10min:
            data.append({"p_num": "p06", "time": time})
        df = pd.DataFrame(data)

        patients_5min = get_patient_ids(df, is_5min=True)
        patients_15min = get_patient_ids(df, is_5min=False)

        self.assertEqual(len(patients_5min), 0)
        self.assertEqual(len(patients_15min), 0)
