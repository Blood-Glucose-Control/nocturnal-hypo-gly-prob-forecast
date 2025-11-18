import unittest
import numpy as np
from sktime.split import ExpandingSlidingWindowSplitter, ExpandingWindowSplitter
from src.tuning.benchmark import get_cv_splitter


class TestGetCVSplitter(unittest.TestCase):
    def setUp(self):
        self.initial_window = 48
        self.step_length = 24
        self.steps_per_hour = 4
        self.hours_to_forecast = 6

    def test_expanding_window_splitter(self):
        """Test that an ExpandingWindowSplitter is returned with correct parameters"""
        cv = get_cv_splitter(
            self.initial_window,
            self.step_length,
            self.steps_per_hour,
            self.hours_to_forecast,
            cv_type="expanding",
        )

        # Verify the return type
        self.assertIsInstance(cv, ExpandingWindowSplitter)

        # Verify the parameters
        self.assertEqual(cv.initial_window, self.initial_window)
        self.assertEqual(cv.step_length, self.step_length)

        # Check forecast horizon
        expected_fh = np.arange(1, self.steps_per_hour * self.hours_to_forecast + 1)
        np.testing.assert_array_equal(cv.fh, expected_fh)

    def test_expanding_sliding_window_splitter(self):
        """Test that an ExpandingSlidingWindowSplitter is returned with correct parameters"""
        cv = get_cv_splitter(
            self.initial_window,
            self.step_length,
            self.steps_per_hour,
            self.hours_to_forecast,
            cv_type="expanding_sliding",
        )

        # Verify the return type
        self.assertIsInstance(cv, ExpandingSlidingWindowSplitter)

        # Verify the parameters
        self.assertEqual(cv.initial_window, self.initial_window)
        self.assertEqual(cv.step_length, self.step_length)

        # Check forecast horizon
        expected_fh = np.arange(1, self.steps_per_hour * self.hours_to_forecast + 1)
        np.testing.assert_array_equal(cv.fh, expected_fh)

    def test_invalid_cv_type(self):
        """Test that ValueError is raised for invalid cv_type"""
        with self.assertRaises(ValueError) as context:
            get_cv_splitter(
                self.initial_window,
                self.step_length,
                self.steps_per_hour,
                self.hours_to_forecast,
                cv_type="invalid_type",
            )

        self.assertIn("Invalid cv_type", str(context.exception))
