#!/usr/bin/env python3
"""
Tests for the logging_helper module.

This module tests the info_print, error_print, debug_print functions
and the internal _get_caller_name helper function.
"""

import io
import sys
import unittest
from unittest.mock import patch, MagicMock
from contextlib import redirect_stderr

# Import the functions we want to test
from src.utils.logging_helper import (
    info_print,
    error_print,
    debug_print,
    _get_caller_name,
)


class TestLoggingHelper(unittest.TestCase):
    """Test cases for logging helper functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.stderr_buffer = io.StringIO()

    def tearDown(self):
        """Clean up after tests."""
        self.stderr_buffer.close()

    def test_info_print_basic_message(self):
        """Test info_print prints basic message to stderr."""
        with redirect_stderr(self.stderr_buffer):
            info_print("Test message")

        output = self.stderr_buffer.getvalue()
        self.assertIn("INFO:", output)
        self.assertIn("Test message", output)

    def test_error_print_basic_message(self):
        """Test error_print prints basic message to stderr."""
        with redirect_stderr(self.stderr_buffer):
            error_print("Test error")

        output = self.stderr_buffer.getvalue()
        self.assertIn("ERROR:", output)
        self.assertIn("Test error", output)

    def test_debug_print_basic_message(self):
        """Test debug_print prints basic message to stderr."""
        with redirect_stderr(self.stderr_buffer):
            debug_print("Test debug")

        output = self.stderr_buffer.getvalue()
        self.assertIn("DEBUG:", output)
        self.assertIn("Test debug", output)

    def test_info_print_with_caller_name(self):
        """Test that info_print includes caller function name."""

        def test_function():
            with redirect_stderr(self.stderr_buffer):
                info_print("Message from function")

        test_function()
        output = self.stderr_buffer.getvalue()
        self.assertIn("INFO:", output)
        self.assertIn("[test_function]", output)
        self.assertIn("Message from function", output)

    def test_error_print_with_caller_name(self):
        """Test that error_print includes caller function name."""

        def test_function():
            with redirect_stderr(self.stderr_buffer):
                error_print("Error from function")

        test_function()
        output = self.stderr_buffer.getvalue()
        self.assertIn("ERROR:", output)
        self.assertIn("[test_function]", output)
        self.assertIn("Error from function", output)

    def test_debug_print_with_caller_name(self):
        """Test that debug_print includes caller function name."""

        def test_function():
            with redirect_stderr(self.stderr_buffer):
                debug_print("Debug from function")

        test_function()
        output = self.stderr_buffer.getvalue()
        self.assertIn("DEBUG:", output)
        self.assertIn("[test_function]", output)
        self.assertIn("Debug from function", output)

    def test_multiple_arguments(self):
        """Test that functions handle multiple arguments correctly."""
        with redirect_stderr(self.stderr_buffer):
            info_print("Message", "with", "multiple", "args")

        output = self.stderr_buffer.getvalue()
        self.assertIn("INFO:", output)
        self.assertIn("Message with multiple args", output)

    def test_kwargs_handling(self):
        """Test that functions pass through kwargs to print."""
        with redirect_stderr(self.stderr_buffer):
            info_print("Test", "message", sep="-", end="!\n")

        output = self.stderr_buffer.getvalue()
        self.assertIn("INFO:", output)
        self.assertIn("Test-message!", output)

    def test_nested_function_calls(self):
        """Test caller name detection with nested function calls."""

        def outer_function():
            def inner_function():
                with redirect_stderr(self.stderr_buffer):
                    info_print("Nested call")

            inner_function()

        outer_function()
        output = self.stderr_buffer.getvalue()
        # Should show inner_function as the caller
        self.assertIn("[inner_function]", output)

    def test_class_method_caller(self):
        """Test caller name detection from class methods."""
        # Capture stderr_buffer in local scope to avoid self reference issues
        stderr_buffer = self.stderr_buffer

        class TestClass:
            def test_method(self):
                with redirect_stderr(stderr_buffer):
                    error_print("From class method")

        test_obj = TestClass()
        test_obj.test_method()
        output = self.stderr_buffer.getvalue()
        # The caller name should contain test_method, but might have additional info
        self.assertIn("test_method", output)
        self.assertIn("ERROR:", output)
        self.assertIn("From class method", output)

    def test_get_caller_name_direct(self):
        """Test _get_caller_name function directly."""

        def test_caller():
            return _get_caller_name()

        result = test_caller()
        # When called directly, _get_caller_name goes back 2 frames, so it picks up the test method name
        # This is expected behavior since the function is designed for use within logging functions
        self.assertEqual(result, "[test_get_caller_name_direct]")

    def test_get_caller_name_filtered_names(self):
        """Test that _get_caller_name filters out unwanted function names."""
        # Test by mocking the frame inspection
        with patch("inspect.currentframe") as mock_frame:
            # Create mock frame structure
            mock_actual_frame = MagicMock()
            mock_actual_frame.f_code.co_name = "<module>"

            mock_helper_frame = MagicMock()
            mock_helper_frame.f_back = mock_actual_frame

            mock_current_frame = MagicMock()
            mock_current_frame.f_back = mock_helper_frame

            mock_frame.return_value = mock_current_frame

            result = _get_caller_name()
            self.assertEqual(result, "")  # Should be filtered out

    def test_get_caller_name_with_wrapper(self):
        """Test that wrapper functions are filtered out."""
        with patch("inspect.currentframe") as mock_frame:
            mock_actual_frame = MagicMock()
            mock_actual_frame.f_code.co_name = "wrapper"

            mock_helper_frame = MagicMock()
            mock_helper_frame.f_back = mock_actual_frame

            mock_current_frame = MagicMock()
            mock_current_frame.f_back = mock_helper_frame

            mock_frame.return_value = mock_current_frame

            result = _get_caller_name()
            self.assertEqual(result, "")  # Should be filtered out

    def test_get_caller_name_with_main(self):
        """Test that main function is filtered out."""
        with patch("inspect.currentframe") as mock_frame:
            mock_actual_frame = MagicMock()
            mock_actual_frame.f_code.co_name = "main"

            mock_helper_frame = MagicMock()
            mock_helper_frame.f_back = mock_actual_frame

            mock_current_frame = MagicMock()
            mock_current_frame.f_back = mock_helper_frame

            mock_frame.return_value = mock_current_frame

            result = _get_caller_name()
            self.assertEqual(result, "")  # Should be filtered out

    def test_get_caller_name_none_frame(self):
        """Test handling when frame is None."""
        with patch("inspect.currentframe") as mock_frame:
            mock_frame.return_value = None
            result = _get_caller_name()
            self.assertEqual(result, "")

    def test_stderr_output_destination(self):
        """Test that all functions write to stderr, not stdout."""
        stdout_buffer = io.StringIO()

        with redirect_stderr(self.stderr_buffer), patch("sys.stdout", stdout_buffer):
            info_print("Info test")
            error_print("Error test")
            debug_print("Debug test")

        # stderr should have content
        stderr_content = self.stderr_buffer.getvalue()
        self.assertIn("Info test", stderr_content)
        self.assertIn("Error test", stderr_content)
        self.assertIn("Debug test", stderr_content)

        # stdout should be empty
        stdout_content = stdout_buffer.getvalue()
        self.assertEqual(stdout_content, "")

    def test_flush_behavior(self):
        """Test that flush=True is working (by checking it's called)."""
        with patch("builtins.print") as mock_print:
            info_print("Test flush")

            # Verify print was called with flush=True
            mock_print.assert_called_once()
            call_args = mock_print.call_args
            self.assertEqual(call_args.kwargs.get("flush"), True)
            self.assertEqual(call_args.kwargs.get("file"), sys.stderr)

    def test_empty_message(self):
        """Test handling of empty messages."""
        with redirect_stderr(self.stderr_buffer):
            info_print()

        output = self.stderr_buffer.getvalue()
        self.assertIn("INFO:", output)

    def test_unicode_message(self):
        """Test handling of unicode characters."""
        with redirect_stderr(self.stderr_buffer):
            info_print("Test with unicode: 🧪 ✅ 🚀")

        output = self.stderr_buffer.getvalue()
        self.assertIn("INFO:", output)
        self.assertIn("🧪 ✅ 🚀", output)

    def test_long_message(self):
        """Test handling of very long messages."""
        long_message = "x" * 1000
        with redirect_stderr(self.stderr_buffer):
            info_print(long_message)

        output = self.stderr_buffer.getvalue()
        self.assertIn("INFO:", output)
        self.assertIn(long_message, output)

    def test_numeric_arguments(self):
        """Test handling of numeric arguments."""
        with redirect_stderr(self.stderr_buffer):
            info_print("Number:", 42, "Float:", 3.14159, "Bool:", True)

        output = self.stderr_buffer.getvalue()
        self.assertIn("INFO:", output)
        self.assertIn("Number: 42", output)
        self.assertIn("Float: 3.14159", output)
        self.assertIn("Bool: True", output)

    def test_real_world_usage_scenario(self):
        """Test a realistic usage scenario."""

        def train_model():
            info_print("Starting model training...")

            for epoch in range(3):
                debug_print(f"Processing epoch {epoch}")

                if epoch == 1:
                    error_print("Warning: some predictions were NaN")

            info_print("Training completed successfully")

        with redirect_stderr(self.stderr_buffer):
            train_model()

        output = self.stderr_buffer.getvalue()
        lines = output.strip().split("\n")

        # Check that we have the expected number of log lines
        self.assertEqual(len(lines), 6)  # 1 start + 3 debug + 1 error + 1 complete

        # Check content and format
        self.assertIn("INFO: [train_model] Starting model training", lines[0])
        self.assertIn("DEBUG: [train_model] Processing epoch 0", lines[1])
        self.assertIn("DEBUG: [train_model] Processing epoch 1", lines[2])
        self.assertIn(
            "ERROR: [train_model] Warning: some predictions were NaN", lines[3]
        )
        self.assertIn("DEBUG: [train_model] Processing epoch 2", lines[4])
        self.assertIn("INFO: [train_model] Training completed successfully", lines[5])


class TestIntegrationWithExistingCode(unittest.TestCase):
    """Integration tests to ensure compatibility with existing codebase."""

    def setUp(self):
        self.stderr_buffer = io.StringIO()

    def tearDown(self):
        self.stderr_buffer.close()

    def test_backward_compatibility(self):
        """Test that the refactored functions work the same as before."""

        def existing_function_pattern():
            with redirect_stderr(self.stderr_buffer):
                info_print("Model training started")
                debug_print("Loading data from cache")
                error_print("CUDA out of memory")

        existing_function_pattern()
        output = self.stderr_buffer.getvalue()

        # Verify expected patterns exist
        self.assertIn(
            "INFO: [existing_function_pattern] Model training started", output
        )
        self.assertIn(
            "DEBUG: [existing_function_pattern] Loading data from cache", output
        )
        self.assertIn("ERROR: [existing_function_pattern] CUDA out of memory", output)


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
