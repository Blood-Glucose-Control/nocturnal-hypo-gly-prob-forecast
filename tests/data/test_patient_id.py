# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)
# For commercial licensing, contact: christopher/cjrisi AT gluroo/uwaterloo DOT com/ca

"""Tests for patient ID formatting utilities."""

import pytest

from src.data.utils.patient_id import (
    DATASET_PREFIXES,
    format_patient_id,
    get_prefix_for_dataset,
)


class TestDatasetPrefixes:
    """Test DATASET_PREFIXES constant."""

    def test_all_datasets_have_prefixes(self):
        """Verify all expected datasets have 3-letter prefixes defined."""
        expected_datasets = [
            "aleppo_2017",
            "brown_2019",
            "tamborlane_2008",
            "lynch_2022",
            "kaggle_brisT1D",
            "gluroo",
        ]
        for dataset in expected_datasets:
            assert dataset in DATASET_PREFIXES
            assert len(DATASET_PREFIXES[dataset]) == 3

    def test_prefix_format(self):
        """Verify all prefixes are lowercase 3-letter strings."""
        for dataset, prefix in DATASET_PREFIXES.items():
            assert isinstance(prefix, str)
            assert len(prefix) == 3
            assert prefix.islower()


class TestFormatPatientId:
    """Test format_patient_id function."""

    def test_basic_integer_input(self):
        """Test with integer patient IDs."""
        assert format_patient_id("aleppo_2017", 140) == "ale_140"
        assert format_patient_id("brown_2019", 92) == "bro_92"
        assert format_patient_id("tamborlane_2008", 50) == "tam_50"
        assert format_patient_id("lynch_2022", 270) == "lyn_270"
        assert format_patient_id("kaggle_brisT1D", 12345) == "bri_12345"

    def test_string_input(self):
        """Test with string patient IDs."""
        assert format_patient_id("aleppo_2017", "140") == "ale_140"
        assert format_patient_id("brown_2019", "92") == "bro_92"

    def test_float_input(self):
        """Test with float patient IDs (should convert to int)."""
        assert format_patient_id("aleppo_2017", 140.0) == "ale_140"
        assert format_patient_id("brown_2019", 92.5) == "bro_92"

    def test_string_float_input(self):
        """Test with string representations of floats like '2.0'."""
        assert format_patient_id("aleppo_2017", "2.0") == "ale_2"
        assert format_patient_id("aleppo_2017", "140.0") == "ale_140"
        assert format_patient_id("brown_2019", "92.0") == "bro_92"

    def test_strip_existing_p_prefix(self):
        """Test stripping existing 'p' prefix from Aleppo IDs."""
        assert format_patient_id("aleppo_2017", "p140") == "ale_140"
        assert format_patient_id("aleppo_2017", "p33") == "ale_33"

    def test_strip_existing_lynch_prefix(self):
        """Test stripping existing 'lynch_' prefix."""
        assert format_patient_id("lynch_2022", "lynch_270") == "lyn_270"
        assert format_patient_id("lynch_2022", "lynch_400") == "lyn_400"

    def test_strip_leading_zeros(self):
        """Test that leading zeros are stripped from numeric IDs."""
        assert format_patient_id("brown_2019", "007") == "bro_7"
        assert format_patient_id("tamborlane_2008", "001") == "tam_1"

    def test_preserve_zero(self):
        """Test that ID of 0 is preserved as '0'."""
        assert format_patient_id("gluroo", 0) == "glu_0"
        assert format_patient_id("gluroo", "0") == "glu_0"

    def test_unknown_dataset_raises_error(self):
        """Test that unknown dataset names raise ValueError."""
        with pytest.raises(ValueError, match="Unknown dataset"):
            format_patient_id("unknown_dataset", 123)

    def test_non_numeric_id_raises_error(self):
        """Test that non-numeric IDs raise ValueError."""
        with pytest.raises(ValueError, match="Cannot extract numeric ID"):
            format_patient_id("aleppo_2017", "abc")


class TestGetPrefixForDataset:
    """Test get_prefix_for_dataset function."""

    def test_valid_datasets(self):
        """Test retrieval of prefixes for valid datasets."""
        assert get_prefix_for_dataset("aleppo_2017") == "ale"
        assert get_prefix_for_dataset("brown_2019") == "bro"
        assert get_prefix_for_dataset("tamborlane_2008") == "tam"
        assert get_prefix_for_dataset("lynch_2022") == "lyn"
        assert get_prefix_for_dataset("kaggle_brisT1D") == "bri"
        assert get_prefix_for_dataset("gluroo") == "glu"

    def test_unknown_dataset_raises_error(self):
        """Test that unknown dataset names raise ValueError."""
        with pytest.raises(ValueError, match="Unknown dataset"):
            get_prefix_for_dataset("unknown_dataset")
