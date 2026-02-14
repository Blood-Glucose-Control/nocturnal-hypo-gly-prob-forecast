# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)
# For commercial licensing, contact: christopher/cjrisi AT gluroo/uwaterloo DOT com/ca

"""Patient ID formatting utilities for standardized patient identification across datasets.

All patient IDs follow the format: {3-letter-prefix}_{numeric_id}
Examples: ale_140, bro_164, tam_50, lyn_270, bri_12345

This ensures consistent string-type patient identifiers across all datasets,
avoiding issues with int64 or mixed-type patient numbers.
"""

from __future__ import annotations

import re
from typing import Union

# Mapping from dataset names to 3-letter prefixes
DATASET_PREFIXES: dict[str, str] = {
    "aleppo_2017": "ale",
    "brown_2019": "bro",
    "tamborlane_2008": "tam",
    "lynch_2022": "lyn",
    "kaggle_brisT1D": "bri",
    "gluroo": "glu",  # Reserved for future use
}


def format_patient_id(dataset_name: str, raw_id: Union[int, str, float]) -> str:
    """Create standardized patient ID with prefix_### format.

    Args:
        dataset_name: Name of the dataset (must be in DATASET_PREFIXES).
        raw_id: Raw patient identifier (int, float, or string).
            Floats are converted to int. Strings are cleaned of non-numeric chars
            except for existing prefixes which are stripped.

    Returns:
        Formatted patient ID string like "ale_140" or "bro_92".

    Raises:
        ValueError: If dataset_name is not in DATASET_PREFIXES or raw_id cannot be parsed.

    Examples:
        >>> format_patient_id("aleppo_2017", 140)
        'ale_140'
        >>> format_patient_id("aleppo_2017", "p140")  # Strip existing 'p' prefix
        'ale_140'
        >>> format_patient_id("brown_2019", 92.0)
        'bro_92'
        >>> format_patient_id("lynch_2022", "lynch_270")  # Strip existing prefix
        'lyn_270'
    """
    if dataset_name not in DATASET_PREFIXES:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Valid datasets: {list(DATASET_PREFIXES.keys())}"
        )

    prefix = DATASET_PREFIXES[dataset_name]

    # Handle float inputs by converting to int first
    if isinstance(raw_id, float):
        raw_id = int(raw_id)

    # Convert to string for processing
    raw_str = str(raw_id)

    # Handle string representations of floats like "2.0" -> "2"
    # Try to convert to float then int if it looks like a number
    if "." in raw_str:
        try:
            raw_str = str(int(float(raw_str)))
        except ValueError:
            pass  # Not a valid float, continue with string processing

    # Strip existing prefixes (p###, lynch_###, gluroo_#, etc.)
    # Match common prefix patterns and extract the numeric part
    numeric_match = re.search(r"(\d+)$", raw_str)
    if numeric_match:
        numeric_id = numeric_match.group(1)
    else:
        # If no numeric suffix found, try to use the whole thing as-is
        # This handles edge cases like purely numeric strings
        cleaned = re.sub(r"[^\d]", "", raw_str)
        if cleaned:
            numeric_id = cleaned
        else:
            raise ValueError(
                f"Cannot extract numeric ID from raw_id '{raw_id}' for dataset '{dataset_name}'"
            )

    # Remove leading zeros but keep at least one digit
    numeric_id = str(int(numeric_id))

    return f"{prefix}_{numeric_id}"


def get_prefix_for_dataset(dataset_name: str) -> str:
    """Get the 3-letter prefix for a dataset.

    Args:
        dataset_name: Name of the dataset.

    Returns:
        The 3-letter prefix string.

    Raises:
        ValueError: If dataset_name is not in DATASET_PREFIXES.
    """
    if dataset_name not in DATASET_PREFIXES:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Valid datasets: {list(DATASET_PREFIXES.keys())}"
        )
    return DATASET_PREFIXES[dataset_name]
