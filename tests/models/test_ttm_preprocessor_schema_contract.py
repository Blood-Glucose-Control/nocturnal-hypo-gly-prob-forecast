"""Unit tests for TTM preprocessor schema validation."""

import pytest

pytest.importorskip("tsfm_public")


class _LegacyPreprocessor:
    """Minimal stand-in for old pickled preprocessors."""

    def __init__(self):
        self.target_columns = ["bg_mM"]


def test_legacy_preprocessor_is_rejected_with_actionable_error():
    from src.models.ttm.model import _validate_preprocessor_schema

    legacy = _LegacyPreprocessor()
    assert not hasattr(legacy, "other_columns_to_scale")

    with pytest.raises(ValueError, match="unsupported by the current runtime"):
        _validate_preprocessor_schema(legacy)  # type: ignore[arg-type]
