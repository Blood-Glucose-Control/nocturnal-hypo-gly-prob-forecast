"""Unit tests for TTM preprocessor schema validation."""

import pytest

pytest.importorskip("tsfm_public")


class _UnsupportedSchemaPreprocessor:
    """Minimal stand-in for a preprocessor missing required schema fields."""

    def __init__(self):
        self.target_columns = ["bg_mM"]


def test_unsupported_schema_preprocessor_is_rejected_with_actionable_error():
    from src.models.ttm.model import _validate_preprocessor_schema

    preprocessor = _UnsupportedSchemaPreprocessor()
    assert not hasattr(preprocessor, "other_columns_to_scale")

    with pytest.raises(ValueError, match="unsupported by the current runtime"):
        _validate_preprocessor_schema(preprocessor)  # type: ignore[arg-type]
