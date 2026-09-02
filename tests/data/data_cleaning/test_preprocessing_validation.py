import pandas as pd
import pytest

from src.data.preprocessing.validation import validate_no_legacy_columns


def test_validate_no_legacy_columns_accepts_carbohydrate_g():
    df = pd.DataFrame({"bg_mM": [5.5], "carbohydrate_g": [15.0]})
    assert validate_no_legacy_columns(df)


def test_validate_no_legacy_columns_rejects_food_g():
    df = pd.DataFrame({"bg_mM": [5.5], "food_g": [15.0]})
    with pytest.raises(ValueError, match="Legacy carbohydrate column names"):
        validate_no_legacy_columns(df)
