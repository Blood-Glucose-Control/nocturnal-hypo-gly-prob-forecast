from typing import cast

import pandas as pd

from src.data.diabetes_datasets.data_loader import get_loader


def test_brist1d_load_data_with_missing_values():
    # This test checks that it works even if data is cached
    kaggle_loader = get_loader(
        data_source_name="kaggle_brisT1D", dataset_type="train", use_cached=True
    )
    assert kaggle_loader is not None
    data = kaggle_loader.processed_data
    data = cast(pd.DataFrame, data)
    assert len(data["p01"]) > 0
    assert len(data["p01"].columns) > 0


def test_brist1d_keep_columns():
    kaggle_loader = get_loader(
        data_source_name="kaggle_brisT1D",
        dataset_type="train",
        keep_columns=["bg_mM", "p_num", "datetime"],
        use_cached=True,
    )
    assert kaggle_loader is not None
    data = kaggle_loader.processed_data
    data = cast(pd.DataFrame, data)
    assert len(data["p01"]) > 0
    assert len(data["p01"].columns) == 2
    assert "bg_mM" in data["p01"].columns
    assert "p_num" in data["p01"].columns
    assert "datetime" == data["p01"].index.name


# NOTE: can be added later
# def test_load_gluroo_data():
#     data = load_data(data_source_name="gluroo", dataset_type="train")
