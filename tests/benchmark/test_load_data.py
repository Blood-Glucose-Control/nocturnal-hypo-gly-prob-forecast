from src.data.diabetes_datasets.data_loader import get_loader


# def test_load_brist1d_data():
#     data = load_data(data_source_name="kaggle_brisT1D", dataset_type="train")
#     assert data is not None
#     assert len(data) > 0
#     assert len(data.columns) > 0


def test_load_brist1d_data_with_missing_values():
    # This test checks that it works even if data is cached
    kaggle_loader = get_loader(
        data_source_name="kaggle_brisT1D", dataset_type="train", use_cached=True
    )
    assert kaggle_loader is not None
    data = kaggle_loader.processed_data
    assert len(data) > 0
    assert len(data.columns) > 0


def test_keep_columns():
    kaggle_loader = get_loader(
        data_source_name="kaggle_brisT1D",
        dataset_type="train",
        keep_columns=["bg-0:00", "p_num", "datetime"],
        use_cached=True,
    )
    assert kaggle_loader is not None
    data = kaggle_loader.processed_data
    assert len(data) > 0
    assert len(data.columns) == 3
    assert "bg-0:00" in data.columns
    assert "p_num" in data.columns


# NOTE: can be added later
# def test_load_gluroo_data():
#     data = load_data(data_source_name="gluroo", dataset_type="train")
