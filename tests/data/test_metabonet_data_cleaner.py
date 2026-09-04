import pandas as pd
import pytest

from src.data.diabetes_datasets.metabonet.data_cleaner import (
    build_nested_test_data,
    normalize_metabonet_dataframe,
    split_by_patient_id,
)


def test_normalize_metabonet_dataframe_renames_id_and_date_columns():
    raw_df = pd.DataFrame(
        {
            "id": [1, 1, 2],
            "date": [
                "2024-01-01 00:05:00",
                "2024-01-01 00:00:00",
                "2024-01-01 00:00:00",
            ],
            "bg_mM": [6.0, 5.8, 7.1],
            "dose_units": [0.0, 0.5, 0.2],
            "source_file": ["cohort_a.csv", "cohort_a.csv", "cohort_b.csv"],
        }
    )

    normalized_df = normalize_metabonet_dataframe(
        raw_df,
        split_name="train",
        require_bg=True,
    )

    assert isinstance(normalized_df.index, pd.DatetimeIndex)
    assert normalized_df.index.name == "datetime"
    patient_id_prefixes = sorted(
        {
            patient_id.split("_", 1)[0]
            for patient_id in normalized_df["patient_id"].unique()
        }
    )
    assert patient_id_prefixes == ["1", "2"]
    assert list(normalized_df.index) == sorted(normalized_df.index.tolist())


def test_normalize_metabonet_dataframe_converts_bg_mgdl_to_mmol():
    raw_df = pd.DataFrame(
        {
            "patient_id": ["a"],
            "datetime": ["2024-01-01 00:00:00"],
            "bg_mg_dl": [180],
            "source_file": ["cohort_a.csv"],
        }
    )

    normalized_df = normalize_metabonet_dataframe(
        raw_df,
        split_name="train",
        require_bg=True,
    )

    assert normalized_df["bg_mM"].iloc[0] == pytest.approx(10.0)


def test_normalize_metabonet_dataframe_converts_cgm_to_mmol():
    raw_df = pd.DataFrame(
        {
            "patient_id": ["a"],
            "datetime": ["2024-01-01 00:00:00"],
            "CGM": [180],
            "source_file": ["cohort_a.csv"],
        }
    )

    normalized_df = normalize_metabonet_dataframe(
        raw_df,
        split_name="train",
        require_bg=True,
    )

    assert normalized_df["bg_mM"].iloc[0] == pytest.approx(10.0)


def test_normalize_metabonet_dataframe_requires_bg_for_train_split():
    raw_df = pd.DataFrame(
        {
            "patient_id": ["a"],
            "datetime": ["2024-01-01 00:00:00"],
            "dose_units": [1.0],
            "source_file": ["cohort_a.csv"],
        }
    )

    with pytest.raises(ValueError, match="must contain 'bg_mM'"):
        normalize_metabonet_dataframe(
            raw_df,
            split_name="train",
            require_bg=True,
        )


def test_normalize_metabonet_dataframe_requires_source_file():
    raw_df = pd.DataFrame(
        {
            "patient_id": ["a"],
            "datetime": ["2024-01-01 00:00:00"],
            "bg_mM": [6.0],
        }
    )

    with pytest.raises(ValueError, match="must contain 'source_file'"):
        normalize_metabonet_dataframe(
            raw_df,
            split_name="train",
            require_bg=True,
        )


def test_split_by_patient_id_creates_patient_map():
    normalized_df = pd.DataFrame(
        {
            "patient_id": ["p1", "p2"],
            "bg_mM": [5.6, 6.3],
        },
        index=pd.DatetimeIndex(
            pd.to_datetime(["2024-01-01 00:00:00", "2024-01-02 00:00:00"]),
            name="datetime",
        ),
    )

    patient_map = split_by_patient_id(normalized_df)

    assert set(patient_map) == {"p1", "p2"}
    assert all(isinstance(df.index, pd.DatetimeIndex) for df in patient_map.values())


def test_build_nested_test_data_uses_segment_column_when_available():
    test_df = pd.DataFrame(
        {
            "patient_id": ["p1", "p1", "p1"],
            "row_id": ["r1", "r1", "r2"],
            "bg_mM": [5.1, 5.2, 6.0],
        },
        index=pd.DatetimeIndex(
            pd.to_datetime(
                [
                    "2024-01-01 00:00:00",
                    "2024-01-01 00:05:00",
                    "2024-01-01 01:00:00",
                ]
            ),
            name="datetime",
        ),
    )

    nested = build_nested_test_data(test_df)

    assert set(nested["p1"]) == {"r1", "r2"}
    assert isinstance(nested["p1"]["r1"], pd.DataFrame)
