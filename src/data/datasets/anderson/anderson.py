import pandas as pd
from src.data.datasets.base_cgm import BaseAwesomeCGMLoader
from src.data.datasets.anderson.clean_data import clean_cgm_data


class AndersonDataLoader(BaseAwesomeCGMLoader):
    def __init__(self, ketones_file_path: str | None = None, *args, **kwargs):
        self.ketones_file_path = ketones_file_path
        self.cleaner = clean_cgm_data
        super().__init__(*args, **kwargs)

    @property
    def dataset_name(self):
        """Return the name of the dataset."""
        return "anderson2016"

    def load_data(self):
        super().load_data()
        # now load ketones data as well, if it exists
        if self.ketones_file_path is None:
            return
        # NOTE: sep is '|' since its in a .txt file by default
        ketones_df = pd.read_csv(self.ketones_file_path, low_memory=False, sep="|")
        # rename columns
        ketones_df = ketones_df.rename(
            columns={"DeidentID": "p_num", "DataDtTm": "datetime"}
        )
        # Ensure datetime is parsed
        ketones_df["datetime"] = pd.to_datetime(ketones_df["datetime"], format="mixed")
        self._align_ketone_columns(ketones_df)

        # Merge ketones with train and validation data
        # NOTE: removed this for now, since ketones data is completely misaligned
        # from bgl data (by order of 6 months!)
        # self.train_data = self._merge_ketones(self.train_data, ketones_df)
        # self.validation_data = self._merge_ketones(self.validation_data, ketones_df)

    def _align_ketone_columns(self, df: pd.DataFrame):
        """Standardize column names in ketone file."""
        # Adjust if column names differ
        df.rename(columns={"id": "p_num"}, inplace=True)  # If necessary
        df["datetime"] = pd.to_datetime(df["datetime"])
        df.sort_values(["p_num", "datetime"], inplace=True)

    def _merge_ketones(
        self, cgm_df: pd.DataFrame, ketones_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merges ketones data into CGM data using nearest datetime merge per patient.
        Returns a new dataframe with added ketone values.
        """
        merged_dfs = []

        for patient_id, patient_data in cgm_df.groupby("p_num"):
            ketone_data = ketones_df[ketones_df["p_num"] == patient_id]

            if ketone_data.empty:
                merged_dfs.append(patient_data)
                continue

            merged = pd.merge_asof(
                patient_data.sort_values("datetime"),
                ketone_data.sort_values("datetime"),
                on="datetime",
                direction="nearest",
                tolerance=pd.Timedelta(minutes=10),  # Adjust tolerance as needed
            )

            merged_dfs.append(merged)

        return pd.concat(merged_dfs, ignore_index=True)
