# TODO: Moved the original data split logic from gluroo.py to here temporarily.
# Figure out where to put this
 
 # TODO: Move this out
    # def _split_train_validation(
    #     self,
    # ) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    #     """
    #     Split processed data into train and validation dicts per patient.

    #     Uses train_percentage to split each patient's data chronologically.

    #     Returns:
    #         tuple: (train_dict, val_dict) where each is a dict mapping patient IDs to DataFrames
    #     """
    #     train_dict: dict[str, pd.DataFrame] = {}
    #     val_dict: dict[str, pd.DataFrame] = {}

    #     for patient_id, df in self.processed_data.items():
    #         try:
    #             patient_df = df.copy()

    #             # Ensure DatetimeIndex
    #             if not isinstance(patient_df.index, pd.DatetimeIndex):
    #                 if "datetime" in patient_df.columns:
    #                     patient_df = patient_df.sort_values("datetime").set_index(
    #                         "datetime"
    #                     )
    #                 else:
    #                     logger.warning(
    #                         f"Patient {patient_id} skipped: missing 'datetime' column"
    #                     )
    #                     continue

    #             patient_df = patient_df.sort_index()

    #             # Split by percentage
    #             train_df, val_df, _ = get_train_validation_split_by_percentage(
    #                 patient_df, train_percentage=self.train_percentage
    #             )

    #             train_dict[patient_id] = train_df
    #             val_dict[patient_id] = val_df

    #         except Exception as e:
    #             logger.warning(f"Patient {patient_id} skipped due to error: {e}")
    #             continue

    #     return train_dict, val_dict

    # TODO: Move this out
    # def get_validation_day_splits(self, patient_id: str):
    #     """
    #     Generate day-by-day training and testing periods for a specific patient.

    #     For each day in the validation data, yields:
    #     - Current day's data from 6am-12am (training period)
    #     - Next day's data from 12am-6am (testing/prediction period)

    #     Args:
    #         patient_id (str): Identifier for the patient whose data to split.

    #     Yields:
    #         tuple: (patient_id, train_period_data, test_period_data)
    #     """
    #     if self.validation_data is None:
    #         raise ValueError("Validation data is not loaded")

    #     if patient_id not in self.validation_data:
    #         raise ValueError(f"Patient {patient_id} not found in validation data")

    #     patient_data = self.validation_data[patient_id]
    #     for train_period, test_period in self._get_day_splits(patient_data):
    #         yield patient_id, train_period, test_period

    # TODO: Move this out
    # def _get_day_splits(
    #     self,
    #     patient_data: pd.DataFrame,
    #     context_period: tuple[int, int] = (6, 24),
    #     forecast_horizon: tuple[int, int] = (0, 6),
    # ):
    #     """
    #     Split each day's data into context period and forecast horizon.
    #     TODO: Not sure if this is needed.

    #     Args:
    #         patient_data: Data for a single patient with DatetimeIndex
    #         context_period: Start and end hours for context period (default: 6am-midnight)
    #         forecast_horizon: Start and end hours for forecast period (default: midnight-6am)

    #     Yields:
    #         tuple: (context_data, forecast_data)
    #     """
    #     yield from iter_daily_context_forecast_splits(
    #         patient_data,
    #         context_period=context_period,
    #         forecast_horizon=forecast_horizon,
    #     )
