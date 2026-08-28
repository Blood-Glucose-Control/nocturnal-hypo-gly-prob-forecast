"""
Snapshot of every `_prepare_training_data` implementation under `src/models`.
Generated for P1-38 MC2 consolidation review.
"""

# ruff: noqa
# pyright: reportUnusedFunction=false, reportUnusedClass=false


# --- 01. src/models/autogluon_base.py :: AutoGluonBaseModel._prepare_training_data ---
class Snapshot_autogluon_base_AutoGluonBaseModel:
    def _prepare_training_data(
        self,
        train_data: Any,
    ) -> Tuple[Any, None, None]:
        """Convert flat DataFrame to AutoGluon TimeSeriesDataFrame.

        Pipeline: flat_df -> patient_dict -> gap-handled segments ->
        TimeSeriesDataFrame with covariates.

        Args:
            train_data: Flat DataFrame from the registry (all patients
                concatenated with patient_col column).

        Returns:
            Tuple of (TimeSeriesDataFrame, None, None). The Nones satisfy the
            base-class signature (train, val, test); AutoGluon handles
            validation internally via sliding windows.
        """
        config = self._cfg

        patient_dict = convert_to_patient_dict(
            train_data, config.patient_col, config.time_col
        )
        info_print(f"Converted to {len(patient_dict)} patient dicts")

        assert config.min_segment_length is not None
        segments = segment_all_patients(
            patient_dict,
            imputation_threshold_mins=config.imputation_threshold_mins,
            min_segment_length=config.min_segment_length,
            bg_col=config.target_col,
        )
        info_print(f"Gap handling: {len(segments)} segments")

        ts_train = format_segments_for_autogluon(
            segments, config.target_col, config.covariate_cols
        )
        info_print(f"Training data: {ts_train.shape}")

        return (ts_train, None, None)


# --- 02. src/models/base/base_model.py :: BaseTimeSeriesFoundationModel._prepare_training_data ---
class Snapshot_base_model_BaseTimeSeriesFoundationModel:
    def _prepare_training_data(
        self,
        train_data: Any,
    ) -> Tuple[Any, Optional[Any], Optional[Any]]:
        """
        Prepare backend-specific training inputs and optional validation inputs.

        Data splitting is controlled by model configuration.

        Args:
            train_data: Training dataset (will be split based on config)

        Returns:
            Tuple of (train_input, val_input, aux_eval_data). For
            DataLoader-based backends, the first two values are DataLoaders.
            Custom backends may return backend-native structures instead.
        """
        pass


# --- 03. src/models/chronos2/model.py :: Chronos2Forecaster._prepare_training_data ---
class Snapshot_model_Chronos2Forecaster:
    def _prepare_training_data(
        self,
        train_data: Any,
    ) -> Tuple[Any, None, None]:
        """Convert flat DataFrame to AutoGluon TimeSeriesDataFrame.

        Pipeline: flat_df -> patient_dict -> gap-handled segments ->
        TimeSeriesDataFrame with covariates.

        Args:
            train_data: Flat DataFrame from the registry (all patients
                concatenated with patient_col column).

        Returns:
            Tuple of (TimeSeriesDataFrame, None, None). The Nones are
            because the base class signature expects (train, val, test)
            but Chronos-2 does not split — AutoGluon handles validation
            internally via sliding windows.
        """
        config = self.config

        # flat df -> per-patient dict
        patient_dict = convert_to_patient_dict(
            train_data, config.patient_col, config.time_col
        )
        info_print(f"Converted to {len(patient_dict)} patient dicts")

        # gap handling: interpolate small gaps, segment at large gaps
        # min_segment_length is guaranteed non-None by Chronos2Config.__init__
        assert config.min_segment_length is not None
        segments = segment_all_patients(
            patient_dict,
            imputation_threshold_mins=config.imputation_threshold_mins,
            min_segment_length=config.min_segment_length,
            bg_col=config.target_col,
        )
        info_print(f"Gap handling: {len(segments)} segments")

        # Multi-target mode: stack each target col as a separate item
        if config.is_multitarget:
            info_print(f"Multi-target mode: {config.joint_target_cols}")
            ts_train = format_segments_for_autogluon(
                segments, target_cols=config.joint_target_cols
            )
        else:
            all_covariate_cols = list(
                dict.fromkeys(config.covariate_cols + config.known_covariate_cols)
            )
            ts_train = format_segments_for_autogluon(
                segments, config.target_col, all_covariate_cols
            )
        info_print(f"Training data: {ts_train.shape}")

        return (ts_train, None, None)


# --- 04. src/models/darts_base.py :: DartsGlobalModelBase._prepare_training_data ---
class Snapshot_darts_base_DartsGlobalModelBase:
    def _prepare_training_data(
        self,
        train_data: Any,
    ) -> Tuple[List[Any], Optional[List[Any]], Optional[Any]]:
        """Prepare segmented per-series Darts training inputs."""
        if not isinstance(train_data, pd.DataFrame):
            raise TypeError(
                f"{self.__class__.__name__} expects train_data as pandas DataFrame, "
                f"got {type(train_data).__name__}"
            )

        config = self._cfg
        patient_dict = convert_to_patient_dict(
            train_data,
            patient_col=config.patient_col,
            time_col=config.time_col,
        )
        info_print(f"Converted to {len(patient_dict)} patient dicts")

        assert config.min_segment_length is not None
        segments = segment_all_patients(
            patient_dict,
            imputation_threshold_mins=config.imputation_threshold_mins,
            min_segment_length=config.min_segment_length,
            bg_col=config.target_col,
        )
        info_print(f"Gap handling: {len(segments)} segments")
        if not segments:
            raise ValueError(
                "No usable segments after gap handling. "
                "Check imputation_threshold_mins/min_segment_length."
            )

        train_series: List[Any] = []
        covariate_series: List[Any] = []
        expected_delta = pd.Timedelta(minutes=config.interval_mins)
        split_segments = 0
        dropped_short_chunks = 0
        for segment_key, segment_df in segments.items():
            contiguous_chunks = self._split_segment_on_time_gaps(
                segment_df=segment_df,
                expected_delta=expected_delta,
                min_chunk_length=config.min_segment_length,
            )
            split_segments += len(contiguous_chunks)
            if not contiguous_chunks:
                dropped_short_chunks += 1
                logger.warning(
                    "Dropped segment %s after time-gap splitting: no chunk met "
                    "min_segment_length=%s",
                    segment_key,
                    config.min_segment_length,
                )
                continue

            for chunk_df in contiguous_chunks:
                target_ts, cov_ts = self._to_target_and_covariates(chunk_df)
                target_values = np.asarray(target_ts.values(copy=False))
                if not np.isfinite(target_values).all():
                    raise ValueError(
                        "Non-finite target values detected in Darts training series "
                        f"for segment {segment_key}."
                    )
                train_series.append(target_ts)
                if cov_ts is not None:
                    cov_values = np.asarray(cov_ts.values(copy=False))
                    if not np.isfinite(cov_values).all():
                        raise ValueError(
                            "Non-finite covariate values detected in Darts training "
                            f"series for segment {segment_key}."
                        )
                    covariate_series.append(cov_ts)

        if not train_series:
            raise ValueError(
                "No trainable series produced from segmented training data."
            )

        if split_segments != len(segments):
            info_print(
                f"Time-gap split expanded {len(segments)} segments into "
                f"{split_segments} contiguous chunks "
                f"(dropped: {dropped_short_chunks})"
            )
        info_print(f"Prepared {len(train_series)} Darts training series")
        return train_series, (covariate_series or None), None


# --- 05. src/models/moirai/model.py :: MoiraiForecaster._prepare_training_data ---
class Snapshot_model_MoiraiForecaster:
    def _prepare_training_data(
        self, train_data: Any, split: str | None = None
    ) -> tuple[DataLoader, DataLoader | None, DataLoader | None]:
        """Prepare patched training loader used by Moirai fine-tuning."""
        del split
        if self.model is None:
            self.model = self._initialize_model()
        model = cast(Any, self.model)

        tensors = self._prepare_training_tensors(train_data)
        sample_count = len(tensors[0])
        if sample_count == 0:
            raise ValueError("No valid training samples could be extracted")

        patch_size = self._select_patch_size()
        self.config.patch_size = patch_size
        dataset = self._convert_tensors_to_patched_dataset(
            model=model,
            tensors=tensors,
            patch_size=patch_size,
        )
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
        )
        return loader, None, None


# --- 06. src/models/moment/model.py :: MomentForecaster._prepare_training_data ---
class Snapshot_model_MomentForecaster:
    def _prepare_training_data(
        self,
        train_data: Any,
        batch_size: Optional[int] = None,
    ) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
        """Build DataLoaders of (context, target) batches with train/val/test splitting.

        Splits data based on self.config.split_config if available, otherwise uses all data for training.
        """
        pairs = self._get_context_target_pairs(train_data, require_target=True)
        if not pairs:
            raise ValueError("No (context, target) pairs produced from train_data")

        split_config = getattr(self.config, "split_config", None)
        if split_config is None and hasattr(self.config, "data_config"):
            split_config = getattr(self.config.data_config, "split_config", None)
        train_pairs, val_pairs, test_pairs = self._split_context_target_pairs(
            pairs,
            split_config,
        )
        train_loader = self._build_context_target_loader(
            train_pairs,
            batch_size=batch_size,
            shuffle=True,
        )
        val_loader = self._build_context_target_loader(
            val_pairs,
            batch_size=batch_size,
            shuffle=False,
        )
        test_loader = self._build_context_target_loader(
            test_pairs,
            batch_size=batch_size,
            shuffle=False,
        )
        if train_loader is None:
            raise ValueError("No training loader could be built from training pairs")

        return train_loader, val_loader, test_loader


# --- 07. src/models/sundial/model.py :: SundialForecaster._prepare_training_data ---
class Snapshot_model_SundialForecaster:
    def _prepare_training_data(
        self, train_data: Any
    ) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
        del train_data
        raise NotImplementedError("Sundial zero-shot mode does not support training")


# --- 08. src/models/tide/model.py :: TiDEForecaster._prepare_training_data ---
class Snapshot_model_TiDEForecaster:
    def _prepare_training_data(
        self,
        train_data: Any,
    ) -> Tuple[Any, None, None]:
        """Convert flat DataFrame to AutoGluon TimeSeriesDataFrame.

        Pipeline: flat_df -> patient_dict -> gap-handled segments ->
        TimeSeriesDataFrame with covariates.

        Args:
            train_data: Flat DataFrame from the registry (all patients
                concatenated with patient_col column).

        Returns:
            Tuple of (TimeSeriesDataFrame, None, None). The Nones are
            because the base class signature expects (train, val, test)
            but AutoGluon handles validation internally via sliding windows.
        """
        config = self.config

        patient_dict = convert_to_patient_dict(
            train_data, config.patient_col, config.time_col
        )
        info_print(f"Converted to {len(patient_dict)} patient dicts")

        assert config.min_segment_length is not None
        segments = segment_all_patients(
            patient_dict,
            imputation_threshold_mins=config.imputation_threshold_mins,
            min_segment_length=config.min_segment_length,
            bg_col=config.target_col,
        )
        info_print(f"Gap handling: {len(segments)} segments")

        ts_train = format_segments_for_autogluon(
            segments, config.target_col, config.covariate_cols
        )
        info_print(f"Training data: {ts_train.shape}")

        return (ts_train, None, None)


# --- 09. src/models/timegrad/model.py :: TimeGradForecaster._prepare_training_data ---
class Snapshot_model_TimeGradForecaster:
    def _prepare_training_data(
        self, train_data: Any
    ) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
        """Not used — TimeGrad handles data loading internally via GluonTS."""
        raise NotImplementedError(
            "TimeGrad uses GluonTS data loading internally. "
            "Use _train_model() directly."
        )


# --- 10. src/models/timesfm/model.py :: TimesFMForecaster._prepare_training_data ---
class Snapshot_model_TimesFMForecaster:
    def _prepare_training_data(
        self, train_data: Any
    ) -> Tuple[DataLoader, Optional[DataLoader], Optional[Dataset]]:
        """Prepare DataLoaders with gap handling and per-patient windowing.

        Pipeline: extract per-patient DataFrames → gap handling (interpolate
        small gaps, segment at large gaps) → patient-level train/val split →
        sliding windows within each segment.

        Returns:
            (train_loader, val_loader, temporal_eval_dataset) where the third
            element is a Dataset/Subset reserved for the mid-training eval
            callback (not yet wrapped in a DataLoader).
        """
        info_print("Preparing data for TimesFM finetuning...")
        target_col = self.config.target_col
        patient_dfs = self._extract_patient_dataframes(
            train_data, target_col=target_col
        )
        patient_dfs = self._ensure_datetime_index(patient_dfs)
        segments, min_seg_length = self._segment_patient_data(
            patient_dfs=patient_dfs,
            target_col=target_col,
        )
        patient_to_segments = self._group_segments_by_patient(
            segments=segments,
            target_col=target_col,
        )
        train_arrays, val_arrays, train_pids, val_pids = self._build_train_val_arrays(
            patient_to_segments=patient_to_segments,
            min_seg_length=min_seg_length,
        )

        stride = self.config.window_stride or self.config.horizon_length

        train_dataset = self._build_window_dataset(
            patient_series=train_arrays,
            stride=stride,
        )
        val_dataset = self._build_window_dataset(
            patient_series=val_arrays,
            stride=stride,
        )

        info_print(
            f"Patients: {len(train_pids)} train, {len(val_pids)} val | "
            f"Windows: {len(train_dataset):,} train, {len(val_dataset):,} val"
        )

        if len(train_dataset) == 0:
            raise ValueError(
                "TimesFM generated 0 training windows after patient split/windowing. "
                f"patients={len(patient_to_segments)}, "
                f"train_patients={len(train_pids)}, val_patients={len(val_pids)}, "
                f"context_length={self.config.context_length}, "
                f"horizon_length={self.config.horizon_length}, stride={stride}. "
                "Try reducing context/horizon lengths or val_patient_ratio."
            )

        train_loader = self._build_data_loader(train_dataset, shuffle=True)
        val_loader = None
        if len(val_dataset) > 0:
            val_loader = self._build_data_loader(val_dataset, shuffle=False)
        elif len(val_pids) > 0:
            info_print(
                "Validation patient split produced 0 windows; continuing without "
                "patient-level validation loader."
            )

        temporal_eval_dataset = self._build_temporal_eval_dataset(
            patient_to_segments=patient_to_segments
        )

        return train_loader, val_loader, temporal_eval_dataset


# --- 11. src/models/toto/model.py :: TotoForecaster._prepare_training_data ---
class Snapshot_model_TotoForecaster:
    def _prepare_training_data(
        self, train_data: Any
    ) -> Tuple[Any, Optional[Any], Optional[Any]]:
        """Prepare data for Toto fine-tuning.

        Converts the flat DataFrame to a HuggingFace Dataset, then wraps it
        in a FinetuneDataModule. Returns (datamodule, None, None) — Lightning
        handles train/val splitting internally.
        """
        from toto.data.datamodule.finetune_datamodule import (  # pyright: ignore[reportMissingImports]
            FinetuneDataModule,
        )

        hf_dataset = self._dataframe_to_hf_dataset(train_data)

        # Compute context length aligned to patch size
        max_context_length = self.config.context_length
        if max_context_length is None:
            max_context_length = 8 * self._patch_size

        covariate_cols = self.config.covariate_cols or []
        if covariate_cols:
            ev_fields = list(covariate_cols)
            ev_transform_fns = [lambda x: np.asarray(x, dtype=np.float32)] * len(
                covariate_cols
            )
            add_exogenous = True
        else:
            ev_fields = ["feat_dynamic_real"]
            ev_transform_fns = [lambda x: np.asarray(x, dtype=np.float32)]
            add_exogenous = False

        dm = FinetuneDataModule(
            dataset=hf_dataset,
            max_context_length=max_context_length,
            prediction_horizon=self.config.forecast_length,
            patch_size=self._patch_size,
            train_batch_size=self.config.train_batch_size,
            val_batch_size=self.config.val_batch_size,
            num_workers=0,
            num_train_samples=1,
            add_exogenous_features=add_exogenous,
            target_fields=["target"],
            target_transform_fns=[lambda x: np.asarray(x, dtype=np.float32)],
            ev_fields=ev_fields,
            ev_transform_fns=ev_transform_fns,
        )

        return (dm, None, None)


# --- 12. src/models/ttm/model.py :: TTMForecaster._prepare_training_data ---
class Snapshot_model_TTMForecaster:
    def _prepare_training_data(
        self,
        train_data: Any,
    ) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
        """Prepare data loaders for training, validation, and testing.

        Data splitting is controlled by self.config.split_config.

        Args:
            train_data: Training data (DataFrame or dict of patient DataFrames)

        Returns:
            Tuple of train, validation, and test DataLoaders (split based on config)

        Raises:
            ValueError: If train_data is not a DataFrame or dict
            Exception: If data preprocessing fails
        """
        info_print("Preparing data for TTM training...")
        data = self._normalize_training_input(train_data)
        column_specifiers = self._get_or_create_column_specifiers(data)
        self._log_column_specifiers(column_specifiers)
        preprocessor = self._ensure_training_preprocessor(column_specifiers)

        logger.info("\n")
        info_print("Splitting data into train/val/test sets...")
        info_print(f"  Split config: {self.config.split_config}")
        try:
            dset_train, dset_val, dset_test = self._build_training_datasets(
                data=data,
                preprocessor=preprocessor,
            )
            train_loader = self._build_data_loader(
                dset_train,
                shuffle=True,
            )
            val_loader = self._build_optional_data_loader(
                dset_val,
                shuffle=False,
            )
            test_loader = self._build_optional_data_loader(
                dset_test,
                shuffle=False,
            )
            self._log_dataset_sizes(dset_train, dset_val, dset_test)

            return train_loader, val_loader, test_loader

        except Exception as e:
            error_print(f"Failed to prepare data: {str(e)}")
            raise
