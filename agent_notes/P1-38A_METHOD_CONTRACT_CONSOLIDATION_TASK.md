# P1-38A Method Contract Consolidation Task

## Objective

Complete cross-model method-contract consolidation using a single LOC-weighted
matrix across **all** `src/models/*/model.py` modules (including non-TSFM
architectures). This subtask is a hard closeout gate: **P1-38 cannot be marked
complete until P1-38A is complete**.

## Scope

### In scope model modules

- [ttm/model.py](../src/models/ttm/model.py)
- [timesfm/model.py](../src/models/timesfm/model.py)
- [moirai/model.py](../src/models/moirai/model.py)
- [moment/model.py](../src/models/moment/model.py)
- [chronos2/model.py](../src/models/chronos2/model.py)
- [toto/model.py](../src/models/toto/model.py)
- [tide/model.py](../src/models/tide/model.py)
- [timegrad/model.py](../src/models/timegrad/model.py)
- [patchtst/model.py](../src/models/patchtst/model.py)
- [tsmixer/model.py](../src/models/tsmixer/model.py)
- [deepar/model.py](../src/models/deepar/model.py)
- [tft/model.py](../src/models/tft/model.py)
- [statistical/model.py](../src/models/statistical/model.py)
- [naive_baseline/model.py](../src/models/naive_baseline/model.py)
- [sundial/model.py](../src/models/sundial/model.py)

### Planned consolidation destinations

- [base_model.py](../src/models/base/base_model.py) (`BaseTimeSeriesFoundationModel`)
- [autogluon_base.py](../src/models/autogluon_base.py) (`AutoGluonBaseModel`)
- [autogluon_data_utils.py](../src/models/autogluon_data_utils.py) (shared AG adapters)
- New shared helper modules under [src/models/base/](../src/models/base/) when
  behavior is backend-agnostic and used by >=2 families.

### Out of scope

- Model architecture redesigns.
- Metric-definition changes.
- Dataset semantic changes.

## Expanded cross-model LOC matrix (all model modules)

Matrix semantics:
- Numeric cells = implementation LOC in that model module.
- `x` = method/function not present in that model module.
- `Promote and Combine candidate` maps each row to work package ownership.

Current snapshot: **152** distinct methods/functions across **15** model
modules.

<details>
<summary>Expand full all-model method/function LOC matrix</summary>

| Method / function | Purpose | Promote and Combine candidate | TTM | TimesFM | Moirai | Moment | Chronos2 | Toto | Tide | TimeGrad | PatchTST | TSMixer | DeepAR | TFT | Statistical | NaiveBaseline | Sundial |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| __init__ | Initialize forecaster runtime state and configuration. | MC1: promote shared lifecycle/checkpoint contracts in base_model.py. | 25 | 4 | 26 | 26 | 10 | x | 8 | 2 | x | x | x | x | x | x | 7 |
| training_backend | Report which backend is used for model training. | MC1: promote shared lifecycle/checkpoint contracts in base_model.py. | 7 | 2 | 3 | 12 | 2 | 2 | 2 | 2 | x | x | x | x | x | x | 2 |
| supports_zero_shot | Report whether zero-shot inference is supported. | MC1: promote shared lifecycle/checkpoint contracts in base_model.py. | 2 | 2 | 3 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 3 | 2 |
| supports_probabilistic_forecast | Report whether probabilistic forecasts are supported. | MC1: promote shared lifecycle/checkpoint contracts in base_model.py. | x | 2 | 3 | 2 | 2 | 2 | 2 | 2 | x | 2 | x | x | x | x | 2 |
| _initialize_model | Initialize or hydrate model weights/components for runtime use. | MC1: promote shared lifecycle/checkpoint contracts in base_model.py. | 58 | 35 | 48 | 24 | 4 | 19 | 4 | 55 | x | x | x | x | x | x | 24 |
| _prepare_training_data | Prepare normalized training/validation/test inputs for the backend. | MC2: combine into shared training-data adapters. | 51 | 74 | 28 | 39 | 54 | 49 | 40 | 8 | x | x | x | x | x | x | 5 |
| _train_model | Execute model training/fine-tuning for the current backend. | MC3: combine into shared trainer orchestration helpers. | 56 | 48 | 168 | 191 | 93 | 35 | 44 | 39 | x | x | x | x | x | x | 5 |
| _predict | Run single-input inference and return forecast outputs. | MC4: combine into shared inference + quantile extraction helpers. | 67 | 104 | 14 | 31 | 32 | 58 | 30 | 54 | x | x | x | x | x | x | 52 |
| _predict_batch | Run multi-episode inference and return per-episode outputs. | MC4: combine into shared inference + quantile extraction helpers. | 46 | x | x | 64 | 43 | 29 | 31 | x | x | x | x | x | x | x | x |
| _save_checkpoint | Persist model state and required runtime artifacts. | MC1: promote shared lifecycle/checkpoint contracts in base_model.py. | 13 | 12 | 20 | 4 | 21 | 17 | 13 | 13 | x | x | x | x | x | x | 3 |
| _load_checkpoint | Restore model state and required runtime artifacts. | MC1: promote shared lifecycle/checkpoint contracts in base_model.py. | 66 | 10 | 48 | 39 | 38 | 34 | 15 | 29 | x | x | x | x | x | x | 3 |
| _ag_item_id | Map an episode ID to the model item_id used for extraction. | MC5: combine into shared AutoGluon adapter utilities. | x | x | x | x | 10 | x | x | x | x | x | x | x | x | x | x |
| _apply_univariate_patch | Monkey-patch pytorchts for univariate (target_dim=1) compatibility. | x | x | x | x | x | x | x | x | 29 | x | x | x | x | x | x | x |
| _autogluon_extract | Run fine-tuned model inference and extract specified columns. | MC5: combine into shared AutoGluon adapter utilities. | x | x | x | x | 43 | x | x | x | x | x | x | x | x | x | x |
| _build_autogluon_frequency | Convert interval minutes to model frequency string. | MC5: combine into shared AutoGluon adapter utilities. | x | x | x | x | x | x | 3 | x | x | x | x | x | x | x | x |
| _build_autogluon_predict_kwargs | Helper for AutoGluon adapter shaping/extraction. | MC5: combine into shared AutoGluon adapter utilities. | x | x | x | x | 9 | x | x | x | x | x | x | x | x | x | x |
| _build_batch_inputs | Build a padded MaskedTimeseries batch for episode-level inference. | x | x | x | x | x | x | 53 | x | x | x | x | x | x | x | x | x |
| _build_context_matrix | Build finite context matrix [time, channels] from selected columns. | x | x | x | x | 9 | x | x | x | x | x | x | x | x | x | x | x |
| _build_context_target_dataset | Helper for data normalization/windowing/dataset assembly. | MC2: combine into shared training-data adapters. | x | x | x | 24 | x | x | x | x | x | x | x | x | x | x | x |
| _build_context_target_loader | Helper for data normalization/windowing/dataset assembly. | MC2: combine into shared training-data adapters. | x | x | x | 19 | x | x | x | x | x | x | x | x | x | x | x |
| _build_data_loader | Helper for data normalization/windowing/dataset assembly. | MC2: combine into shared training-data adapters. | 7 | 6 | x | x | x | x | x | x | x | x | x | x | x | x | x |
| _build_finetuning_module | Construct the model Lightning finetuning module from current backbone. | x | x | x | x | x | x | 18 | x | x | x | x | x | x | x | x | x |
| _build_fit_kwargs | Build fit kwargs for TimeSeriesPredictor.fit(). | MC5: combine into shared AutoGluon adapter utilities. | x | x | x | x | x | x | 11 | x | x | x | x | x | x | x | x |
| _build_forecast_inputs | Helper for forecast generation/extraction. | MC4: combine into shared inference + quantile extraction helpers. | x | x | x | 47 | x | x | x | x | x | x | x | x | x | x | x |
| _build_hf_trainer | Helper for training setup/execution. | MC3: combine into shared trainer orchestration helpers. | x | 18 | x | x | x | x | x | x | x | x | x | x | x | x | x |
| _build_known_covariates | Build known_covariates DataFrame for all episodes in data. | MC5: combine into shared AutoGluon adapter utilities. | x | x | x | x | 54 | x | x | x | x | x | x | x | x | x | x |
| _build_optional_data_loader | Helper for data normalization/windowing/dataset assembly. | MC2: combine into shared training-data adapters. | 9 | x | x | x | x | x | x | x | x | x | x | x | x | x | x |
| _build_prediction_context | Build model context frame with item/timestamp index. | MC5: combine into shared AutoGluon adapter utilities. | x | x | x | x | x | x | 16 | x | x | x | x | x | x | x | x |
| _build_predictor_kwargs | Build TimeSeriesPredictor constructor kwargs for model. | MC5: combine into shared AutoGluon adapter utilities. | x | x | x | x | x | x | 12 | x | x | x | x | x | x | x | x |
| _build_shadow_predictor_snapshot | Helper for AutoGluon adapter shaping/extraction. | MC1: combine into shared checkpoint IO utilities. | x | x | x | x | 70 | x | x | x | x | x | x | x | x | x | x |
| _build_temporal_eval_dataset | Build temporal eval windows for mid-training callback. | MC2: combine into shared training-data adapters. | x | 58 | x | x | x | x | x | x | x | x | x | x | x | x | x |
| _build_train_val_arrays | Split patient segments into training and validation arrays. | x | x | 36 | x | x | x | x | x | x | x | x | x | x | x | x | x |
| _build_trainer | Helper for training setup/execution. | MC3: combine into shared trainer orchestration helpers. | 15 | x | x | x | x | x | x | x | x | x | x | x | x | x | x |
| _build_trainer_kwargs | Build trainer kwargs using either epoch- or step-based limits. | MC3: combine into shared trainer orchestration helpers. | x | x | x | x | x | 26 | x | x | x | x | x | x | x | x | x |
| _build_trainer_model | Helper for training setup/execution. | MC3: combine into shared trainer orchestration helpers. | x | 11 | x | x | x | x | x | x | x | x | x | x | x | x | x |
| _build_training_callbacks | Helper for training setup/execution. | MC3: combine into shared trainer orchestration helpers. | x | 29 | x | x | x | x | x | x | x | x | x | x | x | x | x |
| _build_training_datasets | Build train/val/test datasets from normalized training input. | MC2: combine into shared training-data adapters. | 18 | x | x | x | x | x | x | x | x | x | x | x | x | x | x |
| _build_variates | Extract BG target + covariate tensors from a single episode. | x | x | x | x | x | x | 11 | x | x | x | x | x | x | x | x | x |
| _build_window_dataset | Helper for data normalization/windowing/dataset assembly. | MC2: combine into shared training-data adapters. | x | 13 | x | x | x | x | x | x | x | x | x | x | x | x | x |
| _build_zero_shot_batch_context_tensor | Build a padded (N, 1, L) context tensor for zero-shot batch inference. | x | x | x | x | x | 28 | x | x | x | x | x | x | x | x | x | x |
| _build_zero_shot_pipeline | Create the zero-shot inference pipeline with shared settings. | x | 20 | x | x | x | x | x | x | x | x | x | x | x | x | x | x |
| _build_zero_shot_pipeline_for_data | Build a zero-shot pipeline and return the resolved target column. | x | 24 | x | x | x | x | x | x | x | x | x | x | x | x | x | x |
| _capture_training_history | Capture in-memory trainer history for metadata/reporting. | x | 16 | x | x | x | x | x | x | x | x | x | x | x | x | x | x |
| _checkpoint_config_payload | Helper for checkpoint path/state handling. | MC1: combine into shared checkpoint IO utilities. | x | 8 | x | x | x | x | x | x | x | x | x | x | x | x | x |
| _checkpoint_paths | Helper for checkpoint path/state handling. | MC1: combine into shared checkpoint IO utilities. | x | 5 | x | x | x | x | x | x | x | x | x | x | x | x | x |
| _collate_fn | Custom collator for HF Trainer. | x | x | 14 | x | x | x | x | x | x | x | x | x | x | x | x | x |
| _collect_batch_predictions | Collect per-episode outputs from model predictions. | MC4: combine into shared inference + quantile extraction helpers. | x | x | x | x | x | x | 20 | x | x | x | x | x | x | x | x |
| _collect_zero_shot_batch_results | Collect per-episode numpy outputs from model zero-shot tensors. | x | x | x | x | x | 15 | x | x | x | x | x | x | x | x | x | x |
| _compute_input_size | Compute model's GRU input_size from the frequency string. | x | x | x | x | x | x | x | x | 16 | x | x | x | x | x | x | x |
| _compute_trainer_metrics | Compute evaluation metrics for Trainer. | MC3: combine into shared trainer orchestration helpers. | 118 | x | x | x | x | x | x | x | x | x | x | x | x | x | x |
| _compute_validation_loss | Model-specific runtime helper. | x | x | x | x | 26 | x | x | x | x | x | x | x | x | x | x | x |
| _configure_training_environment | Set runtime environment knobs for stable Trainer execution. | x | 18 | x | x | x | x | x | x | x | x | x | x | x | x | x | x |
| _convert_tensors_to_patched_dataset | Helper for data normalization/windowing/dataset assembly. | MC2: combine into shared training-data adapters. | x | x | 72 | x | x | x | x | x | x | x | x | x | x | x | x |
| _create_column_specifiers | Create column specifiers for TimeSeriesPreprocessor. | x | 42 | x | x | x | x | x | x | x | x | x | x | x | x | x | x |
| _create_darts_model | Model-specific runtime helper. | x | x | x | x | x | x | x | x | x | x | 35 | x | x | x | x | x |
| _create_training_arguments | Create TrainingArguments for model training. | MC3: combine into shared trainer orchestration helpers. | 46 | 22 | x | x | x | x | x | x | x | x | x | x | x | x | x |
| _create_training_artifacts | Create checkpoint and logger artifacts for fine-tuning. | x | x | x | x | x | x | 20 | x | x | x | x | x | x | x | x | x |
| _dataframe_to_gluonts | Convert a single-episode DataFrame to a one-entry GluonTS dataset. | MC2: combine into shared training-data adapters. | x | x | 39 | x | x | x | x | x | x | x | x | x | x | x | x |
| _dataframe_to_hf_dataset | Convert a flat DataFrame to HuggingFace Dataset format for model. | MC2: combine into shared training-data adapters. | x | x | x | x | x | 66 | x | x | x | x | x | x | x | x | x |
| _dataframe_to_list_dataset | Convert a DataFrame with bg_mM + p_num into a GluonTS ListDataset. | MC2: combine into shared training-data adapters. | x | x | x | x | x | x | x | 32 | x | x | x | x | x | x | x |
| _empty_training_tensors | Helper for training setup/execution. | x | x | x | 20 | x | x | x | x | x | x | x | x | x | x | x | x |
| _ensure_datetime_index | Ensure each patient frame has DatetimeIndex when available. | x | x | 15 | x | x | x | x | x | x | x | x | x | x | x | x | x |
| _ensure_training_preprocessor | Create or validate the training preprocessor. | MC1: combine into shared checkpoint IO utilities. | 19 | x | x | x | x | x | x | x | x | x | x | x | x | x | x |
| _ensure_zs_pipeline | Lazily initialise the zero-shot Chronos2Pipeline. | x | x | x | x | x | 12 | x | x | x | x | x | x | x | x | x | x |
| _episode_ids_from | Return the array of episode IDs present in *data*. | x | x | x | x | x | 5 | x | x | x | x | x | x | x | x | x | x |
| _episode_predictions_frame | Return per-item prediction payload as a DataFrame. | MC4: combine into shared inference + quantile extraction helpers. | x | x | x | x | x | x | 9 | x | x | x | x | x | x | x | x |
| _episodes_to_gluonts | Convenience wrapper that uses config defaults. | x | x | x | 7 | x | x | x | x | x | x | x | x | x | x | x | x |
| _evaluate_test_loader | Evaluate trainer on the test dataset when available. | MC2: combine into shared training-data adapters. | 24 | x | x | x | x | x | x | x | x | x | x | x | x | x | x |
| _extract_bg_forecast | Extract BG (variate 0) from a model Forecast object. | MC4: combine into shared inference + quantile extraction helpers. | x | x | x | x | x | 9 | x | x | x | x | x | x | x | x | x |
| _extract_episode_predictions | Helper for forecast generation/extraction. | MC4: combine into shared inference + quantile extraction helpers. | x | x | x | x | 19 | x | x | x | x | x | x | x | x | x | x |
| _extract_ground_truth | Extract ground truth values from the end of the test data. | MC4: combine into shared inference + quantile extraction helpers. | x | 10 | x | 11 | x | x | x | x | x | x | x | x | x | x | x |
| _extract_mean_forecasts | Helper for forecast generation/extraction. | MC4: combine into shared inference + quantile extraction helpers. | x | x | 2 | x | x | x | x | x | x | x | x | x | x | x | x |
| _extract_patient_dataframes | Normalize training input into a patient->DataFrame mapping. | MC2: combine into shared training-data adapters. | x | 34 | x | x | x | x | x | x | x | x | x | x | x | x | x |
| _extract_quantile_forecasts | Helper for probabilistic/quantile forecast handling. | MC4: combine into shared inference + quantile extraction helpers. | x | x | 8 | x | x | x | x | x | x | x | x | x | x | x | x |
| _extract_quantile_predictions | Return quantile predictions and fail if requested levels are unavailable. | MC4: combine into shared inference + quantile extraction helpers. | x | x | x | x | x | x | 26 | x | x | x | x | x | x | x | x |
| _extract_timestamps | Get timestamps from DatetimeIndex or 'datetime' column. | MC4: combine into shared inference + quantile extraction helpers. | x | x | x | x | x | 7 | x | x | x | x | x | x | x | x | x |
| _forecast_batch | Batch forecasts in a single model forward pass when possible. | MC4: combine into shared inference + quantile extraction helpers. | x | x | x | 109 | x | x | x | x | x | x | x | x | x | x | x |
| _forecast_single | Single univariate forecast with optional wrapper-side normalization. | MC4: combine into shared inference + quantile extraction helpers. | x | x | x | 60 | x | x | x | x | x | x | x | x | x | x | x |
| _get_callbacks | Get training callbacks. | MC3: combine into shared trainer orchestration helpers. | 20 | x | x | x | x | x | x | x | x | x | x | x | x | x | x |
| _get_context_target_pairs | Build list of (context, target) from dataset name, DataFrame, or dict of DataFrames. | x | x | x | x | 171 | x | x | x | x | x | x | x | x | x | x | x |
| _get_covariate_columns | Return configured covariate columns that are available in df. | x | x | x | x | 5 | x | x | x | x | x | x | x | x | x | x | x |
| _get_input_columns | Input channel order for model: target first, then covariates. | x | x | x | x | 3 | x | x | x | x | x | x | x | x | x | x | x |
| _get_or_create_column_specifiers | Get cached column specifiers, creating them lazily on first use. | x | 5 | x | x | x | x | x | x | x | x | x | x | x | x | x | x |
| _get_or_create_predictor | Helper for AutoGluon adapter shaping/extraction. | MC4: combine into shared inference + quantile extraction helpers. | x | x | 7 | x | x | x | x | x | x | x | x | x | x | x | x |
| _get_target_column | Model-specific runtime helper. | x | x | x | x | 23 | x | x | x | x | x | x | x | x | x | x | x |
| _group_segments_by_patient | Group segmented arrays by original patient id. | MC2: combine into shared training-data adapters. | x | 13 | x | x | x | x | x | x | x | x | x | x | x | x | x |
| _inverse_scale_predictions | Inverse scale predictions back to original units. | MC4: combine into shared inference + quantile extraction helpers. | 102 | x | x | x | x | x | x | x | x | x | x | x | x | x | x |
| _iter_episode_dicts | Normalize dict/list training payloads to a flat episode list. | x | x | x | 13 | x | x | x | x | x | x | x | x | x | x | x | x |
| _list_intermediate_checkpoints | Helper for checkpoint path/state handling. | MC1: combine into shared checkpoint IO utilities. | x | x | x | x | 19 | x | x | x | x | x | x | x | x | x | x |
| _load_darts_model | Model-specific runtime helper. | x | x | x | x | x | x | x | x | x | x | 9 | x | x | x | x | x |
| _load_hf_model_weights | Model-specific runtime helper. | x | x | 16 | x | x | x | x | x | x | x | x | x | x | x | x | x |
| _load_preprocessor_checkpoint | Load preprocessor artifact from known checkpoint locations. | MC1: combine into shared checkpoint IO utilities. | 20 | x | x | x | x | x | x | x | x | x | x | x | x | x | x |
| _load_saved_checkpoint_config | Helper for checkpoint path/state handling. | MC1: combine into shared checkpoint IO utilities. | x | 11 | x | x | x | x | x | x | x | x | x | x | x | x | x |
| _log_column_specifiers | Model-specific runtime helper. | x | 4 | x | x | x | x | x | x | x | x | x | x | x | x | x | x |
| _log_dataset_sizes | Helper for data normalization/windowing/dataset assembly. | MC2: combine into shared training-data adapters. | 9 | x | x | x | x | x | x | x | x | x | x | x | x | x | x |
| _log_training_start | Helper for training setup/execution. | x | x | x | x | x | x | x | 8 | x | x | x | x | x | x | x | x |
| _longest_nan_run | Return the length of the longest contiguous True run in a boolean array. | x | x | 13 | x | x | x | x | x | x | x | x | x | x | x | x | x |
| _materialize_intermediate_checkpoints | Create standalone eval-ready snapshots from HF Trainer checkpoint-N dirs. | MC1: combine into shared checkpoint IO utilities. | x | x | x | x | 58 | x | x | x | x | x | x | x | x | x | x |
| _normalize_context_array | Helper for context/covariate shaping. | MC2: combine into shared training-data adapters. | x | x | x | 7 | x | x | x | x | x | x | x | x | x | x | x |
| _normalize_predict_input | Helper for forecast generation/extraction. | MC2: combine into shared training-data adapters. | x | x | 6 | x | x | x | x | x | x | x | x | x | x | x | x |
| _normalize_split_config | Helper for data normalization/windowing/dataset assembly. | MC2: combine into shared training-data adapters. | x | x | x | 10 | x | x | x | x | x | x | x | x | x | x | x |
| _normalize_training_input | Normalize supported training inputs to a DataFrame. | MC2: combine into shared training-data adapters. | 19 | x | x | x | x | x | x | x | x | x | x | x | x | x | x |
| _optional_moment_import | Import MOMENTPipeline if momentfm is installed. | x | x | x | x | 11 | x | x | x | x | x | x | x | x | x | x | x |
| _predict_batch_fitted | Run one model batch prediction call and extract episode outputs. | MC4: combine into shared inference + quantile extraction helpers. | x | x | x | x | 32 | x | x | x | x | x | x | x | x | x | x |
| _predict_batch_point | Generate point batch forecasts, chunked across episodes. | MC4: combine into shared inference + quantile extraction helpers. | x | x | x | x | x | 15 | x | x | x | x | x | x | x | x | x |
| _predict_batch_quantiles | Generate quantile batch forecasts, chunked across episodes. | MC4: combine into shared inference + quantile extraction helpers. | x | x | x | x | x | 27 | x | x | x | x | x | x | x | x | x |
| _predict_batch_zero_shot | Run zero-shot Chronos2Pipeline batch inference. | MC4: combine into shared inference + quantile extraction helpers. | x | x | x | x | 47 | x | x | x | x | x | x | x | x | x | x |
| _predict_impl | Run inference and return mean forecasts. | MC4: combine into shared inference + quantile extraction helpers. | x | x | 53 | x | x | x | x | x | x | x | x | x | x | x | x |
| _predict_quantiles_impl | Internal quantile forecast logic shared by _predict and _predict_batch. | MC4: combine into shared inference + quantile extraction helpers. | x | x | x | x | 16 | x | x | x | x | x | x | x | x | x | x |
| _predict_with_context | Run predictor inference on a prebuilt model context frame. | MC5: combine into shared AutoGluon adapter utilities. | x | x | x | x | x | x | 11 | x | x | x | x | x | x | x | x |
| _prepare_autogluon_data | Format a raw inference DataFrame into a time-series frame. | MC5: combine into shared AutoGluon adapter utilities. | x | x | x | x | 69 | x | x | x | x | x | x | x | x | x | x |
| _prepare_training_tensors | Convert training data to aligned tensor batches on CPU. | MC2: combine into shared training-data adapters. | x | x | 111 | x | x | x | x | x | x | x | x | x | x | x | x |
| _prepare_zero_shot_context | Validate target column and return a model context tensor. | x | x | x | x | x | 17 | x | x | x | x | x | x | x | x | x | x |
| _preprocessor_paths | Return supported preprocessor artifact locations for a checkpoint. | MC1: combine into shared checkpoint IO utilities. | 6 | x | x | x | x | x | x | x | x | x | x | x | x | x | x |
| _rel_symlink | Model-specific runtime helper. | x | x | x | x | x | 4 | x | x | x | x | x | x | x | x | x | x |
| _require_initialized_model | Return model object or raise if weights are not initialized. | x | 5 | x | x | x | x | x | x | x | x | x | x | x | x | x | x |
| _resolve_eval_chunk_size | Resolve evaluation chunk size from config. | x | x | x | x | x | x | 11 | x | x | x | x | x | x | x | x | x |
| _resolve_freeze_backbone | Helper for runtime normalization/validation. | x | x | x | x | 7 | x | x | x | x | x | x | x | x | x | x | x |
| _resolve_predictor_path | Resolve predictor path from reference file with fallback semantics. | MC4: combine into shared inference + quantile extraction helpers. | x | x | x | x | x | x | 20 | x | x | x | x | x | x | x | x |
| _resolve_target_columns | Resolve target columns and fail fast when no valid target remains. | x | 17 | x | x | x | x | x | x | x | x | x | x | x | x | x | x |
| _resolve_trainable_parameters | Helper for training setup/execution. | x | x | x | x | 13 | x | x | x | x | x | x | x | x | x | x | x |
| _resolve_training_input_dtype | Helper for training setup/execution. | x | x | 10 | x | x | x | x | x | x | x | x | x | x | x | x | x |
| _restore_trained_backbone | Restore best-validated backbone, falling back to final weights. | x | x | x | x | x | x | 29 | x | x | x | x | x | x | x | x | x |
| _run_forecast | Run forecaster and return BG predictions as numpy array. | MC4: combine into shared inference + quantile extraction helpers. | x | x | x | x | x | 10 | x | x | x | x | x | x | x | x | x |
| _save_preprocessor_checkpoint | Persist preprocessor artifacts to checkpoint-compatible locations. | MC1: combine into shared checkpoint IO utilities. | 21 | x | x | x | x | x | x | x | x | x | x | x | x | x | x |
| _save_training_config | Helper for training setup/execution. | x | x | x | x | 4 | x | x | x | x | x | x | x | x | x | x | x |
| _segment_patient_data | Apply gap handling and segment data by patient. | MC2: combine into shared training-data adapters. | x | 23 | x | x | x | x | x | x | x | x | x | x | x | x | x |
| _select_patch_size | Choose a fixed patch size for training. | x | x | x | 18 | x | x | x | x | x | x | x | x | x | x | x | x |
| _slice_masked_timeseries | Model-specific runtime helper. | x | x | x | x | x | x | 11 | x | x | x | x | x | x | x | x | x |
| _split_context_target_pairs | Helper for data normalization/windowing/dataset assembly. | MC2: combine into shared training-data adapters. | x | x | x | 30 | x | x | x | x | x | x | x | x | x | x | x |
| _split_train_val_patients | Split patient IDs into train/validation with at least one train patient. | MC2: combine into shared training-data adapters. | x | 19 | x | x | x | x | x | x | x | x | x | x | x | x | x |
| _timestamps_to_seconds | Convert pandas timestamps to seconds-since-epoch float tensor. | x | x | x | x | x | x | 6 | x | x | x | x | x | x | x | x | x |
| _train_model_info_log | Helper for training setup/execution. | MC3: combine into shared trainer orchestration helpers. | x | x | x | x | x | x | x | x | 14 | 16 | 12 | 12 | 18 | 6 | x |
| _truncate_segment_for_training | Helper for training setup/execution. | MC2: combine into shared training-data adapters. | x | 7 | x | x | x | x | x | x | x | x | x | x | x | x | x |
| _use_wrapper_normalization | Model-specific runtime helper. | x | x | x | x | 2 | x | x | x | x | x | x | x | x | x | x | x |
| _validate_covariate_columns | Helper for context/covariate shaping. | x | x | x | 12 | x | x | x | x | x | x | x | x | x | x | x | x |
| _validate_preprocessor_schema | Validate preprocessor schema required by the current runtime. | MC1: combine into shared checkpoint IO utilities. | 14 | x | x | x | x | x | x | x | x | x | x | x | x | x | x |
| _validate_registered_quantile_levels | Helper for probabilistic/quantile forecast handling. | MC4: combine into shared inference + quantile extraction helpers. | x | x | x | x | 15 | x | x | x | x | x | x | x | x | x | x |
| _warn_quantiles_not_supported | Helper for probabilistic/quantile forecast handling. | MC4: combine into shared inference + quantile extraction helpers. | 12 | x | x | x | x | x | x | x | x | x | x | x | x | x | x |
| _write_checkpoint_config | Helper for checkpoint path/state handling. | MC1: combine into shared checkpoint IO utilities. | x | 3 | x | x | x | x | x | x | x | x | x | x | x | x | x |
| _write_snapshot_model_pt | Model-specific runtime helper. | MC1: combine into shared checkpoint IO utilities. | x | x | x | x | 11 | x | x | x | x | x | x | x | x | x | x |
| _zero_shot_forecast | Run zero-shot inference via Chronos2Pipeline. | MC4: combine into shared inference + quantile extraction helpers. | x | x | x | x | 29 | x | x | x | x | x | x | x | x | x | x |
| build_gluonts_dataset | Build a GluonTS ``ListDataset`` from a list of episode dicts. | MC2: combine into shared training-data adapters. | x | x | 63 | x | x | x | x | x | x | x | x | x | x | x | x |
| create_moirai_model | Factory function to create a ``MoiraiForecaster`` with sensible defaults. | MC6: module-level factory/eval helper normalization. | x | x | 70 | x | x | x | x | x | x | x | x | x | x | x | x |
| create_moment_model | Factory to create a model with sensible defaults. | MC6: module-level factory/eval helper normalization. | x | x | x | 16 | x | x | x | x | x | x | x | x | x | x | x |
| create_timesfm_model | Factory function to create a model with sensible defaults. | MC6: module-level factory/eval helper normalization. | x | 25 | x | x | x | x | x | x | x | x | x | x | x | x | x |
| evaluate | Evaluate model on test data using rolling-window evaluation. | MC6: module-level factory/eval helper normalization. | x | 107 | x | x | x | x | x | x | x | x | x | x | x | x | x |
| evaluate_probabilistic | Evaluate model with full probabilistic outputs. | MC6: module-level factory/eval helper normalization. | x | x | 111 | x | x | x | x | x | x | x | x | x | x | x | x |
| get_ttm_specific_info | Get model-specific model information. | x | 20 | x | x | x | x | x | x | x | x | x | x | x | x | x | x |
| predict_episodes | Evaluate model on a list of episodes and return per-episode metrics. | MC6: module-level factory/eval helper normalization. | x | x | 51 | x | x | x | x | x | x | x | x | x | x | x | x |
| predict_single_window | Predict one step ahead for a single context window (for holdout/eval scripts). | MC6: module-level factory/eval helper normalization. | x | x | x | 21 | x | x | x | x | x | x | x | x | x | x | x |
| predict_with_metadata | Make predictions and return dict with metadata. | MC6: module-level factory/eval helper normalization. | x | x | x | 12 | x | x | x | x | x | x | x | x | x | x | x |

</details>

## Work packages (reassessed and detailed)

### MC1 — Lifecycle + checkpoint contract promotion

**Primary reduction target in model.py files:** **180-260 LOC** net reduction.

**Destination surfaces**
- Promote shared lifecycle/checkpoint wrappers into
  [BaseTimeSeriesFoundationModel](../src/models/base/base_model.py).
- Place checkpoint-path and artifact-location helpers in a base shared helper
  module under [src/models/base/](../src/models/base/) and call from concrete
  model classes.

**Methods in scope for promotion/combination**
- Core lifecycle contract rows: `__init__`, `training_backend`,
  `supports_zero_shot`, `supports_probabilistic_forecast`,
  `_initialize_model`, `_save_checkpoint`, `_load_checkpoint`.
- Checkpoint/serialization rows: `_checkpoint_paths`,
  `_checkpoint_config_payload`, `_load_saved_checkpoint_config`,
  `_preprocessor_paths`, `_save_preprocessor_checkpoint`,
  `_load_preprocessor_checkpoint`, `_materialize_intermediate_checkpoints`,
  `_write_checkpoint_config`, `_write_snapshot_model_pt`,
  `_resolve_predictor_path`.

**Methods explicitly targeted for promotion in MC1**

| Method | Destination | Input contract | Output contract |
| --- | --- | --- | --- |
| `training_backend` | `BaseTimeSeriesFoundationModel` in [base_model.py](../src/models/base/base_model.py) | Concrete model instance with resolved config. | Stable backend identifier (`"pytorch"`, `"autogluon"`, etc.). |
| `supports_zero_shot` | `BaseTimeSeriesFoundationModel` in [base_model.py](../src/models/base/base_model.py) | Concrete model instance and capability flags. | Deterministic boolean capability signal. |
| `supports_probabilistic_forecast` | `BaseTimeSeriesFoundationModel` in [base_model.py](../src/models/base/base_model.py) | Concrete model instance and backend feature support. | Deterministic boolean capability signal. |
| `_checkpoint_paths` | shared checkpoint helper in [src/models/base/](../src/models/base/) | `checkpoint_dir`, optional relative artifact names, fallback policy. | Canonical resolved artifact-path mapping. |
| `_save_checkpoint` | shared checkpoint writer in [src/models/base/](../src/models/base/) + thin class wrapper | `checkpoint_dir`, runtime metadata, optional preprocessor state. | Persisted artifact set; raises on write failure. |
| `_load_checkpoint` | shared checkpoint loader in [src/models/base/](../src/models/base/) + thin class wrapper | `checkpoint_dir`, required-artifact policy, optional strictness flags. | Restored model state + metadata; raises explicit contract errors when required artifacts are missing. |

**Promotion I/O contracts**
- **Checkpoint load helper**
  - Input: `checkpoint_dir: str | Path`, optional artifact names / fallback paths.
  - Output: resolved paths + loaded metadata dict; raises explicit error on missing required artifacts.
- **Checkpoint save helper**
  - Input: destination path + runtime metadata payload.
  - Output: persisted artifact set with deterministic filenames.
- **Lifecycle capability contract**
  - Input: model config/runtime state.
  - Output: typed capability booleans + backend name with no side effects.

---

### MC2 — Training-data normalization + dataset assembly consolidation

**Primary reduction target in model.py files:** **220-320 LOC** net reduction.

**Destination surfaces**
- Backend-agnostic input normalization helpers to
  [BaseTimeSeriesFoundationModel](../src/models/base/base_model.py) or a base
  shared helper module.
- AutoGluon-specific frame conversion remains in
  [autogluon_data_utils.py](../src/models/autogluon_data_utils.py).

**Methods in scope for promotion/combination**
- `_prepare_training_data`, `_normalize_training_input`, `_extract_patient_dataframes`,
  `_build_train_val_arrays`, `_split_train_val_patients`, `_segment_patient_data`,
  `_build_training_datasets`, `_build_window_dataset`, `_build_data_loader`,
  `_build_optional_data_loader`, `_prepare_training_tensors`,
  `_convert_tensors_to_patched_dataset`, `build_gluonts_dataset`,
  `_dataframe_to_gluonts`, `_dataframe_to_hf_dataset`.

**Methods explicitly targeted for promotion in MC2**

| Method | Destination | Input contract | Output contract |
| --- | --- | --- | --- |
| `_normalize_training_input` | shared normalization helper in [src/models/base/](../src/models/base/) | `pd.DataFrame | dict[str, pd.DataFrame] | list[dict]` plus config column aliases. | Canonical panel `pd.DataFrame` with validated patient/time/target columns. |
| `_extract_patient_dataframes` | shared normalization helper in [src/models/base/](../src/models/base/) | Canonical panel frame + patient/time keys. | `dict[str, pd.DataFrame]` with chronological sorting guarantees. |
| `_split_train_val_patients` | shared split helper in [src/models/base/](../src/models/base/) | patient ID list + split ratio/seed. | `(train_ids, val_ids)` with at least one train patient. |
| `_build_training_datasets` | backend-agnostic data assembly helper in [src/models/base/](../src/models/base/) | Canonical panel frame + context/horizon + split policy. | Typed train/val/test datasets for downstream trainer code. |
| `_prepare_training_tensors` | tensor adaptation helper in [src/models/base/](../src/models/base/) | Canonical episode iterable + selected columns + device hints. | Aligned tensor payload with deterministic shape/order semantics. |
| `_dataframe_to_gluonts` / `build_gluonts_dataset` | shared GluonTS adapter helper in [src/models/base/](../src/models/base/) | Single or multi-episode frame payload, target/covariate schema. | Contract-safe GluonTS `ListDataset` payload. |

**Promotion I/O contracts**
- **Panel normalization contract**
  - Input: `pd.DataFrame | dict[str, pd.DataFrame] | list[episode-dict]`.
  - Output: canonical, schema-validated panel frame with stable patient/time ordering.
- **Window/dataset assembly contract**
  - Input: canonical panel frame + context/horizon + split policy.
  - Output: backend-ready train/val/test dataset or dataloader triplet.
- **Tensor conversion contract**
  - Input: canonical episode iterable + selected target/covariate columns.
  - Output: aligned tensor batch payload with deterministic shape guarantees.

---

### MC3 — Trainer orchestration consolidation

**Primary reduction target in model.py files:** **180-260 LOC** net reduction.

**Destination surfaces**
- Trainer argument and callback assembly promoted to shared training helpers in
  [src/models/base/](../src/models/base/) and reused by concrete classes.

**Methods in scope for promotion/combination**
- `_train_model`, `_create_training_arguments`, `_build_trainer_kwargs`,
  `_build_hf_trainer`, `_build_training_callbacks`, `_compute_trainer_metrics`,
  `_get_callbacks`, `_resolve_training_input_dtype`, `_train_model_info_log`.

**Methods explicitly targeted for promotion in MC3**

| Method | Destination | Input contract | Output contract |
| --- | --- | --- | --- |
| `_create_training_arguments` | shared trainer-arg helper in [src/models/base/](../src/models/base/) | Typed config (epochs/steps/lr/warmup/batch size/device settings). | Backend-specific argument object/dict with validated defaults. |
| `_build_trainer_kwargs` | shared trainer-arg helper in [src/models/base/](../src/models/base/) | Base args + dataset/runtime metadata. | Canonical kwargs for trainer constructors. |
| `_build_hf_trainer` | shared trainer factory helper in [src/models/base/](../src/models/base/) | model module, tokenizer/processor, datasets, callbacks, args. | Ready-to-train HF Trainer-compatible object. |
| `_build_training_callbacks` / `_get_callbacks` | shared callback assembly helper in [src/models/base/](../src/models/base/) | validation policy, eval cadence, metrics hooks. | Ordered callback list with stable callback contract. |
| `_compute_trainer_metrics` | shared metric callback helper in [src/models/base/](../src/models/base/) | predictions, labels, optional quantile/sample tensors. | Serializable metric dict with stable metric keys. |

**Promotion I/O contracts**
- **Trainer argument builder**
  - Input: typed runtime config (`epochs`, `steps`, `batch_size`, `lr`, device hints).
  - Output: backend-specific argument object/dict with validated defaults.
- **Trainer execution wrapper**
  - Input: model module + datasets/loaders + callbacks.
  - Output: deterministic training artifact payload (history, best checkpoint ref, final weights ref).
- **Metric callback contract**
  - Input: prediction outputs + labels + optional quantile/sample payload.
  - Output: serializable metric dict with stable key names.

---

### MC4 — Inference + batch + quantile consolidation

**Primary reduction target in model.py files:** **240-360 LOC** net reduction.

**Destination surfaces**
- Shared inference extraction and batch result assembly helpers in
  [BaseTimeSeriesFoundationModel](../src/models/base/base_model.py) or a new
  base inference helper module.

**Methods in scope for promotion/combination**
- `_predict`, `_predict_batch`, `_forecast_batch`, `_forecast_single`,
  `_predict_batch_point`, `_predict_batch_quantiles`, `_predict_batch_zero_shot`,
  `_predict_impl`, `_predict_quantiles_impl`, `_extract_mean_forecasts`,
  `_extract_quantile_forecasts`, `_extract_quantile_predictions`,
  `_collect_batch_predictions`, `_collect_zero_shot_batch_results`, `_run_forecast`.

**Methods explicitly targeted for promotion in MC4**

| Method | Destination | Input contract | Output contract |
| --- | --- | --- | --- |
| `_extract_mean_forecasts` | shared inference helper in [src/models/base/](../src/models/base/) | backend-specific prediction payload, target selection. | Ordered mean forecast `np.ndarray` with stable axis semantics. |
| `_extract_quantile_forecasts` | shared inference helper in [src/models/base/](../src/models/base/) | probabilistic backend payload + requested quantile levels. | Quantile matrix ordered by requested levels. |
| `_extract_quantile_predictions` | shared inference helper in [src/models/base/](../src/models/base/) | backend prediction frame/object + level selection. | Episode-aligned quantile arrays; explicit failure on missing levels. |
| `_collect_batch_predictions` | shared batch collector in [src/models/base/](../src/models/base/) | batched prediction payload + episode index mapping. | `dict[str, np.ndarray]` keyed by episode ID. |
| `_predict_batch_point` / `_predict_batch_quantiles` | shared batch orchestrator in [src/models/base/](../src/models/base/) | canonical batch context + per-episode metadata. | Deterministic point/quantile batch outputs with identical key set. |
| `_predict_batch_zero_shot` | shared zero-shot batch helper in [src/models/base/](../src/models/base/) | zero-shot context tensor + episode mapping + horizon. | `dict[str, np.ndarray]` batch outputs for zero-shot forecasters. |

**Promotion I/O contracts**
- **Single inference contract**
  - Input: one canonical context frame/tensor and optional quantile levels.
  - Output: `np.ndarray` point forecast or quantile matrix with explicit shape contract.
- **Batch inference contract**
  - Input: canonical panel + `episode_col` + target specification.
  - Output: `dict[str, np.ndarray]` keyed by episode ID; each value shape-consistent per backend.
- **Quantile extraction contract**
  - Input: backend output object + requested quantile levels.
  - Output: ordered quantile matrix (or explicit failure when requested levels are unavailable).

---

### MC5 — AutoGluon context/covariate adapter consolidation

**Primary reduction target in model.py files:** **40-70 LOC** net reduction.

**Destination surfaces**
- Consolidate in [autogluon_data_utils.py](../src/models/autogluon_data_utils.py)
  and integrate via [AutoGluonBaseModel](../src/models/autogluon_base.py).

**Methods in scope for promotion/combination**
- `_prepare_autogluon_data`, `_build_known_covariates`, `_ag_item_id`,
  `_build_autogluon_frequency`, `_build_autogluon_predict_kwargs`,
  `_build_prediction_context`, `_predict_with_context`.

**Methods explicitly targeted for promotion in MC5**

| Method | Destination | Input contract | Output contract |
| --- | --- | --- | --- |
| `_prepare_autogluon_data` | [autogluon_data_utils.py](../src/models/autogluon_data_utils.py) | panel DataFrame + patient/time/target/covariate columns. | AutoGluon-compatible `TimeSeriesDataFrame`. |
| `_build_known_covariates` | [autogluon_data_utils.py](../src/models/autogluon_data_utils.py) | future-covariate frame + episode/time alignment metadata. | Horizon-aligned known-covariate frame. |
| `_build_autogluon_predict_kwargs` | [autogluon_base.py](../src/models/autogluon_base.py) helper surface | runtime config + quantile settings + optional known covariates. | Stable kwargs dict for `predictor.predict(...)`. |
| `_build_prediction_context` | [autogluon_data_utils.py](../src/models/autogluon_data_utils.py) | canonical input frame + target/covariate selection policy. | Context frame with deterministic item/time index semantics. |
| `_predict_with_context` | [autogluon_base.py](../src/models/autogluon_base.py) helper surface | prebuilt context frame + prediction kwargs. | Standardized prediction payload for extraction helpers. |
| `_ag_item_id` | [autogluon_data_utils.py](../src/models/autogluon_data_utils.py) | episode identifier. | Stable string item ID used across AG adapters. |

**Promotion I/O contracts**
- **AG context frame contract**
  - Input: panel DataFrame + patient/time/target columns + optional covariates.
  - Output: AutoGluon-compatible `TimeSeriesDataFrame` with deterministic item/time index semantics.
- **Known-covariate contract**
  - Input: future covariate frame + episode index map.
  - Output: covariate frame aligned to forecast horizon per episode.
- **Prediction extraction contract**
  - Input: AG predictor result frame + target/quantile selection.
  - Output: stable numpy outputs keyed by episode.

---

### MC6 — Module-level factory/evaluation helper normalization

**Primary reduction target in model.py files:** **60-100 LOC** net reduction.

**Destination surfaces**
- Move reusable module-level helper surfaces to explicit utility modules under
  each family folder (or shared family utility modules) and keep model classes
  focused on runtime lifecycle methods.

**Methods in scope for promotion/combination**
- `create_timesfm_model`, `create_moirai_model`, `create_moment_model`,
  `evaluate`, `evaluate_probabilistic`, `predict_episodes`,
  `predict_single_window`, `predict_with_metadata`.

**Methods explicitly targeted for promotion/relocation in MC6**

| Method | Destination | Input contract | Output contract |
| --- | --- | --- | --- |
| `create_timesfm_model` / `create_moirai_model` / `create_moment_model` | family-local utility modules under each model folder | explicit config object/kwargs and optional overrides. | Constructed forecaster instance with no hidden side effects. |
| `evaluate` / `evaluate_probabilistic` | family-local evaluation helper modules | trained forecaster, dataset, and metric settings. | Typed evaluation metrics payload (point + optional probabilistic metrics). |
| `predict_episodes` / `predict_single_window` / `predict_with_metadata` | family-local inference utility modules | explicit context payload + forecast parameters. | Structured per-episode outputs with explicit metadata fields. |

**Promotion I/O contracts**
- **Factory helper contract**
  - Input: explicit config fields (or config object) + optional overrides.
  - Output: initialized forecaster instance with no hidden global state dependencies.
- **Eval helper contract**
  - Input: trained forecaster + evaluation dataset + metric settings.
  - Output: typed metrics payload, with optional per-episode details as separate field.

## Acceptance criteria

1. P1-38A matrix remains the single source-of-truth for promote/combine routing.
2. Each MC package lands with concrete helper promotion PR slices and no contract
   drift in model train/predict/checkpoint flows.
3. Net model-module LOC decreases are reported per package using the matrix as
   baseline.
4. P1-38 remains open until MC1-MC6 implementation and validation gates are done.

## Validation gates

- `pytest -q tests/models/test_model_family_contract_suite.py`
- `pytest -q tests/workflows/forecasting/test_model_config_schema_loader.py -k "ttm or timesfm or moirai or moment or chronos2 or toto or tide or timegrad or patchtst or tsmixer or deepar or tft or statistical or naive_baseline or sundial"`
- Family-targeted runtime regression suites for touched slices.
- `SKIP=pyright pre-commit run --files <touched_python_files>`
- Final Pylance diagnostics clean (error-severity) on touched Python files.
