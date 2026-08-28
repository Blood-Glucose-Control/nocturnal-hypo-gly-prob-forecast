# P1-38A Method Contract Consolidation Task

## Objective

Own all method-contract consolidation planning for P1-38 using one matrix across
**all** [src/models/](../src/models/) `*/model.py` modules (TSFM and non-TSFM).
This task is now a hard closeout gate: **P1-38 cannot be completed until P1-38A
is completed**.

## Architectural stance (pushback / best-practice position)

Not everything should be promoted to parent classes.

- Child classes must keep model-identity behavior (backend capability flags,
  model-specific init/training/predict orchestration).
- Shared **logic** should move to shared helpers; child methods should become
  thin wrappers only where helpful.
- Capability properties like `supports_zero_shot`, `training_backend`, and
  `supports_probabilistic_forecast` should exist on each model contract surface
  (directly or via inheritance) and be standardized as tiny implementations,
  not merged into a giant parent method body.

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

### Exclusions

- [src/models/ttm/_deprecated/](../src/models/ttm/_deprecated/) is explicitly
  excluded from this matrix analysis.
- Architecture redesigns, dataset semantic changes, and metric-definition
  changes are out of scope.

### Planned consolidation destinations

- [base_model.py](../src/models/base/base_model.py)
  (`BaseTimeSeriesFoundationModel`)
- [autogluon_base.py](../src/models/autogluon_base.py)
  (`AutoGluonBaseModel`)
- [autogluon_data_utils.py](../src/models/autogluon_data_utils.py)
- New focused helper modules under [src/models/base/](../src/models/base/)
  where logic is backend-agnostic and reused by >=2 model families.

## Matrix methodology

- Numeric cells = LOC of the method/function implementation in that model file.
- `Call refs (AST)` = repository-wide AST usage references for that method name.
- `No refs? = YES` means zero detected usage refs and requires manual review.
- `Expected consolidated method name` is the planned post-consolidation target.
- Analysis excludes `ttm/_deprecated` code and venv folders.

Current snapshot: **142 methods/functions** across **15 model modules**.

<details>
<summary>Expand full all-model LOC + caller-evidence matrix</summary>

| Method / function | Purpose | Call refs (AST) | No refs? | Consolidation decision | Expected consolidated method name | TTM | TimesFM | Moirai | Moment | Chronos2 | Toto | Tide | TimeGrad | PatchTST | TSMixer | DeepAR | TFT | Statistical | NaiveBaseline | Sundial |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| __init__ | Initialize forecaster runtime state and configuration. | 37 |  | Retain contract method, extract shared helper logic | __init__ -> wraps _shared_init | 25 | 4 | 26 | 26 | 10 | x | 8 | 2 | x | x | x | x | x | x | 7 |
| training_backend | Report which backend is used for model training. | 6 |  | Retain in child (standardize one-line override) | training_backend | 7 | 2 | 3 | 12 | 2 | 2 | 2 | 2 | x | x | x | x | x | x | 2 |
| supports_zero_shot | Report whether zero-shot inference is supported. | 12 |  | Retain in child (standardize one-line override) | supports_zero_shot | 2 | 2 | 3 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 3 | 2 |
| supports_probabilistic_forecast | Report whether probabilistic forecasts are supported. | 9 |  | Retain in child (standardize one-line override) | supports_probabilistic_forecast | x | 2 | 3 | 2 | 2 | 2 | 2 | 2 | x | 2 | x | x | x | x | 2 |
| _initialize_model | Initialize or hydrate model weights/components for runtime use. | 5 |  | Retain contract method, extract shared helper logic | _initialize_model -> wraps _shared_initialize_model | 58 | 35 | 48 | 24 | 4 | 19 | 4 | 55 | x | x | x | x | x | x | 24 |
| _prepare_training_data | Prepare normalized training/validation/test inputs for the backend. | 15 |  | Retain contract method, extract shared helper logic | _prepare_training_data -> wraps _shared_prepare_training_data | 51 | 74 | 28 | 39 | 54 | 49 | 40 | 8 | x | x | x | x | x | x | 5 |
| _train_model | Execute model training/fine-tuning for the current backend. | 2 |  | Retain contract method, extract shared helper logic | _train_model -> wraps _shared_train_model | 56 | 48 | 168 | 191 | 93 | 35 | 44 | 39 | x | x | x | x | x | x | 5 |
| _predict | Run single-input inference and return forecast outputs. | 5 |  | Retain contract method, extract shared helper logic | _predict -> wraps _shared_predict | 67 | 104 | 14 | 31 | 32 | 58 | 30 | 54 | x | x | x | x | x | x | 52 |
| _predict_batch | Run multi-episode inference and return per-episode outputs. | 6 |  | Retain contract method, extract shared helper logic | _predict_batch -> wraps _shared_predict_batch | 46 | x | x | 64 | 43 | 29 | 31 | x | x | x | x | x | x | x | x |
| _save_checkpoint | Persist model state and required runtime artifacts. | 4 |  | Retain contract method, extract shared helper logic | _save_checkpoint -> wraps _shared_save_checkpoint | 13 | 12 | 20 | 4 | 21 | 17 | 13 | 13 | x | x | x | x | x | x | 3 |
| _load_checkpoint | Restore model state and required runtime artifacts. | 6 |  | Retain contract method, extract shared helper logic | _load_checkpoint -> wraps _shared_load_checkpoint | 66 | 10 | 48 | 39 | 38 | 34 | 15 | 29 | x | x | x | x | x | x | 3 |
| _ag_item_id | Map an episode ID to the model item_id used for extraction. | 3 |  | Extract shared AutoGluon adapter helper | _ag_shared_ag_item_id | x | x | x | x | 10 | x | x | x | x | x | x | x | x | x | x |
| _apply_univariate_patch | Monkey-patch pytorchts for univariate (target_dim=1) compatibility. | 1 |  | Retain family-specific (no safe consolidation yet) | retain family-specific | x | x | x | x | x | x | x | 29 | x | x | x | x | x | x | x |
| _autogluon_extract | Run fine-tuned model inference and extract specified columns. | 2 |  | Extract shared AutoGluon adapter helper | _ag_shared_autogluon_extract | x | x | x | x | 43 | x | x | x | x | x | x | x | x | x | x |
| _build_autogluon_frequency | Convert interval minutes to model frequency string. | 1 |  | Extract shared AutoGluon adapter helper | _ag_shared_build_autogluon_frequency | x | x | x | x | x | x | 3 | x | x | x | x | x | x | x | x |
| _build_autogluon_predict_kwargs | Helper for AutoGluon adapter shaping/extraction. | 2 |  | Extract shared AutoGluon adapter helper | _ag_shared_build_autogluon_predict_kwargs | x | x | x | x | 9 | x | x | x | x | x | x | x | x | x | x |
| _build_batch_inputs | Build a padded MaskedTimeseries batch for episode-level inference. | 1 |  | Retain family-specific (no safe consolidation yet) | retain family-specific | x | x | x | x | x | 53 | x | x | x | x | x | x | x | x | x |
| _build_context_matrix | Build finite context matrix [time, channels] from selected columns. | 7 |  | Retain family-specific (no safe consolidation yet) | retain family-specific | x | x | x | 9 | x | x | x | x | x | x | x | x | x | x | x |
| _build_context_target_dataset | Helper for data normalization/windowing/dataset assembly. | 1 |  | Extract shared data adapter helper | _shared_data_build_context_target_dataset | x | x | x | 24 | x | x | x | x | x | x | x | x | x | x | x |
| _build_context_target_loader | Helper for data normalization/windowing/dataset assembly. | 3 |  | Extract shared data adapter helper | _shared_data_build_context_target_loader | x | x | x | 19 | x | x | x | x | x | x | x | x | x | x | x |
| _build_data_loader | Helper for data normalization/windowing/dataset assembly. | 4 |  | Extract shared data adapter helper | _shared_data_build_data_loader | 7 | 6 | x | x | x | x | x | x | x | x | x | x | x | x | x |
| _build_finetuning_module | Construct the model Lightning finetuning module from current backbone. | 1 |  | Retain family-specific (no safe consolidation yet) | retain family-specific | x | x | x | x | x | 18 | x | x | x | x | x | x | x | x | x |
| _build_fit_kwargs | Build fit kwargs for TimeSeriesPredictor.fit(). | 1 |  | Extract shared AutoGluon adapter helper | _ag_shared_build_fit_kwargs | x | x | x | x | x | x | 11 | x | x | x | x | x | x | x | x |
| _build_forecast_inputs | Helper for forecast generation/extraction. | 2 |  | Extract shared inference helper | _shared_inference_build_forecast_inputs | x | x | x | 47 | x | x | x | x | x | x | x | x | x | x | x |
| _build_hf_trainer | Helper for training setup/execution. | 1 |  | Extract shared trainer helper | _shared_training_build_hf_trainer | x | 18 | x | x | x | x | x | x | x | x | x | x | x | x | x |
| _build_known_covariates | Build known_covariates DataFrame for all episodes in data. | 1 |  | Extract shared AutoGluon adapter helper | _ag_shared_build_known_covariates | x | x | x | x | 54 | x | x | x | x | x | x | x | x | x | x |
| _build_optional_data_loader | Helper for data normalization/windowing/dataset assembly. | 2 |  | Extract shared data adapter helper | _shared_data_build_optional_data_loader | 9 | x | x | x | x | x | x | x | x | x | x | x | x | x | x |
| _build_prediction_context | Build model context frame with item/timestamp index. | 3 |  | Extract shared AutoGluon adapter helper | _ag_shared_build_prediction_context | x | x | x | x | x | x | 16 | x | x | x | x | x | x | x | x |
| _build_predictor_kwargs | Build TimeSeriesPredictor constructor kwargs for model. | 3 |  | Extract shared AutoGluon adapter helper | _ag_shared_build_predictor_kwargs | x | x | x | x | x | x | 12 | x | x | x | x | x | x | x | x |
| _build_shadow_predictor_snapshot | Helper for AutoGluon adapter shaping/extraction. | 1 |  | Extract shared checkpoint helper | _shared_checkpoint_build_shadow_predictor_snapshot | x | x | x | x | 70 | x | x | x | x | x | x | x | x | x | x |
| _build_temporal_eval_dataset | Build temporal eval windows for mid-training callback. | 1 |  | Extract shared data adapter helper | _shared_data_build_temporal_eval_dataset | x | 58 | x | x | x | x | x | x | x | x | x | x | x | x | x |
| _build_train_val_arrays | Split patient segments into training and validation arrays. | 1 |  | Retain family-specific (no safe consolidation yet) | retain family-specific | x | 36 | x | x | x | x | x | x | x | x | x | x | x | x | x |
| _build_trainer | Helper for training setup/execution. | 1 |  | Extract shared trainer helper | _shared_training_build_trainer | 15 | x | x | x | x | x | x | x | x | x | x | x | x | x | x |
| _build_trainer_kwargs | Build trainer kwargs using either epoch- or step-based limits. | 1 |  | Extract shared trainer helper | _shared_training_build_trainer_kwargs | x | x | x | x | x | 26 | x | x | x | x | x | x | x | x | x |
| _build_trainer_model | Helper for training setup/execution. | 1 |  | Extract shared trainer helper | _shared_training_build_trainer_model | x | 11 | x | x | x | x | x | x | x | x | x | x | x | x | x |
| _build_training_callbacks | Helper for training setup/execution. | 1 |  | Extract shared trainer helper | _shared_training_build_training_callbacks | x | 29 | x | x | x | x | x | x | x | x | x | x | x | x | x |
| _build_training_datasets | Build train/val/test datasets from normalized training input. | 1 |  | Extract shared data adapter helper | _shared_data_build_training_datasets | 18 | x | x | x | x | x | x | x | x | x | x | x | x | x | x |
| _build_variates | Extract BG target + covariate tensors from a single episode. | 3 |  | Retain family-specific (no safe consolidation yet) | retain family-specific | x | x | x | x | x | 11 | x | x | x | x | x | x | x | x | x |
| _build_window_dataset | Helper for data normalization/windowing/dataset assembly. | 2 |  | Extract shared data adapter helper | _shared_data_build_window_dataset | x | 13 | x | x | x | x | x | x | x | x | x | x | x | x | x |
| _build_zero_shot_batch_context_tensor | Build a padded (N, 1, L) context tensor for zero-shot batch inference. | 1 |  | Retain family-specific (no safe consolidation yet) | retain family-specific | x | x | x | x | 28 | x | x | x | x | x | x | x | x | x | x |
| _build_zero_shot_pipeline | Create the zero-shot inference pipeline with shared settings. | 1 |  | Retain family-specific (no safe consolidation yet) | retain family-specific | 20 | x | x | x | x | x | x | x | x | x | x | x | x | x | x |
| _build_zero_shot_pipeline_for_data | Build a zero-shot pipeline and return the resolved target column. | 2 |  | Retain family-specific (no safe consolidation yet) | retain family-specific | 24 | x | x | x | x | x | x | x | x | x | x | x | x | x | x |
| _capture_training_history | Capture in-memory trainer history for metadata/reporting. | 1 |  | Retain family-specific (no safe consolidation yet) | retain family-specific | 16 | x | x | x | x | x | x | x | x | x | x | x | x | x | x |
| _checkpoint_config_payload | Helper for checkpoint path/state handling. | 1 |  | Extract shared checkpoint helper | _shared_checkpoint_checkpoint_config_payload | x | 8 | x | x | x | x | x | x | x | x | x | x | x | x | x |
| _checkpoint_paths | Helper for checkpoint path/state handling. | 2 |  | Extract shared checkpoint helper | _shared_checkpoint_checkpoint_paths | x | 5 | x | x | x | x | x | x | x | x | x | x | x | x | x |
| _collate_fn | Custom collator for HF Trainer. | 2 |  | Retain framework hook, extract internals where shared | retain family-specific | x | 14 | x | x | x | x | x | x | x | x | x | x | x | x | x |
| _collect_batch_predictions | Collect per-episode outputs from model predictions. | 3 |  | Extract shared inference helper | _shared_inference_collect_batch_predictions | x | x | x | x | x | x | 20 | x | x | x | x | x | x | x | x |
| _collect_zero_shot_batch_results | Collect per-episode numpy outputs from model zero-shot tensors. | 1 |  | Retain family-specific (no safe consolidation yet) | retain family-specific | x | x | x | x | 15 | x | x | x | x | x | x | x | x | x | x |
| _compute_input_size | Compute model's GRU input_size from the frequency string. | 1 |  | Retain family-specific (no safe consolidation yet) | retain family-specific | x | x | x | x | x | x | x | 16 | x | x | x | x | x | x | x |
| _compute_trainer_metrics | Compute evaluation metrics for Trainer. | 1 |  | Retain framework hook, extract internals where shared | _shared_training_compute_trainer_metrics | 118 | x | x | x | x | x | x | x | x | x | x | x | x | x | x |
| _compute_validation_loss | Model-specific runtime helper. | 1 |  | Retain family-specific (no safe consolidation yet) | retain family-specific | x | x | x | 26 | x | x | x | x | x | x | x | x | x | x | x |
| _configure_training_environment | Set runtime environment knobs for stable Trainer execution. | 1 |  | Retain family-specific (no safe consolidation yet) | retain family-specific | 18 | x | x | x | x | x | x | x | x | x | x | x | x | x | x |
| _convert_tensors_to_patched_dataset | Helper for data normalization/windowing/dataset assembly. | 1 |  | Extract shared data adapter helper | _shared_data_convert_tensors_to_patched_dataset | x | x | 72 | x | x | x | x | x | x | x | x | x | x | x | x |
| _create_column_specifiers | Create column specifiers for TimeSeriesPreprocessor. | 1 |  | Retain family-specific (no safe consolidation yet) | retain family-specific | 42 | x | x | x | x | x | x | x | x | x | x | x | x | x | x |
| _create_darts_model | Model-specific runtime helper. | 1 |  | Retain family-specific (no safe consolidation yet) | retain family-specific | x | x | x | x | x | x | x | x | x | 35 | x | x | x | x | x |
| _create_training_arguments | Create TrainingArguments for model training. | 2 |  | Extract shared trainer helper | _shared_training_create_training_arguments | 46 | 22 | x | x | x | x | x | x | x | x | x | x | x | x | x |
| _create_training_artifacts | Create checkpoint and logger artifacts for fine-tuning. | 1 |  | Retain family-specific (no safe consolidation yet) | retain family-specific | x | x | x | x | x | 20 | x | x | x | x | x | x | x | x | x |
| _dataframe_to_gluonts | Convert a single-episode DataFrame to a one-entry GluonTS dataset. | 1 |  | Extract shared data adapter helper | _shared_data_dataframe_to_gluonts | x | x | 39 | x | x | x | x | x | x | x | x | x | x | x | x |
| _dataframe_to_hf_dataset | Convert a flat DataFrame to HuggingFace Dataset format for model. | 1 |  | Extract shared data adapter helper | _shared_data_dataframe_to_hf_dataset | x | x | x | x | x | 66 | x | x | x | x | x | x | x | x | x |
| _dataframe_to_list_dataset | Convert a DataFrame with bg_mM + p_num into a GluonTS ListDataset. | 1 |  | Extract shared data adapter helper | _shared_data_dataframe_to_list_dataset | x | x | x | x | x | x | x | 32 | x | x | x | x | x | x | x |
| _empty_training_tensors | Helper for training setup/execution. | 1 |  | Retain family-specific (no safe consolidation yet) | retain family-specific | x | x | 20 | x | x | x | x | x | x | x | x | x | x | x | x |
| _ensure_datetime_index | Ensure each patient frame has DatetimeIndex when available. | 1 |  | Retain family-specific (no safe consolidation yet) | retain family-specific | x | 15 | x | x | x | x | x | x | x | x | x | x | x | x | x |
| _ensure_training_preprocessor | Create or validate the training preprocessor. | 1 |  | Extract shared checkpoint helper | _shared_checkpoint_ensure_training_preprocessor | 19 | x | x | x | x | x | x | x | x | x | x | x | x | x | x |
| _ensure_zs_pipeline | Lazily initialise the zero-shot Chronos2Pipeline. | 3 |  | Retain family-specific (no safe consolidation yet) | retain family-specific | x | x | x | x | 12 | x | x | x | x | x | x | x | x | x | x |
| _episode_ids_from | Return the array of episode IDs present in *data*. | 1 |  | Retain family-specific (no safe consolidation yet) | retain family-specific | x | x | x | x | 5 | x | x | x | x | x | x | x | x | x | x |
| _episode_predictions_frame | Return per-item prediction payload as a DataFrame. | 2 |  | Extract shared inference helper | _shared_inference_episode_predictions_frame | x | x | x | x | x | x | 9 | x | x | x | x | x | x | x | x |
| _episodes_to_gluonts | Convenience wrapper that uses config defaults. | 1 |  | Retain family-specific (no safe consolidation yet) | retain family-specific | x | x | 7 | x | x | x | x | x | x | x | x | x | x | x | x |
| _evaluate_test_loader | Evaluate trainer on the test dataset when available. | 1 |  | Extract shared data adapter helper | _shared_data_evaluate_test_loader | 24 | x | x | x | x | x | x | x | x | x | x | x | x | x | x |
| _extract_bg_forecast | Extract BG (variate 0) from a model Forecast object. | 1 |  | Extract shared inference helper | _shared_inference_extract_bg_forecast | x | x | x | x | x | 9 | x | x | x | x | x | x | x | x | x |
| _extract_episode_predictions | Helper for forecast generation/extraction. | 1 |  | Extract shared inference helper | _shared_inference_extract_episode_predictions | x | x | x | x | 19 | x | x | x | x | x | x | x | x | x | x |
| _extract_mean_forecasts | Helper for forecast generation/extraction. | 1 |  | Extract shared inference helper | _shared_inference_extract_mean_forecasts | x | x | 2 | x | x | x | x | x | x | x | x | x | x | x | x |
| _extract_patient_dataframes | Normalize training input into a patient->DataFrame mapping. | 1 |  | Extract shared data adapter helper | _shared_data_extract_patient_dataframes | x | 34 | x | x | x | x | x | x | x | x | x | x | x | x | x |
| _extract_quantile_forecasts | Helper for probabilistic/quantile forecast handling. | 1 |  | Extract shared inference helper | _shared_inference_extract_quantile_forecasts | x | x | 8 | x | x | x | x | x | x | x | x | x | x | x | x |
| _extract_quantile_predictions | Return quantile predictions and fail if requested levels are unavailable. | 4 |  | Extract shared inference helper | _shared_inference_extract_quantile_predictions | x | x | x | x | x | x | 26 | x | x | x | x | x | x | x | x |
| _extract_timestamps | Get timestamps from DatetimeIndex or 'datetime' column. | 3 |  | Extract shared inference helper | _shared_inference_extract_timestamps | x | x | x | x | x | 7 | x | x | x | x | x | x | x | x | x |
| _forecast_batch | Batch forecasts in a single model forward pass when possible. | 2 |  | Extract shared inference helper | _shared_inference_forecast_batch | x | x | x | 109 | x | x | x | x | x | x | x | x | x | x | x |
| _get_callbacks | Get training callbacks. | 1 |  | Extract shared trainer helper | _shared_training_get_callbacks | 20 | x | x | x | x | x | x | x | x | x | x | x | x | x | x |
| _get_context_target_pairs | Build list of (context, target) from dataset name, DataFrame, or dict of DataFrames. | 1 |  | Retain family-specific (no safe consolidation yet) | retain family-specific | x | x | x | 171 | x | x | x | x | x | x | x | x | x | x | x |
| _get_covariate_columns | Return configured covariate columns that are available in df. | 1 |  | Retain family-specific (no safe consolidation yet) | retain family-specific | x | x | x | 5 | x | x | x | x | x | x | x | x | x | x | x |
| _get_input_columns | Input channel order for model: target first, then covariates. | 5 |  | Retain family-specific (no safe consolidation yet) | retain family-specific | x | x | x | 3 | x | x | x | x | x | x | x | x | x | x | x |
| _get_or_create_column_specifiers | Get cached column specifiers, creating them lazily on first use. | 2 |  | Retain family-specific (no safe consolidation yet) | retain family-specific | 5 | x | x | x | x | x | x | x | x | x | x | x | x | x | x |
| _get_or_create_predictor | Helper for AutoGluon adapter shaping/extraction. | 1 |  | Extract shared inference helper | _shared_inference_get_or_create_predictor | x | x | 7 | x | x | x | x | x | x | x | x | x | x | x | x |
| _get_target_column | Model-specific runtime helper. | 3 |  | Retain family-specific (no safe consolidation yet) | retain family-specific | x | x | x | 23 | x | x | x | x | x | x | x | x | x | x | x |
| _group_segments_by_patient | Group segmented arrays by original patient id. | 1 |  | Extract shared data adapter helper | _shared_data_group_segments_by_patient | x | 13 | x | x | x | x | x | x | x | x | x | x | x | x | x |
| _inverse_scale_predictions | Inverse scale predictions back to original units. | 1 |  | Extract shared inference helper | _shared_inference_inverse_scale_predictions | 102 | x | x | x | x | x | x | x | x | x | x | x | x | x | x |
| _iter_episode_dicts | Normalize dict/list training payloads to a flat episode list. | 1 |  | Retain family-specific (no safe consolidation yet) | retain family-specific | x | x | 13 | x | x | x | x | x | x | x | x | x | x | x | x |
| _list_intermediate_checkpoints | Helper for checkpoint path/state handling. | 1 |  | Extract shared checkpoint helper | _shared_checkpoint_list_intermediate_checkpoints | x | x | x | x | 19 | x | x | x | x | x | x | x | x | x | x |
| _load_darts_model | Model-specific runtime helper. | 1 |  | Retain family-specific (no safe consolidation yet) | retain family-specific | x | x | x | x | x | x | x | x | x | 9 | x | x | x | x | x |
| _load_hf_model_weights | Model-specific runtime helper. | 1 |  | Retain family-specific (no safe consolidation yet) | retain family-specific | x | 16 | x | x | x | x | x | x | x | x | x | x | x | x | x |
| _load_preprocessor_checkpoint | Load preprocessor artifact from known checkpoint locations. | 1 |  | Extract shared checkpoint helper | _shared_checkpoint_load_preprocessor_checkpoint | 20 | x | x | x | x | x | x | x | x | x | x | x | x | x | x |
| _load_saved_checkpoint_config | Helper for checkpoint path/state handling. | 1 |  | Extract shared checkpoint helper | _shared_checkpoint_load_saved_checkpoint_config | x | 11 | x | x | x | x | x | x | x | x | x | x | x | x | x |
| _log_column_specifiers | Model-specific runtime helper. | 1 |  | Retain family-specific (no safe consolidation yet) | retain family-specific | 4 | x | x | x | x | x | x | x | x | x | x | x | x | x | x |
| _log_dataset_sizes | Helper for data normalization/windowing/dataset assembly. | 1 |  | Extract shared data adapter helper | _shared_data_log_dataset_sizes | 9 | x | x | x | x | x | x | x | x | x | x | x | x | x | x |
| _log_training_start | Helper for training setup/execution. | 1 |  | Retain family-specific (no safe consolidation yet) | retain family-specific | x | x | x | x | x | x | 8 | x | x | x | x | x | x | x | x |
| _longest_nan_run | Return the length of the longest contiguous True run in a boolean array. | 1 |  | Retain family-specific (no safe consolidation yet) | retain family-specific | x | 13 | x | x | x | x | x | x | x | x | x | x | x | x | x |
| _materialize_intermediate_checkpoints | Create standalone eval-ready snapshots from HF Trainer checkpoint-N dirs. | 2 |  | Extract shared checkpoint helper | _shared_checkpoint_materialize_intermediate_checkpoints | x | x | x | x | 58 | x | x | x | x | x | x | x | x | x | x |
| _normalize_context_array | Helper for context/covariate shaping. | 1 |  | Extract shared data adapter helper | _shared_data_normalize_context_array | x | x | x | 7 | x | x | x | x | x | x | x | x | x | x | x |
| _normalize_predict_input | Helper for forecast generation/extraction. | 1 |  | Extract shared data adapter helper | _shared_data_normalize_predict_input | x | x | 6 | x | x | x | x | x | x | x | x | x | x | x | x |
| _normalize_split_config | Helper for data normalization/windowing/dataset assembly. | 1 |  | Extract shared data adapter helper | _shared_data_normalize_split_config | x | x | x | 10 | x | x | x | x | x | x | x | x | x | x | x |
| _normalize_training_input | Normalize supported training inputs to a DataFrame. | 1 |  | Extract shared data adapter helper | _shared_data_normalize_training_input | 19 | x | x | x | x | x | x | x | x | x | x | x | x | x | x |
| _optional_moment_import | Import MOMENTPipeline if momentfm is installed. | 2 |  | Retain family-specific (no safe consolidation yet) | retain family-specific | x | x | x | 11 | x | x | x | x | x | x | x | x | x | x | x |
| _predict_batch_fitted | Run one model batch prediction call and extract episode outputs. | 1 |  | Extract shared inference helper | _shared_inference_predict_batch_fitted | x | x | x | x | 32 | x | x | x | x | x | x | x | x | x | x |
| _predict_batch_point | Generate point batch forecasts, chunked across episodes. | 1 |  | Extract shared inference helper | _shared_inference_predict_batch_point | x | x | x | x | x | 15 | x | x | x | x | x | x | x | x | x |
| _predict_batch_quantiles | Generate quantile batch forecasts, chunked across episodes. | 1 |  | Extract shared inference helper | _shared_inference_predict_batch_quantiles | x | x | x | x | x | 27 | x | x | x | x | x | x | x | x | x |
| _predict_batch_zero_shot | Run zero-shot Chronos2Pipeline batch inference. | 1 |  | Extract shared inference helper | _shared_inference_predict_batch_zero_shot | x | x | x | x | 47 | x | x | x | x | x | x | x | x | x | x |
| _predict_impl | Run inference and return mean forecasts. | 3 |  | Extract shared inference helper | _shared_inference_predict_impl | x | x | 53 | x | x | x | x | x | x | x | x | x | x | x | x |
| _predict_quantiles_impl | Internal quantile forecast logic shared by _predict and _predict_batch. | 1 |  | Extract shared inference helper | _shared_inference_predict_quantiles_impl | x | x | x | x | 16 | x | x | x | x | x | x | x | x | x | x |
| _predict_with_context | Run predictor inference on a prebuilt model context frame. | 2 |  | Extract shared AutoGluon adapter helper | _ag_shared_predict_with_context | x | x | x | x | x | x | 11 | x | x | x | x | x | x | x | x |
| _prepare_autogluon_data | Format a raw inference DataFrame into a time-series frame. | 3 |  | Extract shared AutoGluon adapter helper | _ag_shared_prepare_autogluon_data | x | x | x | x | 69 | x | x | x | x | x | x | x | x | x | x |
| _prepare_training_tensors | Convert training data to aligned tensor batches on CPU. | 3 |  | Extract shared data adapter helper | _shared_data_prepare_training_tensors | x | x | 111 | x | x | x | x | x | x | x | x | x | x | x | x |
| _prepare_zero_shot_context | Validate target column and return a model context tensor. | 1 |  | Retain family-specific (no safe consolidation yet) | retain family-specific | x | x | x | x | 17 | x | x | x | x | x | x | x | x | x | x |
| _preprocessor_paths | Return supported preprocessor artifact locations for a checkpoint. | 2 |  | Extract shared checkpoint helper | _shared_checkpoint_preprocessor_paths | 6 | x | x | x | x | x | x | x | x | x | x | x | x | x | x |
| _rel_symlink | Model-specific runtime helper. | 5 |  | Retain family-specific (no safe consolidation yet) | retain family-specific | x | x | x | x | 4 | x | x | x | x | x | x | x | x | x | x |
| _require_initialized_model | Return model object or raise if weights are not initialized. | 3 |  | Retain family-specific (no safe consolidation yet) | retain family-specific | 5 | x | x | x | x | x | x | x | x | x | x | x | x | x | x |
| _resolve_eval_chunk_size | Resolve evaluation chunk size from config. | 4 |  | Retain family-specific (no safe consolidation yet) | retain family-specific | x | x | x | x | x | 11 | x | x | x | x | x | x | x | x | x |
| _resolve_freeze_backbone | Helper for runtime normalization/validation. | 1 |  | Retain family-specific (no safe consolidation yet) | retain family-specific | x | x | x | 7 | x | x | x | x | x | x | x | x | x | x | x |
| _resolve_predictor_path | Resolve predictor path from reference file with fallback semantics. | 1 |  | Extract shared checkpoint helper | _shared_checkpoint_resolve_predictor_path | x | x | x | x | x | x | 20 | x | x | x | x | x | x | x | x |
| _resolve_target_columns | Resolve target columns and fail fast when no valid target remains. | 1 |  | Retain family-specific (no safe consolidation yet) | retain family-specific | 17 | x | x | x | x | x | x | x | x | x | x | x | x | x | x |
| _resolve_trainable_parameters | Helper for training setup/execution. | 1 |  | Retain family-specific (no safe consolidation yet) | retain family-specific | x | x | x | 13 | x | x | x | x | x | x | x | x | x | x | x |
| _resolve_training_input_dtype | Helper for training setup/execution. | 1 |  | Retain family-specific (no safe consolidation yet) | retain family-specific | x | 10 | x | x | x | x | x | x | x | x | x | x | x | x | x |
| _restore_trained_backbone | Restore best-validated backbone, falling back to final weights. | 1 |  | Retain family-specific (no safe consolidation yet) | retain family-specific | x | x | x | x | x | 29 | x | x | x | x | x | x | x | x | x |
| _run_forecast | Run forecaster and return BG predictions as numpy array. | 3 |  | Extract shared inference helper | _shared_inference_run_forecast | x | x | x | x | x | 10 | x | x | x | x | x | x | x | x | x |
| _save_preprocessor_checkpoint | Persist preprocessor artifacts to checkpoint-compatible locations. | 1 |  | Extract shared checkpoint helper | _shared_checkpoint_save_preprocessor_checkpoint | 21 | x | x | x | x | x | x | x | x | x | x | x | x | x | x |
| _save_training_config | Helper for training setup/execution. | 1 |  | Retain family-specific (no safe consolidation yet) | retain family-specific | x | x | x | 4 | x | x | x | x | x | x | x | x | x | x | x |
| _segment_patient_data | Apply gap handling and segment data by patient. | 1 |  | Extract shared data adapter helper | _shared_data_segment_patient_data | x | 23 | x | x | x | x | x | x | x | x | x | x | x | x | x |
| _select_patch_size | Choose a fixed patch size for training. | 2 |  | Retain family-specific (no safe consolidation yet) | retain family-specific | x | x | 18 | x | x | x | x | x | x | x | x | x | x | x | x |
| _slice_masked_timeseries | Model-specific runtime helper. | 2 |  | Retain family-specific (no safe consolidation yet) | retain family-specific | x | x | x | x | x | 11 | x | x | x | x | x | x | x | x | x |
| _split_context_target_pairs | Helper for data normalization/windowing/dataset assembly. | 1 |  | Extract shared data adapter helper | _shared_data_split_context_target_pairs | x | x | x | 30 | x | x | x | x | x | x | x | x | x | x | x |
| _split_train_val_patients | Split patient IDs into train/validation with at least one train patient. | 3 |  | Extract shared data adapter helper | _shared_data_split_train_val_patients | x | 19 | x | x | x | x | x | x | x | x | x | x | x | x | x |
| _timestamps_to_seconds | Convert pandas timestamps to seconds-since-epoch float tensor. | 3 |  | Retain family-specific (no safe consolidation yet) | retain family-specific | x | x | x | x | x | 6 | x | x | x | x | x | x | x | x | x |
| _train_model_info_log | Helper for training setup/execution. | 2 |  | Extract shared trainer helper | _shared_training_train_model_info_log | x | x | x | x | x | x | x | x | 14 | 16 | 12 | 12 | 18 | 6 | x |
| _truncate_segment_for_training | Helper for training setup/execution. | 2 |  | Extract shared data adapter helper | _shared_data_truncate_segment_for_training | x | 7 | x | x | x | x | x | x | x | x | x | x | x | x | x |
| _use_wrapper_normalization | Model-specific runtime helper. | 3 |  | Retain family-specific (no safe consolidation yet) | retain family-specific | x | x | x | 2 | x | x | x | x | x | x | x | x | x | x | x |
| _validate_covariate_columns | Helper for context/covariate shaping. | 2 |  | Retain family-specific (no safe consolidation yet) | retain family-specific | x | x | 12 | x | x | x | x | x | x | x | x | x | x | x | x |
| _validate_preprocessor_schema | Validate preprocessor schema required by the current runtime. | 3 |  | Extract shared checkpoint helper | _shared_checkpoint_validate_preprocessor_schema | 14 | x | x | x | x | x | x | x | x | x | x | x | x | x | x |
| _validate_registered_quantile_levels | Helper for probabilistic/quantile forecast handling. | 4 |  | Extract shared inference helper | _shared_inference_validate_registered_quantile_levels | x | x | x | x | 15 | x | x | x | x | x | x | x | x | x | x |
| _warn_quantiles_not_supported | Helper for probabilistic/quantile forecast handling. | 2 |  | Extract shared inference helper | _shared_inference_warn_quantiles_not_supported | 12 | x | x | x | x | x | x | x | x | x | x | x | x | x | x |
| _write_checkpoint_config | Helper for checkpoint path/state handling. | 1 |  | Extract shared checkpoint helper | _shared_checkpoint_write_checkpoint_config | x | 3 | x | x | x | x | x | x | x | x | x | x | x | x | x |
| _write_snapshot_model_pt | Model-specific runtime helper. | 1 |  | Extract shared checkpoint helper | _shared_checkpoint_write_snapshot_model_pt | x | x | x | x | 11 | x | x | x | x | x | x | x | x | x | x |
| _zero_shot_forecast | Run zero-shot inference via Chronos2Pipeline. | 2 |  | Extract shared inference helper | _shared_inference_zero_shot_forecast | x | x | x | x | 29 | x | x | x | x | x | x | x | x | x | x |
| build_gluonts_dataset | Build a GluonTS ``ListDataset`` from a list of episode dicts. | 1 |  | Extract shared data adapter helper | _shared_data_build_gluonts_dataset | x | x | 63 | x | x | x | x | x | x | x | x | x | x | x | x |
| evaluate | Evaluate model on test data using rolling-window evaluation. | 1 |  | Move to family utility module | MOVE to <family>/utils.py | x | 107 | x | x | x | x | x | x | x | x | x | x | x | x | x |

</details>

## Projected model-level consolidation impact (planning estimate)

Assumption: shared logic is extracted out of model modules; lifecycle contract
methods remain as thin wrappers/properties in child classes.

| Model | Current methods | Projected methods | Methods removed | Current LOC | Projected LOC | Estimated LOC reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TTM | 37 | 20 | 17 | 1041 | 191 | 850 |
| TimesFM | 35 | 16 | 19 | 819 | 128 | 691 |
| Moirai | 27 | 15 | 12 | 1024 | 104 | 920 |
| Moment | 34 | 22 | 12 | 1074 | 312 | 762 |
| Chronos2 | 32 | 17 | 15 | 883 | 119 | 764 |
| Toto | 25 | 18 | 7 | 566 | 193 | 373 |
| Tide | 21 | 12 | 9 | 327 | 46 | 281 |
| TimeGrad | 13 | 12 | 1 | 283 | 79 | 204 |
| PatchTST | 2 | 1 | 1 | 16 | 2 | 14 |
| TSMixer | 5 | 4 | 1 | 64 | 48 | 16 |
| DeepAR | 2 | 1 | 1 | 14 | 2 | 12 |
| TFT | 2 | 1 | 1 | 14 | 2 | 12 |
| Statistical | 2 | 1 | 1 | 20 | 2 | 18 |
| NaiveBaseline | 2 | 1 | 1 | 9 | 2 | 7 |
| Sundial | 10 | 10 | 0 | 105 | 34 | 71 |

## Active model-level consolidation impact (rolling update)

This table is actively updated as methods/LOC move during implementation while
the planning estimate above stays fixed as the baseline target.

| Model | Baseline current methods | Active current methods | Projected methods (target) | Active methods remaining | Baseline current LOC | Active current LOC | Projected LOC (target) | Active LOC remaining | Active method delta vs baseline | Active LOC delta vs baseline |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TTM | 37 | 36 | 20 | 16 | 1041 | 1021 | 191 | 830 | -1 | -20 |
| TimesFM | 35 | 33 | 16 | 17 | 819 | 784 | 128 | 656 | -2 | -35 |
| Moirai | 27 | 24 | 15 | 9 | 1024 | 792 | 104 | 688 | -3 | -232 |
| Moment | 34 | 29 | 22 | 7 | 1074 | 954 | 312 | 642 | -5 | -120 |
| Chronos2 | 32 | 32 | 17 | 15 | 883 | 883 | 119 | 764 | 0 | 0 |
| Toto | 25 | 25 | 18 | 7 | 566 | 566 | 193 | 373 | 0 | 0 |
| Tide | 21 | 21 | 12 | 9 | 327 | 327 | 46 | 281 | 0 | 0 |
| TimeGrad | 13 | 13 | 12 | 1 | 283 | 283 | 79 | 204 | 0 | 0 |
| PatchTST | 2 | 2 | 1 | 1 | 16 | 16 | 2 | 14 | 0 | 0 |
| TSMixer | 5 | 5 | 4 | 1 | 64 | 64 | 48 | 16 | 0 | 0 |
| DeepAR | 2 | 2 | 1 | 1 | 14 | 14 | 2 | 12 | 0 | 0 |
| TFT | 2 | 2 | 1 | 1 | 14 | 14 | 2 | 12 | 0 | 0 |
| Statistical | 2 | 2 | 1 | 1 | 20 | 20 | 2 | 18 | 0 | 0 |
| NaiveBaseline | 2 | 2 | 1 | 1 | 9 | 9 | 2 | 7 | 0 | 0 |
| Sundial | 10 | 10 | 10 | 0 | 105 | 105 | 34 | 71 | 0 | 0 |

## Benefits and negatives (with opinion)

### Benefits

- Reduces duplicated helper logic and test burden across model families.
- Makes model subclasses easier to reason about by narrowing each class to
  model-specific behavior.
- Improves consistency of data, checkpoint, and inference contracts.
- Makes future model additions faster by reusing shared adapters.

### Negatives / risks

- Over-promotion risk: if we move too much into base classes, child classes lose
  clarity and backend-specific behavior becomes harder to audit.
- False unification risk: methods with similar names but different semantics may
  be merged incorrectly.
- Migration risk: changing helper boundaries can introduce subtle inference or
  checkpoint regressions without strong tests.
- Debugging indirection: shared helper layers can hide model-specific bugs if
  naming and ownership are not explicit.

### Decision rule

Promote only when semantics are truly shared; otherwise keep logic in child
classes and standardize signatures/tests only.

## Work packages (detailed)

### MC1 — Lifecycle + checkpoint helper extraction (not lifecycle flattening)

**Target model.py LOC reduction:** **180-260 LOC**

**Methods to standardize but keep in child classes**
- `training_backend`, `supports_zero_shot`, `supports_probabilistic_forecast`
  (contract methods/properties, one-line implementations where possible).

**Methods to keep as child wrappers with shared internals**
- `__init__`, `_initialize_model`, `_prepare_training_data`, `_train_model`,
  `_predict`, `_predict_batch`, `_save_checkpoint`, `_load_checkpoint`.

**Shared helper targets and contracts**
| Expected consolidated method | Destination | Input contract | Output contract |
| --- | --- | --- | --- |
| `_shared_checkpoint_paths` | new helper under [src/models/base/](../src/models/base/) | checkpoint root + artifact policy | canonical path mapping |
| `_shared_save_checkpoint_bundle` | new helper under [src/models/base/](../src/models/base/) | model state + metadata + destination path | persisted artifact bundle |
| `_shared_load_checkpoint_bundle` | new helper under [src/models/base/](../src/models/base/) | checkpoint path + strictness policy | restored state + metadata or explicit error |
| `_shared_preprocessor_artifact_io` | new helper under [src/models/base/](../src/models/base/) | preprocessor object + schema metadata + checkpoint paths | deterministic save/load of preprocessor artifacts |

---

### MC2 — Training-data normalization + dataset assembly

**Target model.py LOC reduction:** **220-320 LOC**

**Shared helper targets and contracts**
| Expected consolidated method | Destination | Input contract | Output contract |
| --- | --- | --- | --- |
| `_shared_data_normalize_training_input` | [base_model.py](../src/models/base/base_model.py) or new base helper | `pd.DataFrame | dict[str, pd.DataFrame] | list[dict]` plus column alias config | canonical sorted panel frame |
| `_shared_data_extract_patient_frames` | new base helper | canonical panel frame + patient/time keys | `dict[str, pd.DataFrame]` per patient |
| `_shared_data_split_train_val_patients` | new base helper | patient IDs + split config/seed | train/val ID sets with non-empty train |
| `_shared_data_build_training_datasets` | new base helper | canonical frame + context/horizon/split policy | backend-ready dataset objects |
| `_shared_data_prepare_training_tensors` | new base helper | canonical episode iterable + target/covariate columns | aligned tensors with deterministic shapes |
| `_shared_data_gluonts_adapter` | new base helper | episode payload(s) + target/covariate mapping | GluonTS-compatible dataset payload |

---

### MC3 — Trainer orchestration reuse

**Target model.py LOC reduction:** **180-260 LOC**

**Shared helper targets and contracts**
| Expected consolidated method | Destination | Input contract | Output contract |
| --- | --- | --- | --- |
| `_shared_training_build_arguments` | new base helper | epochs/steps/lr/batch/device and scheduler settings | validated trainer argument object/dict |
| `_shared_training_build_callbacks` | new base helper | eval cadence + metric hooks + checkpoint policy | ordered callback list |
| `_shared_training_build_trainer` | new base helper | model module + datasets/loaders + args + callbacks | trainer instance ready to train |
| `_shared_training_compute_metrics` | new base helper | prediction outputs + labels + optional quantiles | stable metric dict |
| `_shared_training_finalize_artifacts` | new base helper | trainer output + checkpoint refs + history | normalized training artifact payload |

---

### MC4 — Inference/batch/quantile helper reuse

**Target model.py LOC reduction:** **240-360 LOC**

**Shared helper targets and contracts**
| Expected consolidated method | Destination | Input contract | Output contract |
| --- | --- | --- | --- |
| `_shared_inference_extract_mean_forecasts` | new base inference helper | backend prediction payload + target selector | `np.ndarray` mean forecasts |
| `_shared_inference_extract_quantiles` | new base inference helper | backend prediction payload + requested quantile levels | ordered quantile matrix |
| `_shared_inference_collect_batch_outputs` | new base inference helper | batch predictions + episode mapping | `dict[str, np.ndarray]` keyed by episode |
| `_shared_inference_predict_batch_point` | new base inference helper | canonical batch context + runtime config | deterministic point forecast batch map |
| `_shared_inference_predict_batch_quantiles` | new base inference helper | canonical batch context + quantile requests | deterministic quantile forecast batch map |
| `_shared_inference_predict_batch_zero_shot` | new base inference helper | zero-shot tensor context + episode mapping | zero-shot batch output map |

---

### MC5 — AutoGluon adapter consolidation

**Target model.py LOC reduction:** **40-70 LOC**

**Shared helper targets and contracts**
| Expected consolidated method | Destination | Input contract | Output contract |
| --- | --- | --- | --- |
| `_ag_shared_prepare_data` | [autogluon_data_utils.py](../src/models/autogluon_data_utils.py) | panel frame + patient/time/target/covariate columns | AutoGluon `TimeSeriesDataFrame` |
| `_ag_shared_build_known_covariates` | [autogluon_data_utils.py](../src/models/autogluon_data_utils.py) | future-covariate frame + horizon/episode alignment metadata | horizon-aligned known-covariate frame |
| `_ag_shared_build_predict_kwargs` | [autogluon_base.py](../src/models/autogluon_base.py) | runtime config + quantile settings + optional covariates | stable kwargs for `predictor.predict(...)` |
| `_ag_shared_prediction_context` | [autogluon_data_utils.py](../src/models/autogluon_data_utils.py) | canonical panel input + target/covariate selection | deterministic context frame |
| `_ag_shared_predict_with_context` | [autogluon_base.py](../src/models/autogluon_base.py) | prebuilt context + predict kwargs | standardized prediction payload |
| `_ag_shared_item_id` | [autogluon_data_utils.py](../src/models/autogluon_data_utils.py) | episode identifier | stable AutoGluon item ID string |

---

### MC6 — Module-level utility relocation and dead-surface cleanup

**Target model.py LOC reduction:** **60-100 LOC**

**Repo-wide no-ref audit and disposition**

| Method | Pre-removal evidence | Outside `src/` usage | Decision | Status |
| --- | --- | --- | --- | --- |
| `_extract_ground_truth` | Defined only in [timesfm/model.py](../src/models/timesfm/model.py) and [moment/model.py](../src/models/moment/model.py); no callers. | None in [scripts/](../scripts/) or [tests/](../tests/). | Remove. | Removed in this slice. |
| `create_moirai_model` | Defined in [moirai/model.py](../src/models/moirai/model.py), only re-exported from [moirai/__init__.py](../src/models/moirai/__init__.py). | No calls in workflows/scripts/tests. | Remove convenience factory from runtime module surface. | Removed in this slice. |
| `create_moment_model` | Defined in [moment/model.py](../src/models/moment/model.py), only re-exported from [moment/__init__.py](../src/models/moment/__init__.py). | No calls in workflows/scripts/tests. | Remove convenience factory from runtime module surface. | Removed in this slice. |
| `create_timesfm_model` | Defined in [timesfm/model.py](../src/models/timesfm/model.py), only re-exported from [timesfm/__init__.py](../src/models/timesfm/__init__.py). | No calls in workflows/scripts/tests. | Remove convenience factory from runtime module surface. | Removed in this slice. |
| `evaluate_probabilistic` | Method on Moirai class with no repo callsites. | Notebook-local helper of same name exists in [4.14-ss-moirai-forecasting.ipynb](../docs-internal/notebooks/4.14-ss-moirai-forecasting.ipynb) but does not call the class method. | Remove from model class API. | Removed in this slice. |
| `get_ttm_specific_info` | Defined only in [ttm/model.py](../src/models/ttm/model.py); no callers. | None in workflows/scripts/tests. | Remove; rely on existing base `get_model_info()`. | Removed in this slice. |
| `predict_episodes` | Method on Moirai class with no repo callsites. | No calls in workflows/scripts/tests; only docstring examples before removal. | Remove from model class API. | Removed in this slice. |
| `predict_single_window` | Method on Moment class with no repo callsites. | No calls in workflows/scripts/tests. | Remove from model class API. | Removed in this slice. |
| `predict_with_metadata` | Method on Moment class with no repo callsites. | No calls in workflows/scripts/tests. | Remove from model class API. | Removed in this slice. |

After this slice, the matrix has **0 rows** flagged with `No refs? = YES`.

**Decision logic**
- If there is no active caller in `src/`, `scripts/`, `tests/`, or workflows, default to **remove**.
- Re-keep only if we intentionally preserve an external/public API contract and add direct tests for it.

---

## Acceptance criteria

1. P1-38A is the single source-of-truth matrix for method consolidation.
2. Consolidation decisions clearly separate:
   - child contract methods to retain,
   - shared logic to extract,
   - dead/no-ref candidates to remove or relocate.
3. Each MC package has explicit destination methods and I/O contracts.
4. Projected method/LOC reductions are published per model.
5. P1-38 remains open until MC1-MC6 implementation + validation gates complete.

## Validation gates

- `pytest -q tests/models/test_model_family_contract_suite.py`
- `pytest -q tests/workflows/forecasting/test_model_config_schema_loader.py -k "ttm or timesfm or moirai or moment or chronos2 or toto or tide or timegrad or patchtst or tsmixer or deepar or tft or statistical or naive_baseline or sundial"`
- Family-targeted regression suites for touched slices.
- `SKIP=pyright pre-commit run --files <touched_python_files>`
- Final Pylance diagnostics clean (error-severity) on touched Python files.
