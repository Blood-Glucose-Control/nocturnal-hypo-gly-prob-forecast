# Data Loader Consistency Assessment

> **Created:** January 27, 2026
> **Updated:** January 28, 2026
> **Purpose:** Compare properties and methods across all data loaders to identify inconsistencies and plan standardization

---

## Overview

This document compares the implemented properties and methods across all data loaders in the `src/data/diabetes_datasets/` directory. The goal is to identify gaps and inconsistencies to create a more unified interface.

### Legend
- ✅ = Implemented
- ❌ = Missing
- ⚠️ = Partial/Different implementation

---

## Comparison Table

| Property / Method | DatasetBase (Abstract) | Aleppo2017 | BrisT1D | Brown2019 | Lynch2022 | Tamborlane2008 | Gluroo |
|-------------------|------------------------|------------|---------|-----------|-----------|----------------|--------|
| **PROPERTIES** |||||||||
| `dataset_name` | ✅ (abstract) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `description` | ✅ (abstract) | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| `num_patients` | ✅ (base default) | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| `patient_ids` | ✅ (base default) | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| `data_shape_summary` | ✅ (abstract) | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| `dataset_info` | ❌ | ✅ | ❌ | ✅ | ❌ | ✅ | ❌ |
| `data_metrics` | ❌ | ❌ | ❌ | ✅ | ❌ | ✅ | ❌ |
| ~~`train_data_shape_summary`~~ | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
| ~~`test_data_shape_summary`~~ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
| **INIT ATTRIBUTES** |||||||||
| `keep_columns` | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| ~~`num_validation_days`~~ | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ |
| ~~`train_percentage`~~ | ❌ | ✅ | ❌ | ✅ | ⚠️ (logged as unused) | ✅ | ❌ |
| `use_cached` | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| ~~`dataset_type`~~ | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ | ❌ |
| `parallel` | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ⚠️ (prints warning, not used) |
| `max_workers` | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| `generic_patient_start_date` | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ | ❌ |
| `cache_manager` | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| `dataset_config` | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| `extract_features` | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ |
| **DATA CONTAINERS** |||||||||
| `raw_data` | ✅ (base init) | ⚠️ `raw_data_path` | ✅ | ✅ | ✅ | ⚠️ `raw_data_path` | ✅ |
| `processed_data` | ✅ (base init) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| ~~`train_data`~~ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| ~~`validation_data`~~ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| ~~`test_data`~~ | ❌ | ❌ | ✅ | ❌ | ✅ | ❌ | ❌ |
| **METADATA** |||||||||
| ~~`train_dt_col_type`~~ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ |
| ~~`val_dt_col_type`~~ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ |
| ~~`num_train_days` ~~| ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ |
| **PUBLIC METHODS** |||||||||
| `load_data()` | ✅ (abstract) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `load_raw()` | ✅ (abstract) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `get_patient_data(patient_id)` | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ |
| `get_combined_data(data_type)` | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ |
| `to_dataframe()` | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ |
| `get_validation_day_splits()` | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| `create_validation_table()` | ✅ (base) | ✅ (inherited) | ✅ (inherited) | ✅ (inherited) | ✅ (inherited) | ✅ (inherited) | ✅ (inherited) |
| **PROTECTED METHODS** |||||||||
| `_process_raw_data()` | ✅ (abstract) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `_process_and_cache_data()` | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| `_split_train_validation()` | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| `_load_from_cache()` | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ | ❌ |
| `_validate_dataset()` | ❌ | ❌ | ❌ | ✅ | ❌ | ✅ | ❌ |
| `_validate_data()` | ✅ (base) | ✅ (inherited) | ✅ (inherited) | ✅ (inherited) | ✅ (inherited) | ✅ (inherited) | ✅ (inherited) |
| `_determine_date_type()` | ✅ (base) | ✅ (inherited) | ✅ (inherited) | ✅ (inherited) | ✅ (inherited) | ✅ (inherited) | ✅ (inherited) |
| `_extract_patient_stats()` | ✅ (base) | ✅ (inherited) | ✅ (inherited) | ✅ (inherited) | ✅ (inherited) | ✅ (inherited) | ✅ (inherited) |
| `_process_raw_train_data()` | ❌ | ❌ | ✅ | ❌ | ✅ | ❌ | ❌ |
| `_process_raw_test_data()` | ❌ | ❌ | ✅ | ❌ | ✅ | ❌ | ❌ |
| `_load_nested_test_data_from_cache()` | ❌ | ❌ | ✅ | ❌ | ✅ | ❌ | ❌ |
| `_clean_and_format_raw_data()` | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| `_get_day_splits()` | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |

---

## Summary Statistics

| Loader | Properties | Init Attrs | Data Containers | Public Methods | Protected Methods |
|--------|------------|------------|-----------------|----------------|-------------------|
| **Aleppo2017** | 6/9 | 8/11 | 4/5 | 4/6 | 4/12 |
| **BrisT1D** | 7/9 | 9/11 | 5/5 | 3/6 | 7/12 |
| **Brown2019** | 7/9 | 7/11 | 4/5 | 3/6 | 5/12 |
| **Lynch2022** | 5/9 | 10/11 | 5/5 | 4/6 | 7/12 |
| **Tamborlane2008** | 7/9 | 10/11 | 4/5 | 4/6 | 6/12 |
| **Gluroo** | 1/9 | 5/11 | 4/5 | 3/6 | 3/12 |

---

## Key Observations

### 1. Properties Inconsistencies
- **`description`**: Now implemented in all loaders except Gluroo ✅ (previously noted as missing in several)
- **`data_shape_summary`**: Now implemented in BrisT1D ✅ (previously marked as missing)
- **`dataset_info`**: Aleppo2017, Brown2019, and Tamborlane2008 have it, but not BrisT1D or Lynch2022
- **`data_metrics`**: Only Brown2019 and Tamborlane2008 compute validation metrics
- **`train_data_shape_summary`**: Only BrisT1D and Brown2019 have this property
- **`test_data_shape_summary`**: Only BrisT1D has this (for test data handling)

### 2. Init Attribute Inconsistencies
- **Split method**: Some use `train_percentage` (Aleppo, Brown, Tamborlane), others use `num_validation_days` (BrisT1D, Lynch). Aleppo now supports both.
- **`generic_patient_start_date`**: Only BrisT1D, Lynch2022, and Tamborlane2008 use artificial dates
- **`config` parameter**: ✅ **RESOLVED** - Removed from all loaders (previously unused in Aleppo, Lynch, Gluroo)
- **`dataset_type`**: Aleppo2017 does not have this, unlike BrisT1D, Lynch2022, and Tamborlane2008
- **`cache_manager`**: Gluroo doesn't use the centralized cache manager

### 3. Method Inconsistencies
- **`get_patient_data()`**: Only 2 loaders have it (Aleppo, Tamborlane) - Brown2019 no longer has it
- **`get_combined_data()`**: Only 2 loaders have it (Aleppo, Tamborlane) - Brown2019 no longer has it
- **`to_dataframe()`**: Only Lynch2022 has this alternative approach
- **`_validate_dataset()`**: Only Brown2019 and Tamborlane2008 validate after loading
- **`_process_and_cache_data()`**: Tamborlane2008 now has this ✅ (previously missing)
- **Test data handling**: Only BrisT1D and Lynch2022 handle test sets separately

### 4. Gluroo is Significantly Different
- Single-patient focused (no multi-patient dict structure)
- No cache manager integration
- Missing most common properties and methods
- Has unique methods like `get_validation_day_splits()` and `_get_day_splits()`
- `parallel` parameter exists but prints a warning that it's not implemented

---

## Recommended Standardization

### Phase 1: Core Properties (Add to DatasetBase) - PARTIALLY COMPLETE
1. ✅ `num_patients` - Now has default implementation in base class
2. ✅ `patient_ids` - Now has default implementation in base class
3. ✅ `description` - Now abstract in base class, implemented in all except Gluroo
4. ✅ `data_shape_summary` - Now abstract in base class, implemented in all except Gluroo
5. ❌ `dataset_info` - Still missing from BrisT1D, Lynch2022, Gluroo

### Phase 2: Core Methods (Add to DatasetBase) - IN PROGRESS
1. ❌ `get_patient_data(patient_id)` - Only in Aleppo2017, Tamborlane2008
2. ❌ `get_combined_data(data_type)` - Only in Aleppo2017, Tamborlane2008
3. ❌ `_split_train_validation()` - Not in base class (implemented individually in each loader)

### Phase 3: Standardize Init Parameters - PARTIALLY COMPLETE
1. ✅ Removed unused `config` parameter from all loaders
2. ❌ Need to decide on split strategy: `train_percentage` vs `num_validation_days`
   - Recommend: Support both with a flag to choose method
   - Aleppo2017 now supports both parameters
3. ❌ Add `cache_manager` integration to Gluroo

### Phase 4: Optional Enhancements
1. ❌ Add `data_metrics` and `_validate_dataset()` to all loaders
2. ❌ Standardize test data handling across loaders that support it
3. ❌ Add `description` to Gluroo

---

## Files to Update

| File | Priority | Changes Needed |
|------|----------|----------------|
| `dataset_base.py` | MEDIUM | Add default `get_patient_data()`, `get_combined_data()` implementations |
| `gluroo.py` | HIGH | Major refactor: add description, num_patients, patient_ids, data_shape_summary, cache_manager integration |
| `bris_t1d.py` | LOW | Add `dataset_info`, `get_patient_data()`, `get_combined_data()` |
| `lynch_2022.py` | LOW | Add `dataset_info`, `get_patient_data()`, `get_combined_data()` |
| `brown_2019.py` | LOW | Add `get_patient_data()`, `get_combined_data()` (were removed?) |
| `aleppo_2017.py` | LOW | Add `data_metrics`, `_validate_dataset()` |
| `tamborlane_2008.py` | LOW | Mostly complete |

---

## Recent Changes (January 28, 2026)

1. **Removed `config` parameter**: Previously marked as unused technical debt in Aleppo, Lynch, and Gluroo. Now removed from all loaders.
2. **BrisT1D now has `data_shape_summary`**: Previously marked as missing.
3. **`description` is now abstract in base class**: All loaders except Gluroo implement it.
4. **`num_patients` and `patient_ids` have base defaults**: These now have default implementations in DatasetBase.
5. **Brown2019 no longer has `get_patient_data()` and `get_combined_data()`**: These appear to have been removed.
6. **Tamborlane2008 now has `_process_and_cache_data()`**: Previously marked as missing.
