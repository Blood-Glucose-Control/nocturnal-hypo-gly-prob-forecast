# Data Loader Consistency Assessment

> **Created:** January 27, 2026  
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
| `description` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| `num_patients` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| `patient_ids` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| `data_shape_summary`       | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ |
| `dataset_info` | ❌ | ✅ | ❌ | ✅ | ❌ | ✅ | ❌ |
| `data_metrics` | ❌ | ❌ | ❌ | ✅ | ❌ | ✅ | ❌ |
| **INIT ATTRIBUTES** |||||||||
| `keep_columns` | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `num_validation_days` | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ |
| `train_percentage` | ❌ | ✅ | ❌ | ✅ | ⚠️ (unused) | ✅ | ❌ |
| `use_cached` | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `dataset_type` | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ |
| `parallel` | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ⚠️ (not used) |
| `max_workers` | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| `generic_patient_start_date` | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ | ❌ |
| `config` | ❌ | ⚠️ (unused) | ❌ | ❌ | ⚠️ (unused) | ❌ | ⚠️ (unused) |
| `cache_manager` | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| `dataset_config` | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| `extract_features` | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ |
| **DATA CONTAINERS** |||||||||
| `raw_data` | ✅ | ⚠️ `raw_data_path` | ✅ | ✅ | ✅ | ⚠️ `raw_data_path` | ✅ |
| `processed_data` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `train_data` | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `validation_data` | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `test_data` | ❌ | ❌ | ✅ | ❌ | ✅ | ❌ | ❌ |
| **METADATA** |||||||||
| `train_dt_col_type` | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ |
| `val_dt_col_type` | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ |
| `num_train_days` | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ |
| **PUBLIC METHODS** |||||||||
| `load_data()` | ✅ (abstract) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `load_raw()` | ✅ (abstract) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `get_patient_data(patient_id)` | ❌ | ✅ | ❌ | ✅ | ❌ | ✅ | ❌ |
| `get_combined_data(data_type)` | ❌ | ✅ | ❌ | ✅ | ❌ | ✅ | ❌ |
| `to_dataframe()` | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ |
| `save_processed_data()` | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ |
| `get_validation_day_splits()` | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| `create_validation_table()` | ✅ (base) | ✅ (inherited) | ✅ (inherited) | ✅ (inherited) | ✅ (inherited) | ✅ (inherited) | ✅ (inherited) |
| **PROTECTED METHODS** |||||||||
| `_process_raw_data()` | ✅ (abstract) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `_process_and_cache_data()` | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| `_split_train_validation()` | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| `_load_from_cache()` | ❌ | ❌ | ✅ | ❌ | ✅ | ⚠️ | ❌ |
| `_validate_dataset()` | ❌ | ❌ | ❌ | ✅ | ❌ | ✅ | ❌ |
| `_validate_data()` | ✅ (base) | ✅ (inherited) | ✅ (inherited) | ✅ (inherited) | ✅ (inherited) | ✅ (inherited) | ✅ (inherited) |
| `_determine_date_type()` | ✅ (base) | ✅ (inherited) | ✅ (inherited) | ✅ (inherited) | ✅ (inherited) | ✅ (inherited) | ✅ (inherited) |
| `_extract_patient_stats()` | ✅ (base) | ✅ (inherited) | ✅ (inherited) | ✅ (inherited) | ✅ (inherited) | ✅ (inherited) | ✅ (inherited) |
| `_process_patients_parallel()` | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ |
| `_process_patients_sequential()` | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ |
| `_process_raw_train_data()` | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
| `_process_raw_test_data()` | ❌ | ❌ | ✅ | ❌ | ✅ | ❌ | ❌ |
| `_load_nested_test_data_from_cache()` | ❌ | ❌ | ✅ | ❌ | ✅ | ❌ | ❌ |
| `_clean_and_format_raw_data()` | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| `_get_day_splits()` | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |

---

## Summary Statistics

| Loader | Properties | Init Attrs | Data Containers | Public Methods | Protected Methods |
|--------|------------|------------|-----------------|----------------|-------------------|
| **Aleppo2017** | 5/7 | 10/14 | 4/5 | 5/8 | 4/14 |
| **BrisT1D** | 4/7 | 10/14 | 5/5 | 3/8 | 8/14 |
| **Brown2019** | 5/7 | 6/14 | 4/5 | 5/8 | 5/14 |
| **Lynch2022** | 4/7 | 10/14 | 5/5 | 4/8 | 8/14 |
| **Tamborlane2008** | 5/7 | 13/14 | 4/5 | 6/8 | 6/14 |
| **Gluroo** | 1/7 | 5/14 | 4/5 | 3/8 | 2/14 |

---

## Key Observations

### 1. Properties Inconsistencies
- **`description`**: Only Aleppo2017 and Brown2019 have this
- **`dataset_info`**: Aleppo2017, Brown2019, and Tamborlane2008 have it, but not BrisT1D or Lynch2022
- **`data_metrics`**: Only Brown2019 and Tamborlane2008 compute validation metrics
- **`test_data_shape_summary`**: Only BrisT1D has this (for test data handling)

### 2. Init Attribute Inconsistencies
- **Split method**: Some use `train_percentage` (Aleppo, Brown, Tamborlane), others use `num_validation_days` (BrisT1D, Lynch)
- **`generic_patient_start_date`**: Only BrisT1D, Lynch2022, and Tamborlane2008 use artificial dates
- **`config`**: Three loaders have it but none actually use it (technical debt)
- **`cache_manager`**: Gluroo doesn't use the centralized cache manager

### 3. Method Inconsistencies
- **`get_patient_data()`**: Only 3 loaders have it (Aleppo, Brown, Tamborlane)
- **`get_combined_data()`**: Only 3 loaders have it (Aleppo, Brown, Tamborlane)
- **`to_dataframe()`**: Only Lynch2022 has this alternative approach
- **`_validate_dataset()`**: Only Brown2019 and Tamborlane2008 validate after loading
- **Test data handling**: Only BrisT1D and Lynch2022 handle test sets separately

### 4. Gluroo is Significantly Different
- Single-patient focused (no multi-patient dict structure)
- No cache manager integration
- Missing most common properties and methods
- Has unique methods like `get_validation_day_splits()` and `_get_day_splits()`

---

## Recommended Standardization

### Phase 1: Core Properties (Add to DatasetBase)
1. Add abstract properties to `DatasetBase`:
   - `num_patients` 
   - `patient_ids`
   - `description` (optional, with default)
   - `dataset_info`
   - `train_data_shape_summary`

### Phase 2: Core Methods (Add to DatasetBase)
1. Add abstract or default methods to `DatasetBase`:
   - `get_patient_data(patient_id)` - concrete with default implementation
   - `get_combined_data(data_type)` - concrete with default implementation
   - `_split_train_validation()` - abstract

### Phase 3: Standardize Init Parameters
1. Decide on split strategy: `train_percentage` vs `num_validation_days`
   - Recommend: Support both with a flag to choose method
2. Remove unused `config` parameter from all loaders
3. Add `cache_manager` integration to Gluroo

### Phase 4: Optional Enhancements
1. Add `data_metrics` and `_validate_dataset()` to all loaders
2. Add `save_processed_data()` to all loaders
3. Standardize test data handling across loaders that support it

---

## Files to Update

| File | Priority | Changes Needed |
|------|----------|----------------|
| `dataset_base.py` | HIGH | Add abstract properties, default method implementations |
| `gluroo.py` | HIGH | Major refactor to align with other loaders |
| `bris_t1d.py` | MEDIUM | Add `dataset_info`, `get_patient_data()`, `get_combined_data()` |
| `lynch_2022.py` | MEDIUM | Add `dataset_info`, `get_patient_data()`, `get_combined_data()`, `description` |
| `aleppo_2017.py` | LOW | Add `data_metrics`, `_validate_dataset()` |
| `brown_2019.py` | LOW | Minor: mostly complete |
| `tamborlane_2008.py` | LOW | Add `description`, mostly complete |
