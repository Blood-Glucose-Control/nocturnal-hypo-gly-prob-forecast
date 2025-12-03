# Brown 2019 (DCLP3) Dataset

## Overview

The DCLP3 study compared Closed-Loop Control (Control-IQ) vs Sensor-Augmented Pump therapy in adults with Type 1 diabetes.

- **Participants**: 168 total (125 with pump data, 43 CGM-only)
- **Duration**: ~6 months per patient (Baseline + Post Randomization periods)
- **CGM**: Dexcom G6, 5-minute intervals
- **Pump**: Tandem t:slim X2 with Control-IQ

## Download Instructions

1. Download from https://public.jaeb.org/datasets/diabetes
2. Look for "DCLP3 Public Dataset"
3. Extract to `cache/data/brown_2019/raw/`

Expected structure:
```
cache/data/brown_2019/raw/
    DCLP3 Public Dataset - Release 3 - 2022-08-04/
        Data Files/
            cgm.txt
            Pump_BasalRateChange.txt
            Pump_BolusDelivered.txt
```

## Data Files

| File | Rows | Description |
|------|------|-------------|
| cgm.txt | ~9M | CGM glucose readings |
| Pump_BasalRateChange.txt | ~2.6M | Basal rate change events |
| Pump_BolusDelivered.txt | ~221K | Bolus insulin deliveries |

## Basal Handling

This dataset uses **automated basal** (Control-IQ closed-loop). The algorithm adjusts basal every ~5 minutes based on CGM predictions.

- No explicit `basal_duration_mins` field
- Duration is calculated from sequential rate-change events
- Pipeline uses **forward-fill** strategy: rate persists until next change

See [Tidepool Automated Basal](http://developer.tidepool.org/data-model/device-data/types/basal/automated.html) for the data model.

## Usage

```python
from src.data.diabetes_datasets import Brown2019DataLoader

# Load with default settings
loader = Brown2019DataLoader(use_cached=True)

# Access data
print(f"Patients: {loader.num_patients}")
train_data = loader.train_data      # dict[patient_id, DataFrame]
val_data = loader.validation_data

# Load without preprocessing pipeline (faster)
loader = Brown2019DataLoader(
    use_cached=False,
    run_preprocessing_pipeline=False,
    include_patients_without_pump=False,  # Exclude 43 CGM-only patients
)
```

## Output Columns

| Column | Description |
|--------|-------------|
| datetime (index) | Timestamp (5-min intervals) |
| p_num | Patient ID |
| period | "1. Baseline" or "2. Post Randomization" |
| bg_mM | Blood glucose (mmol/L) |
| rate | Basal rate (U/hr), NaN for CGM-only patients |
| dose_units | Bolus insulin (U) |
| bolus_type | "Standard", "Extended", etc. |

## Citation

Brown SA, et al. Six-Month Randomized, Multicenter Trial of Closed-Loop Control in Type 1 Diabetes. N Engl J Med. 2019.
