# Data Holdout System Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Data Holdout System                             │
│                                                                         │
│  ┌────────────────────┐      ┌────────────────────┐                     │
│  │   Configuration    │      │   Split Manager    │                     │
│  │   (holdout_config) │─────▶│ (holdout_manager)  │                     │
│  └────────────────────┘      └────────────────────┘                     │
│           │                            │                                │
│           │                            │                                │
│           ▼                            ▼                                │
│  ┌────────────────────────────────────────────────┐                     │
│  │         Dataset Registry                       │                     │
│  │       (dataset_registry)                       │                     │
│  │                                                │                     │
│  │  • load_training_data()    ✅ TRAINING         │                     │
│  │  • load_holdout_data()     🔒 EVALUATION       │                     │
│  │  • load_split_data()       📊 BOTH             │                     │
│  └────────────────────────────────────────────────┘                     │
│           │                                                             │
│           ▼                                                             │
│  ┌────────────────────────────────────────────────┐                     │
│  │           Your Training Pipeline               │                     │
│  └────────────────────────────────────────────────┘                     │
└─────────────────────────────────────────────────────────────────────────┘
```

## Data Flow

```
┌──────────────┐
│  Raw Dataset │
│  (All Data)  │
└──────┬───────┘
       │
       ▼
┌──────────────────────┐
│  Holdout Config      │
│  ┌────────────────┐  │
│  │ temporal: 20%  │  │
│  │ patients: [..] │  │
│  └────────────────┘  │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────────────────────┐
│      Holdout Manager                 │
│      (applies split strategy)        │
└──────┬───────────────────────────────┘
       │
       ├─────────────────────┬────────────────────┐
       ▼                     ▼                    ▼
┌─────────────┐      ┌─────────────┐     ┌─────────────┐
│  Training   │      │  Temporal   │     │  Patient    │
│  Patients   │      │  Holdout    │     │  Holdout    │
│             │      │  (20% end)  │     │  (fixed)    │
│  ┌────┐     │      │             │     │             │
│  │ p01│ 80% │      │  ┌────┐     │     │  ┌────┐     │
│  ├────┤     │      │  │ p01│ 20% │     │  │ p04│ 100%│
│  │ p02│ 80% │      │  ├────┤     │     │  ├────┤     │
│  ├────┤     │      │  │ p02│ 20% │     │  │ p12│ 100%│
│  │... │     │      │  ├────┤     │     │  ├────┤     │
│  └────┘     │      │  │... │     │     │  │ p18│ 100%│
│             │      │  └────┘     │     │  └────┘     │
└─────────────┘      └─────────────┘     └─────────────┘
       │                     │                    │
       └─────────────────────┴────────────────────┘
                            │
                            ▼
                ┌────────────────────┐
                │   Validation       │
                │   ✓ No overlap     │
                │   ✓ Temporal order │
                │   ✓ Min samples    │
                └────────────────────┘
```

## Hybrid Split Strategy

```
Dataset: 15 patients, 1000 samples each

Step 1: Patient Split (20% holdout)
┌─────────────────────────────────────────────────────┐
│ All Patients (15)                                   │
│ ┌────────────────────┬────────────────────────────┐ │
│ │ Training Patients  │  Holdout Patients          │ │
│ │ (12 patients)      │  (3 patients)              │ │
│ │ p01, p02, p03, ... │  p04, p12, p18             │ │
│ └────────────────────┴────────────────────────────┘ │
└─────────────────────────────────────────────────────┘

Step 2: Temporal Split on Training Patients (20% holdout)
┌─────────────────────────────────────────────────────┐
│ Training Patients (12 patients × 1000 samples)      │
│                                                     │
│ Each patient timeline:                              │
│ [═══════════════════════|═════]                     │
│  Training (800 samples) | Temp Holdout (200)        │
│                                                     │
│ Result: 9,600 train samples, 2,400 temp holdout     │
└─────────────────────────────────────────────────────┘

Step 3: Final Split
┌─────────────────────────────────────────────────────┐
│ TRAINING DATA                                       │
│ • 12 patients                                       │
│ • 800 samples each (early 80% of timeline)          │
│ • Total: 9,600 samples                              │
│ • Used for: Model training                          │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│ HOLDOUT DATA                                        │
│ • Temporal: 12 patients × 200 samples = 2,400       │
│   (late 20% of training patients' timelines)        │
│ • Patient: 3 patients × 1000 samples = 3,000        │
│   (complete holdout patients)                       │
│ • Total: 5,400 samples                              │
│ • Used for: Final evaluation                        │
└─────────────────────────────────────────────────────┘

Key Properties:
✓ No patient overlap: Training patients ≠ Holdout patients
✓ Temporal separation: Within training patients, train < holdout time
✓ Fixed splits: Same patients held out every experiment
```

## API Usage Flow

```
┌─────────────────────────────────────────────────────────┐
│                 Training Workflow                       │
└─────────────────────────────────────────────────────────┘

1. Import
   from src.data.versioning.dataset_registry import load_training_data

2. Load Training Data
   train_data = load_training_data("aleppo_2017")
                      │
                      ├─▶ Loads config from configs/data/holdout_10pct/
                      ├─▶ Loads raw dataset
                      ├─▶ Applies holdout strategy
                      ├─▶ Returns ONLY training portion
                      └─▶ Validates no leakage

3. Train Model
   model.fit(train_data)

4. Save Model
   model.save("trained_models/...")

┌─────────────────────────────────────────────────────────┐
│                Evaluation Workflow                      │
└─────────────────────────────────────────────────────────┘

5. Load Holdout Data (FINAL EVALUATION ONLY)
   holdout_data = load_holdout_data("aleppo_2017")
                      │
                      ├─▶ Loads same config
                      ├─▶ Loads raw dataset
                      ├─▶ Applies same holdout strategy
                      ├─▶ Returns ONLY holdout portion
                      └─▶ Validates consistency

6. Evaluate
   results = model.evaluate(holdout_data)

7. Log Results
   experiment.log_results(results)
```

## Configuration Hierarchy

```
┌────────────────────────────────────────────────────────────┐
│                   HoldoutConfig                            │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ dataset_name: "aleppo_2017"                          │  │
│  │ holdout_type: hybrid                                 │  │
│  │ version: "1.0"                                       │  │
│  │ created_date: "2026-01-09"                           │  │
│  └──────────────────────────────────────────────────────┘  │
│           │                          │                     │
│           ▼                          ▼                     │
│  ┌──────────────────┐    ┌──────────────────────┐          │
│  │TemporalConfig    │    │ PatientConfig        │          │
│  ├──────────────────┤    ├──────────────────────┤          │
│  │• holdout_%: 0.2  │    │• holdout_patients:   │          │
│  │• min_train: 50   │    │  [p04, p12, p18]     │          │
│  │• min_holdout: 10 │    │• min_train_pts: 2    │          │
│  └──────────────────┘    │• random_seed: 42     │          │
│                          └──────────────────────┘          │
└────────────────────────────────────────────────────────────┘
```

## File System Layout

```
nocturnal/
│
├── configs/data/
│   ├── ARCHITECTURE.md            📖 This document
│   └── holdout_10pct/             📁 Per-dataset holdout configs
│       ├── aleppo_2017.yaml       📄
│       ├── brown_2019.yaml        📄
│       ├── lynch_2022.yaml        📄
│       └── tamborlane_2008.yaml   📄
│
├── src/data/
│   ├── preprocessing/
│   │   ├── holdout_config.py      🔧 Configuration classes
│   │   └── holdout_manager.py     ⚙️ Split implementation
│   └── versioning/
│       └── dataset_registry.py    🗃️ Registry + API
│
└── scripts/
    └── data_processing_scripts/
        └── generate_holdout_configs.py  🏗️ Generate configs
```

## Validation Pipeline

```
┌────────────────────────────────────────────────────────┐
│        Validation Pipeline                             │
└────────────────────────────────────────────────────────┘

For each dataset:
  │
  ├─▶ [1] Check Config Exists
  │      └─▶ configs/data/holdout_10pct/{dataset}.yaml
  │
  ├─▶ [2] Load Configuration
  │      └─▶ Parse YAML, validate structure
  │
  ├─▶ [3] Load and Split Data
  │      ├─▶ Load raw dataset
  │      ├─▶ Apply holdout strategy
  │      └─▶ Generate train/holdout splits
  │
  ├─▶ [4] Validate Patient Overlap
  │      ├─▶ Get train patients
  │      ├─▶ Get holdout patients
  │      └─▶ Check: train ∩ holdout = ∅
  │
  ├─▶ [5] Validate Temporal Order
  │      ├─▶ For each training patient:
  │      │   └─▶ max(train_time) < min(holdout_time)
  │      └─▶ Check temporal consistency
  │
  ├─▶ [6] Validate Sample Counts
  │      ├─▶ Train samples >= min_train_samples
  │      └─▶ Holdout samples >= min_holdout_samples
  │
  └─▶ [7] Generate Report
         ├─▶ ✓ All checks passed
         └─▶ ✗ Issues found (if any)
```

## Legend

```
📁 Directory
📄 YAML Configuration File
📖 Documentation
🔧 Core Module (configuration)
⚙️ Core Module (implementation)
🗃️ Core Module (registry/API)
🏗️ Utility Script (generation)
```

