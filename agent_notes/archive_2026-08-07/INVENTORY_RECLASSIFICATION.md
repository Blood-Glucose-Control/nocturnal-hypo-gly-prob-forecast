# Data Inventory Reclassification
**Date**: 2026-08-06 19:16 UTC
**Status**: ✅ COMPLETE

## What We Did

User identified that the initial inventory had too many "unknown" model types and requested reclassification without re-scanning all 290 GB.

### Changes Made

1. **Updated model type extraction logic**
   - Added 8 new model types to detection
   - Verified script matches actual directory structure in `trained_models/artifacts/`

2. **Created reclassification script**
   - File: `scripts/update_inventory_classifications.py`
   - Processes existing CSV (42,779 rows) in memory
   - Updates model_type based on path patterns
   - Marks _bad_runs_archive files as deletable

3. **Executed reclassification**
   - Processed all 42,779 files
   - Updated 40,636 model type classifications
   - Marked 1,292 files in _bad_runs_archive as deletable

## Results: Before vs After

### Model Type Coverage
| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Unknown model types | 16,359 files | 13 files | **99.9% improvement** |
| Identified architectures | 6 types | 14 types | +8 new types |
| Coverage | ~62% | 99.99% | Near-complete |

### New Model Types Detected
1. **Chronos2** (was "Chronos") - 3,331 files, 29.62 GB
2. **Toto** - 492 files, 29.53 GB (NEW!)
3. **PatchTST** - 654 files, 15.74 GB (NEW!)
4. **TTM** - 12,718 files, 13.90 GB (NEW!)
5. **Moment** - 685 files, 12.77 GB (NEW!)
6. **Temporal Fusion Transformer** (was "Transformer") - 811 files, 11.87 GB
7. **Statistical Baseline** - 247 files, 3.00 GB (NEW!)
8. **Naive Baseline** - 113 files, 1.59 GB (NEW!)
9. **MOIRAI** - 560 files, 1.49 GB (NEW!)
10. **TimeGrad** - 180 files, 0.11 GB (NEW!)
11. **Sundial** - 24 files, 0.09 GB (NEW!)

### Retention Tier Changes
| Tier | Before | After | Change |
|------|--------|-------|--------|
| Paper-critical | 228 files, 0.47 GB | 228 files, 0.47 GB | Unchanged |
| Archivable | 206 files, 12.84 GB | 49 files, 4.05 GB | Refined |
| Deletable | 703 files, 0.20 GB | **1,460 files, 23.76 GB** | **+23.56 GB** 🎯 |
| Unknown | 20,821 files, 276.62 GB | 20,221 files, 261.85 GB | Improved |

## Key Findings

### 🎯 Quick Win: _bad_runs_archive
**1,292 files (23.75 GB) now marked deletable**

These are failed/abandoned training runs in 5 directories:
- `trained_models/artifacts/deepar/_bad_runs_archive/`
- `trained_models/artifacts/tft/_bad_runs_archive/`
- `trained_models/artifacts/timesfm/_bad_runs_archive/`
- `trained_models/artifacts/statistical/_bad_runs_archive/`
- `trained_models/artifacts/chronos2/_bad_runs_archive/`

**Action**: Safe to delete immediately after backup verification

### 🔥 Storage Hotspots Confirmed
1. **TimesFM**: 147 GB (50.6%) - No change, still #1 target
2. **Toto**: 30 GB (10.2%) - Newly identified
3. **PatchTST**: 16 GB (5.4%) - Newly identified
4. **TTM**: 14 GB (4.8%) - Newly identified, but 12,718 small files

### Dataset Classification Skipped
Per user request:
- Existing dataset labels (gluroo/ohim/general) are incorrect
- True datasets need to be extracted from `split_metadata.json` in each run
- Not critical for retention decisions
- Deferred to future refinement

## Files Updated

1. **trained_models_inventory.csv** (5.8 MB)
   - Backup created: `trained_models_inventory.csv.backup` (6.0 MB)
   - All 42,779 rows updated
   - Model types reclassified
   - _bad_runs_archive marked deletable

2. **agent_notes/DATA_INVENTORY_SUMMARY.md**
   - Updated with new model type breakdown
   - Added _bad_runs_archive findings
   - Revised storage analysis
   - Updated recommendations

3. **scripts/update_inventory_classifications.py** (NEW)
   - Reusable Python script for future reclassifications
   - Can be extended with new model types
   - Fast in-memory processing

## Validation

### Model Type Accuracy
All 14 model types match actual directory names in `trained_models/artifacts/`:
```bash
$ ls trained_models/artifacts/
chronos2/  deepar/  moirai/  moment/  naive_baseline/  patchtst/
statistical/  sundial/  tft/  tide/  timegrad/  timesfm/  toto/  ttm/
```
✅ Perfect match!

### _bad_runs_archive Detection
```bash
$ find trained_models -type d -name "*_bad_runs_archive" | wc -l
5
```
✅ All 5 directories found and processed!

### File Count Verification
```bash
$ wc -l trained_models_inventory.csv
42780 trained_models_inventory.csv  # 42,779 data + 1 header
```
✅ No rows lost!

## Impact on Next Tasks

### data-mirror-plan (P0-4) - NEXT
**Ready to proceed** with clear priorities:
1. Paper-critical: 0.47 GB (must mirror)
2. Archivable: 4.05 GB (should mirror)
3. _bad_runs_archive: 23.75 GB (can delete now)
4. TimesFM hotspot: 147 GB (needs review + selective mirror)

### Quick Wins Available
1. **Delete _bad_runs_archive** → Immediate 23.75 GB recovery
2. **Mirror paper-critical + archivable** → Only 4.5 GB to protect
3. **Review TimesFM for deduplication** → Potential 50-100 GB savings

### Remaining Work
- **Unknown retention tier** (262 GB, 20,221 files): Still needs manual review
  - We now know *what* they are (model types)
  - Just don't know *when* to delete them (retention policy)
  - Requires path-based date extraction or metadata parsing

---

## Commands Used

### Reclassification
```bash
python3 scripts/update_inventory_classifications.py
mv trained_models_inventory.csv trained_models_inventory.csv.backup
mv trained_models_inventory_updated.csv trained_models_inventory.csv
```

### Analysis
```python
import csv
# Parse CSV with proper model type + retention tier aggregation
# Generate summary statistics
```

### Verification
```bash
# Count model directories
ls trained_models/artifacts/ | wc -l  # 14

# Find bad runs archives
find trained_models -type d -name "*_bad_runs_archive" | wc -l  # 5

# Verify row count
wc -l trained_models_inventory.csv  # 42,780
```

---

**Reclassification Status**: COMPLETE ✅
**Next Task**: data-mirror-plan (P0-4)
**Quick Win Available**: Delete 23.75 GB _bad_runs_archive
