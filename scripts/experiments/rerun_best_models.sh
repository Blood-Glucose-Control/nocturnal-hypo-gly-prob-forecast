#!/usr/bin/env bash
# rerun_best_models.sh
# Re-runs all 71 best (model × dataset) evaluations from best_by_model_dataset.csv.
# Restores original output directory structure via --output-dir.
#
# Usage:
#   bash scripts/experiments/rerun_best_models.sh           # normal run
#   bash scripts/experiments/rerun_best_models.sh --dry-run # print commands only
#
# Output:
#   scripts/experiments/rerun_manifest.txt  — successful run_paths (one per line)
#   scripts/experiments/rerun_failed.txt    — failed runs (model|dataset|checkpoint|run_path|exit_code=N)
#   logs/rerun_<index>.log                  — per-job stdout+stderr

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"

# Default venv for models without a dedicated .venvs/<model> dir
DEFAULT_PYTHON="$REPO_DIR/.noctprob-venv/bin/python"
VENVS_DIR="$REPO_DIR/.venvs"
MANIFEST="$REPO_DIR/scripts/experiments/rerun_manifest.txt"
FAILED="$REPO_DIR/scripts/experiments/rerun_failed.txt"
LOG_DIR="$REPO_DIR/logs"
DRY_RUN=0

if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
    echo "[DRY-RUN] Commands will be printed but not executed."
fi

mkdir -p "$LOG_DIR"
# Clear manifest/failed files at start (fresh run)
> "$MANIFEST"
> "$FAILED"

# ---------------------------------------------------------------------------
# Job list: "model_arg|model_csv|dataset|cov_bucket|checkpoint|run_path"
#   model_arg   = value passed to --model (no slash suffix)
#   model_csv   = original model name from CSV (may include slash suffix)
#   cov_bucket  = iob_cob | iob | bg_only | zero_shot
#   checkpoint  = path or "" for zero-shot
# ---------------------------------------------------------------------------
JOBS=(
  # chronos2
  "chronos2|chronos2|aleppo_2017|iob_cob|/data/home/cjrisi/nocturnal/trained_models/artifacts/chronos2/2026-04-25_0843_RID20260425_084356_257848_08_bg_iob_cob_high_lr/model.pt|experiments/nocturnal_forecasting/512ctx_96fh/chronos2/2026-04-25_183908_aleppo_2017_finetuned"
  "chronos2|chronos2|brown_2019|iob_ia|trained_models/artifacts/chronos2/2026-04-26_06:26_RID20260426_062650_516757_holdout_workflow/snapshots/step_20000/model.pt|experiments/nocturnal_forecasting/512ctx_96fh/chronos2/2026-04-30_042011_brown_2019_finetuned"
  "chronos2|chronos2|lynch_2022|iob_ia|trained_models/artifacts/chronos2/2026-04-26_06:26_RID20260426_062650_516757_holdout_workflow/snapshots/step_20000/model.pt|experiments/nocturnal_forecasting/512ctx_96fh/chronos2/2026-04-30_041451_lynch_2022_finetuned"
  "chronos2|chronos2|tamborlane_2008|bg_only|/data/home/cjrisi/nocturnal/trained_models/artifacts/chronos2/2026-04-24_2348_RID20260424_234848_257848_00_bg_only/model.pt|experiments/nocturnal_forecasting/512ctx_96fh/chronos2/2026-04-25_183555_tamborlane_2008_finetuned"
  # deepar
  "deepar|deepar|aleppo_2017|bg_only|/data/home/cjrisi/nocturnal/trained_models/artifacts/deepar/2026-04-28_0725_RID20260428_072516_909736_01_short_ctx|experiments/nocturnal_forecasting/512ctx_96fh/deepar/2026-04-28_213745_aleppo_2017_finetuned"
  "deepar|deepar|brown_2019|bg_only|/data/home/cjrisi/nocturnal/trained_models/artifacts/deepar/2026-04-28_0725_RID20260428_072516_909736_01_short_ctx|experiments/nocturnal_forecasting/512ctx_96fh/deepar/2026-04-28_213704_brown_2019_finetuned"
  "deepar|deepar|lynch_2022|bg_only|/data/home/cjrisi/nocturnal/trained_models/artifacts/deepar/2026-04-28_0725_RID20260428_072516_909752_10_low_lr|experiments/nocturnal_forecasting/512ctx_96fh/deepar/2026-04-28_220011_lynch_2022_finetuned"
  "deepar|deepar|tamborlane_2008|bg_only|/data/home/cjrisi/nocturnal/trained_models/artifacts/deepar/2026-04-28_0725_RID20260428_072516_909752_10_low_lr|experiments/nocturnal_forecasting/512ctx_96fh/deepar/2026-04-28_234036_tamborlane_2008_finetuned"
  # moirai
  "moirai|moirai|aleppo_2017|iob_ia|/data/home/cjrisi/nocturnal/trained_models/artifacts/moirai/2026-04-17_05:44_RID20260417_054446_3342356_holdout_workflow/model.pt|experiments/nocturnal_forecasting/512ctx_96fh/moirai/2026-04-19_2045_aleppo_2017_finetuned"
  "moirai|moirai|brown_2019|iob_ia|/data/home/cjrisi/nocturnal/trained_models/artifacts/moirai/2026-04-17_05:44_RID20260417_054446_3342356_holdout_workflow/model.pt|experiments/nocturnal_forecasting/512ctx_96fh/moirai/2026-04-19_0653_brown_2019_finetuned"
  "moirai|moirai|lynch_2022|iob_ia|/data/home/cjrisi/nocturnal/trained_models/artifacts/moirai/2026-04-17_06:10_RID20260417_061030_3342356_holdout_workflow/model.pt|experiments/nocturnal_forecasting/512ctx_96fh/moirai/2026-04-23_0620_lynch_2022_finetuned"
  "moirai|moirai|tamborlane_2008|bg_only|/data/home/cjrisi/nocturnal/trained_models/artifacts/moirai/2026-04-17_04:52_RID20260417_045208_3342356_holdout_workflow/model.pt|experiments/nocturnal_forecasting/512ctx_96fh/moirai/2026-04-23_0541_tamborlane_2008_finetuned"
  # moment (no --probabilistic)
  "moment|moment|aleppo_2017|bg_only|/data/home/cjrisi/nocturnal/trained_models/artifacts/moment/2026-04-22_02:05_RID20260422_020541_gpu1_21364_holdout_workflow/model.pt|experiments/nocturnal_forecasting/512ctx_96fh/moment/2026-04-23_2122_aleppo_2017_finetuned"
  "moment|moment|brown_2019|bg_only|/data/home/cjrisi/nocturnal/trained_models/artifacts/moment/2026-04-22_02:05_RID20260422_020541_gpu1_21364_holdout_workflow/model.pt|experiments/nocturnal_forecasting/512ctx_96fh/moment/2026-04-23_2124_brown_2019_finetuned"
  "moment|moment|lynch_2022|bg_only|/data/home/cjrisi/nocturnal/trained_models/artifacts/moment/2026-04-22_02:13_RID20260422_021339_gpu1_24042_holdout_workflow/model.pt|experiments/nocturnal_forecasting/512ctx_96fh/moment/2026-04-23_2128_lynch_2022_finetuned"
  "moment|moment|tamborlane_2008|bg_only|/data/home/cjrisi/nocturnal/trained_models/artifacts/moment/2026-04-22_02:13_RID20260422_021339_gpu1_24042_holdout_workflow/model.pt|experiments/nocturnal_forecasting/512ctx_96fh/moment/2026-04-23_2132_tamborlane_2008_finetuned"
  # naive_baseline/Average
  "naive_baseline|naive_baseline/Average|aleppo_2017|bg_only|/data/home/cjrisi/nocturnal/trained_models/artifacts/naive_baseline/2026-04-28_01:46_RID20260428_014636_836040_holdout_workflow/model.pt|experiments/nocturnal_forecasting/512ctx_96fh/naive_baseline/2026-04-28_020700_aleppo_2017_finetuned"
  "naive_baseline|naive_baseline/Average|brown_2019|bg_only|/data/home/cjrisi/nocturnal/trained_models/artifacts/naive_baseline/2026-04-28_01:46_RID20260428_014636_836040_holdout_workflow/model.pt|experiments/nocturnal_forecasting/512ctx_96fh/naive_baseline/2026-04-28_020617_brown_2019_finetuned"
  "naive_baseline|naive_baseline/Average|lynch_2022|bg_only|/data/home/cjrisi/nocturnal/trained_models/artifacts/naive_baseline/2026-04-28_01:46_RID20260428_014636_836040_holdout_workflow/model.pt|experiments/nocturnal_forecasting/512ctx_96fh/naive_baseline/2026-04-28_020627_lynch_2022_finetuned"
  "naive_baseline|naive_baseline/Average|tamborlane_2008|bg_only|/data/home/cjrisi/nocturnal/trained_models/artifacts/naive_baseline/2026-04-28_01:46_RID20260428_014636_836040_holdout_workflow/model.pt|experiments/nocturnal_forecasting/512ctx_96fh/naive_baseline/2026-04-28_021600_tamborlane_2008_finetuned"
  # naive_baseline/Naive
  "naive_baseline|naive_baseline/Naive|aleppo_2017|bg_only|/data/home/cjrisi/nocturnal/trained_models/artifacts/naive_baseline/2026-04-28_01:37_RID20260428_013743_836040_holdout_workflow/model.pt|experiments/nocturnal_forecasting/512ctx_96fh/naive_baseline/2026-04-28_020415_aleppo_2017_finetuned"
  "naive_baseline|naive_baseline/Naive|brown_2019|bg_only|/data/home/cjrisi/nocturnal/trained_models/artifacts/naive_baseline/2026-04-28_01:37_RID20260428_013743_836040_holdout_workflow/model.pt|experiments/nocturnal_forecasting/512ctx_96fh/naive_baseline/2026-04-28_020415_brown_2019_finetuned"
  "naive_baseline|naive_baseline/Naive|lynch_2022|bg_only|/data/home/cjrisi/nocturnal/trained_models/artifacts/naive_baseline/2026-04-28_01:37_RID20260428_013743_836040_holdout_workflow/model.pt|experiments/nocturnal_forecasting/512ctx_96fh/naive_baseline/2026-04-28_020415_lynch_2022_finetuned"
  "naive_baseline|naive_baseline/Naive|tamborlane_2008|bg_only|/data/home/cjrisi/nocturnal/trained_models/artifacts/naive_baseline/2026-04-28_01:37_RID20260428_013743_836040_holdout_workflow/model.pt|experiments/nocturnal_forecasting/512ctx_96fh/naive_baseline/2026-04-28_020415_tamborlane_2008_finetuned"
  # patchtst
  "patchtst|patchtst|aleppo_2017|bg_only|/data/home/cjrisi/nocturnal/trained_models/artifacts/patchtst/2026-04-28_1710_RID20260428_171004_963783_06_wide_d|experiments/nocturnal_forecasting/512ctx_96fh/patchtst/2026-04-29_002450_aleppo_2017_finetuned"
  "patchtst|patchtst|brown_2019|bg_only|/data/home/cjrisi/nocturnal/trained_models/artifacts/patchtst/2026-04-28_0803_RID20260428_080348_916053_09_high_lr|experiments/nocturnal_forecasting/512ctx_96fh/patchtst/2026-04-29_002656_brown_2019_finetuned"
  "patchtst|patchtst|lynch_2022|bg_only|/data/home/cjrisi/nocturnal/trained_models/artifacts/patchtst/2026-04-28_0803_RID20260428_080348_916053_09_high_lr|experiments/nocturnal_forecasting/512ctx_96fh/patchtst/2026-04-29_002822_lynch_2022_finetuned"
  "patchtst|patchtst|tamborlane_2008|bg_only|/data/home/cjrisi/nocturnal/trained_models/artifacts/patchtst/2026-04-28_0803_RID20260428_080348_916053_09_high_lr|experiments/nocturnal_forecasting/512ctx_96fh/patchtst/2026-04-29_015154_tamborlane_2008_finetuned"
  # statistical/AutoARIMA (bg_only)
  "statistical|statistical/AutoARIMA|aleppo_2017|bg_only|/data/home/cjrisi/nocturnal/trained_models/artifacts/statistical/2026-04-28_0308_RID20260428_030824_856335_00_autoarima_bg_only|experiments/nocturnal_forecasting/512ctx_96fh/statistical/2026-04-28_034725_aleppo_2017_finetuned"
  "statistical|statistical/AutoARIMA|brown_2019|bg_only|/data/home/cjrisi/nocturnal/trained_models/artifacts/statistical/2026-04-28_0308_RID20260428_030824_856335_00_autoarima_bg_only|experiments/nocturnal_forecasting/512ctx_96fh/statistical/2026-04-28_035017_brown_2019_finetuned"
  "statistical|statistical/AutoARIMA|lynch_2022|bg_only|/data/home/cjrisi/nocturnal/trained_models/artifacts/statistical/2026-04-28_0308_RID20260428_030824_856335_00_autoarima_bg_only|experiments/nocturnal_forecasting/512ctx_96fh/statistical/2026-04-28_034725_lynch_2022_finetuned"
  "statistical|statistical/AutoARIMA|tamborlane_2008|bg_only|/data/home/cjrisi/nocturnal/trained_models/artifacts/statistical/2026-04-28_0308_RID20260428_030824_856335_00_autoarima_bg_only|experiments/nocturnal_forecasting/512ctx_96fh/statistical/2026-04-28_035053_tamborlane_2008_finetuned"
  # statistical/AutoARIMA+IOB (iob) — no tamborlane entry in CSV
  "statistical|statistical/AutoARIMA+IOB|aleppo_2017|bg_only|trained_models/artifacts/statistical/2026-04-28_0318_RID20260428_031849_856335_01_autoarima_bg_iob|experiments/nocturnal_forecasting/512ctx_96fh/statistical/2026-04-28_044121_aleppo_2017_finetuned"
  "statistical|statistical/AutoARIMA+IOB|brown_2019|bg_only|trained_models/artifacts/statistical/2026-04-28_0318_RID20260428_031849_856335_01_autoarima_bg_iob|experiments/nocturnal_forecasting/512ctx_96fh/statistical/2026-04-28_044421_brown_2019_finetuned"
  "statistical|statistical/AutoARIMA+IOB|lynch_2022|bg_only|trained_models/artifacts/statistical/2026-04-28_0318_RID20260428_031849_856335_01_autoarima_bg_iob|experiments/nocturnal_forecasting/512ctx_96fh/statistical/2026-04-28_044634_lynch_2022_finetuned"
  # statistical/NPTS (bg_only)
  "statistical|statistical/NPTS|aleppo_2017|bg_only|/data/home/cjrisi/nocturnal/trained_models/artifacts/statistical/2026-04-28_0333_RID20260428_033357_856335_03_npts_bg_only|experiments/nocturnal_forecasting/512ctx_96fh/statistical/2026-04-28_040922_aleppo_2017_finetuned"
  "statistical|statistical/NPTS|brown_2019|bg_only|/data/home/cjrisi/nocturnal/trained_models/artifacts/statistical/2026-04-28_0333_RID20260428_033357_856335_03_npts_bg_only|experiments/nocturnal_forecasting/512ctx_96fh/statistical/2026-04-28_041143_brown_2019_finetuned"
  "statistical|statistical/NPTS|lynch_2022|bg_only|/data/home/cjrisi/nocturnal/trained_models/artifacts/statistical/2026-04-28_0333_RID20260428_033357_856335_03_npts_bg_only|experiments/nocturnal_forecasting/512ctx_96fh/statistical/2026-04-28_040903_lynch_2022_finetuned"
  "statistical|statistical/NPTS|tamborlane_2008|bg_only|/data/home/cjrisi/nocturnal/trained_models/artifacts/statistical/2026-04-28_0333_RID20260428_033357_856335_03_npts_bg_only|experiments/nocturnal_forecasting/512ctx_96fh/statistical/2026-04-28_041241_tamborlane_2008_finetuned"
  # statistical/Theta (bg_only)
  "statistical|statistical/Theta|aleppo_2017|bg_only|/data/home/cjrisi/nocturnal/trained_models/artifacts/statistical/2026-04-28_0323_RID20260428_032348_856335_02_theta_bg_only|experiments/nocturnal_forecasting/512ctx_96fh/statistical/2026-04-28_035450_aleppo_2017_finetuned"
  "statistical|statistical/Theta|brown_2019|bg_only|/data/home/cjrisi/nocturnal/trained_models/artifacts/statistical/2026-04-28_0323_RID20260428_032348_856335_02_theta_bg_only|experiments/nocturnal_forecasting/512ctx_96fh/statistical/2026-04-28_040701_brown_2019_finetuned"
  "statistical|statistical/Theta|lynch_2022|bg_only|/data/home/cjrisi/nocturnal/trained_models/artifacts/statistical/2026-04-28_0323_RID20260428_032348_856335_02_theta_bg_only|experiments/nocturnal_forecasting/512ctx_96fh/statistical/2026-04-28_040440_lynch_2022_finetuned"
  "statistical|statistical/Theta|tamborlane_2008|bg_only|/data/home/cjrisi/nocturnal/trained_models/artifacts/statistical/2026-04-28_0323_RID20260428_032348_856335_02_theta_bg_only|experiments/nocturnal_forecasting/512ctx_96fh/statistical/2026-04-28_035739_tamborlane_2008_finetuned"
  # sundial (zero-shot)
  "sundial|sundial|aleppo_2017|zero_shot||experiments/nocturnal_forecasting/512ctx_96fh/sundial/2026-04-16_2002_aleppo_2017_zeroshot"
  "sundial|sundial|brown_2019|zero_shot||experiments/nocturnal_forecasting/512ctx_96fh/sundial/2026-04-16_2007_brown_2019_zeroshot"
  "sundial|sundial|lynch_2022|zero_shot||experiments/nocturnal_forecasting/512ctx_96fh/sundial/2026-04-30_040657_lynch_2022_zeroshot"
  "sundial|sundial|tamborlane_2008|zero_shot||experiments/nocturnal_forecasting/512ctx_96fh/sundial/2026-04-16_2012_tamborlane_2008_zeroshot"
  # tft
  "tft|tft|aleppo_2017|iob_cob|/data/home/cjrisi/nocturnal/trained_models/artifacts/tft/2026-04-29_1853_RID20260429_185322_1168187_16_iob_cob_high_lr|experiments/nocturnal_forecasting/512ctx_96fh/tft/2026-04-29_194226_aleppo_2017_finetuned"
  "tft|tft|brown_2019|iob|/data/home/cjrisi/nocturnal/trained_models/artifacts/tft/2026-04-28_1829_RID20260428_182909_968382_11_iob_high_lr|experiments/nocturnal_forecasting/512ctx_96fh/tft/2026-04-29_025312_brown_2019_finetuned"
  "tft|tft|lynch_2022|iob|/data/home/cjrisi/nocturnal/trained_models/artifacts/tft/2026-04-28_1829_RID20260428_182909_968382_11_iob_high_lr|experiments/nocturnal_forecasting/512ctx_96fh/tft/2026-04-29_034217_lynch_2022_finetuned"
  "tft|tft|tamborlane_2008|bg_only|/data/home/cjrisi/nocturnal/trained_models/artifacts/tft/2026-04-28_1749_RID20260428_174925_968378_01_bg_wide|experiments/nocturnal_forecasting/512ctx_96fh/tft/2026-04-29_023810_tamborlane_2008_finetuned"
  # tide
  "tide|tide|aleppo_2017|iob_ia|/data/home/cjrisi/nocturnal/trained_models/artifacts/tide/2026-04-17_19:05_RID20260417_190554_3405856_holdout_workflow/model.pt|experiments/nocturnal_forecasting/512ctx_96fh/tide/2026-04-17_2240_aleppo_2017_finetuned"
  "tide|tide|brown_2019|bg_only|/data/home/cjrisi/nocturnal/trained_models/artifacts/tide/2026-04-17_16:52_RID20260417_165227_3405856_holdout_workflow/model.pt|experiments/nocturnal_forecasting/512ctx_96fh/tide/2026-04-17_2210_brown_2019_finetuned"
  "tide|tide|lynch_2022|iob_ia|/data/home/cjrisi/nocturnal/trained_models/artifacts/tide/2026-04-17_19:50_RID20260417_195034_3405856_holdout_workflow/model.pt|experiments/nocturnal_forecasting/512ctx_96fh/tide/2026-04-23_2016_lynch_2022_finetuned"
  "tide|tide|tamborlane_2008|bg_only|/data/home/cjrisi/nocturnal/trained_models/artifacts/tide/2026-04-17_16:52_RID20260417_165227_3405856_holdout_workflow/model.pt|experiments/nocturnal_forecasting/512ctx_96fh/tide/2026-04-17_2212_tamborlane_2008_finetuned"
  # timegrad
  "timegrad|timegrad|aleppo_2017|bg_only|trained_models/artifacts/timegrad/2026-02-24_01:12_RID20260224_011201_2800320_holdout_workflow/model.pt|experiments/nocturnal_forecasting/512ctx_96fh/timegrad/2026-04-16_2041_aleppo_2017_finetuned"
  "timegrad|timegrad|brown_2019|bg_only|trained_models/artifacts/timegrad/2026-02-24_01:12_RID20260224_011201_2800320_holdout_workflow/model.pt|experiments/nocturnal_forecasting/512ctx_96fh/timegrad/2026-04-30_163606_brown_2019_finetuned"
  "timegrad|timegrad|lynch_2022|bg_only|trained_models/artifacts/timegrad/2026-02-24_01:12_RID20260224_011201_2800320_holdout_workflow/model.pt|experiments/nocturnal_forecasting/512ctx_96fh/timegrad/2026-04-30_040550_lynch_2022_finetuned"
  "timegrad|timegrad|tamborlane_2008|bg_only|trained_models/artifacts/timegrad/2026-02-24_01:12_RID20260224_011201_2800320_holdout_workflow/model.pt|experiments/nocturnal_forecasting/512ctx_96fh/timegrad/2026-04-30_163613_tamborlane_2008_finetuned"
  # timesfm
  "timesfm|timesfm|aleppo_2017|bg_only|/data/home/cjrisi/nocturnal/trained_models/artifacts/timesfm/2026-04-27_16:57_RID20260427_165709_gpu1_30394_holdout_workflow/model.pt|experiments/nocturnal_forecasting/512ctx_96fh/timesfm/2026-04-28_044903_aleppo_2017_finetuned"
  "timesfm|timesfm|brown_2019|bg_only|/data/home/cjrisi/nocturnal/trained_models/artifacts/timesfm/2026-04-22_10:11_RID20260422_101119_gpu1_625_holdout_workflow/model.pt|experiments/nocturnal_forecasting/512ctx_96fh/timesfm/2026-04-23_0004_brown_2019_finetuned"
  "timesfm|timesfm|lynch_2022|bg_only|/data/home/cjrisi/nocturnal/trained_models/artifacts/timesfm/2026-04-27_21:22_RID20260427_212232_gpu1_3052_holdout_workflow/model.pt|experiments/nocturnal_forecasting/512ctx_96fh/timesfm/2026-04-28_042641_lynch_2022_finetuned"
  "timesfm|timesfm|tamborlane_2008|bg_only|trained_models/artifacts/timesfm/2026-02-27_05:37_RID20260227_053718_211403_holdout_workflow/resumed_training/model.pt|experiments/nocturnal_forecasting/512ctx_96fh/timesfm/2026-02-27_1858_tamborlane_2008_finetuned"
  # toto
  "toto|toto|aleppo_2017|zero_shot||experiments/nocturnal_forecasting/512ctx_96fh/toto/2026-04-29_214226_aleppo_2017_zeroshot"
  "toto|toto|brown_2019|bg_only|/data/home/cjrisi/nocturnal/trained_models/artifacts/toto/2026-04-17_21:12_RID20260417_211202_3443331_holdout_workflow/model.pt|experiments/nocturnal_forecasting/512ctx_96fh/toto/2026-04-23_0418_brown_2019_finetuned"
  "toto|toto|lynch_2022|bg_only|/data/home/cjrisi/nocturnal/trained_models/artifacts/toto/2026-04-17_21:12_RID20260417_211202_3443331_holdout_workflow/model.pt|experiments/nocturnal_forecasting/512ctx_96fh/toto/2026-04-23_0414_lynch_2022_finetuned"
  "toto|toto|tamborlane_2008|bg_only|/data/home/cjrisi/nocturnal/trained_models/artifacts/toto/2026-04-17_21:12_RID20260417_211202_3443331_holdout_workflow/model.pt|experiments/nocturnal_forecasting/512ctx_96fh/toto/2026-04-23_0420_tamborlane_2008_finetuned"
  # ttm (no --probabilistic)
  "ttm|ttm|aleppo_2017|bg_only|trained_models/artifacts/ttm/2026-02-27_03:53_RID20260227_035316_193673_holdout_workflow/model.pt|experiments/nocturnal_forecasting/512ctx_96fh/ttm/2026-02-27_0532_aleppo_2017_finetuned"
  "ttm|ttm|brown_2019|iob|trained_models/artifacts/ttm/2026-04-30_01:40_RID20260430_014019_1231906_holdout_workflow|experiments/nocturnal_forecasting/512ctx_96fh/ttm/2026-04-30_025210_brown_2019_finetuned"
  "ttm|ttm|lynch_2022|iob|trained_models/artifacts/ttm/2026-04-30_01:40_RID20260430_014019_1231906_holdout_workflow|experiments/nocturnal_forecasting/512ctx_96fh/ttm/2026-04-30_025411_lynch_2022_finetuned"
  "ttm|ttm|tamborlane_2008|bg_only|trained_models/artifacts/ttm/2026-02-27_03:53_RID20260227_035316_193673_holdout_workflow/model.pt|experiments/nocturnal_forecasting/512ctx_96fh/ttm/2026-02-27_1626_tamborlane_2008_finetuned"
)

TOTAL=${#JOBS[@]}
echo "Total jobs: $TOTAL"

# ---------------------------------------------------------------------------
# Helper: resolve python binary for a model
# Models backed by AutoGluon but without their own .venvs/<model> dir share
# the chronos2 venv (which has autogluon.timeseries installed).
# All other models with a .venvs/<model> dir use that.
# Fallback: .noctprob-venv.
# ---------------------------------------------------------------------------
AUTOGLUON_MODELS=("deepar" "patchtst" "tft" "naive_baseline" "statistical")

get_python() {
    local model="$1"
    # Check if this model is autogluon-backed but has no own venv
    local venv_python="$VENVS_DIR/$model/bin/python"
    if [[ -x "$venv_python" ]]; then
        echo "$venv_python"
        return
    fi
    for ag_model in "${AUTOGLUON_MODELS[@]}"; do
        if [[ "$model" == "$ag_model" ]]; then
            echo "$VENVS_DIR/chronos2/bin/python"
            return
        fi
    done
    echo "$DEFAULT_PYTHON"
}

# ---------------------------------------------------------------------------
# Helper: build the python command for a job entry
# ---------------------------------------------------------------------------
build_cmd() {
    local entry="$1"
    local gpu="$2"
    IFS='|' read -r model_arg model_csv dataset cov_bucket checkpoint run_path <<< "$entry"
    local python_bin
    python_bin="$(get_python "$model_arg")"

    # Resolve relative checkpoint paths
    local ckpt_abs=""
    if [[ -n "$checkpoint" ]]; then
        if [[ "$checkpoint" = /* ]]; then
            ckpt_abs="$checkpoint"
        else
            ckpt_abs="$REPO_DIR/$checkpoint"
        fi
    fi

    # Covariate args
    local cov_args=""
    case "$cov_bucket" in
        iob_cob) cov_args="--covariate-cols iob cob" ;;
        iob_ia)  cov_args="--covariate-cols iob insulin_availability" ;;
        iob)     cov_args="--covariate-cols iob" ;;
        *)       cov_args="" ;;
    esac

    # Probabilistic flag (moment and ttm do not support it)
    local prob_flag=""
    if [[ "$model_arg" != "moment" && "$model_arg" != "ttm" ]]; then
        prob_flag="--probabilistic"
    fi

    # Checkpoint arg
    local ckpt_arg=""
    if [[ -n "$ckpt_abs" ]]; then
        ckpt_arg="--checkpoint $ckpt_abs"
    fi

    echo "$python_bin scripts/experiments/nocturnal_hypo_eval.py \
--model $model_arg \
--dataset $dataset \
--context-length 512 \
--forecast-length 96 \
--cuda-device $gpu \
--output-dir $run_path \
$ckpt_arg \
$cov_args \
$prob_flag"
}

# ---------------------------------------------------------------------------
# Preflight: verify checkpoint paths exist
# ---------------------------------------------------------------------------
echo ""
echo "=== Checkpoint preflight check ==="
preflight_ok=1
for entry in "${JOBS[@]}"; do
    IFS='|' read -r model_arg model_csv dataset cov_bucket checkpoint run_path <<< "$entry"
    if [[ -z "$checkpoint" ]]; then
        continue  # zero-shot, no checkpoint needed
    fi
    if [[ "$checkpoint" = /* ]]; then
        ckpt_abs="$checkpoint"
    else
        ckpt_abs="$REPO_DIR/$checkpoint"
    fi
    if [[ ! -e "$ckpt_abs" ]]; then
        echo "  MISSING: $ckpt_abs  (for $model_csv / $dataset)"
        preflight_ok=0
    fi
done

if [[ $preflight_ok -eq 1 ]]; then
    echo "  All checkpoints found."
else
    echo ""
    echo "ERROR: One or more checkpoints are missing. Fix paths before running."
    exit 1
fi

if [[ $DRY_RUN -eq 1 ]]; then
    echo ""
    echo "=== Dry-run: commands ==="
    idx=0
    for entry in "${JOBS[@]}"; do
        gpu=$(( (idx % 6) / 3 ))
        echo "--- Job $((idx+1))/$TOTAL (GPU $gpu) ---"
        build_cmd "$entry" "$gpu"
        echo ""
        (( idx++ )) || true
    done
    echo "[DRY-RUN] Done. No jobs were executed."
    exit 0
fi

# ---------------------------------------------------------------------------
# Batch execution: 6 jobs per batch (indices 0-2 → GPU 0, 3-5 → GPU 1)
# ---------------------------------------------------------------------------
echo ""
echo "=== Starting batch execution (6 jobs per batch) ==="

SUCCESS_COUNT=0
FAIL_COUNT=0

run_job() {
    local idx="$1"
    local entry="$2"
    local gpu="$3"
    local logfile="$LOG_DIR/rerun_${idx}.log"
    local cmd
    cmd="$(build_cmd "$entry" "$gpu")"
    echo "[Job $((idx+1))/$TOTAL] GPU=$gpu → $logfile"
    eval "$cmd" > "$logfile" 2>&1
}

batch_start=0
while [[ $batch_start -lt $TOTAL ]]; do
    batch_end=$(( batch_start + 5 ))
    if [[ $batch_end -ge $TOTAL ]]; then
        batch_end=$(( TOTAL - 1 ))
    fi

    echo ""
    echo "--- Batch jobs $((batch_start+1))–$((batch_end+1)) ---"

    declare -A batch_pids
    declare -A batch_entries

    for (( i=batch_start; i<=batch_end; i++ )); do
        entry="${JOBS[$i]}"
        local_idx=$(( i - batch_start ))       # 0-5
        gpu=$(( local_idx / 3 ))               # 0-2→GPU0, 3-5→GPU1
        run_job "$i" "$entry" "$gpu" &
        batch_pids[$i]=$!
        batch_entries[$i]="$entry"
    done

    # Wait for all jobs in this batch and collect exit codes
    for (( i=batch_start; i<=batch_end; i++ )); do
        pid=${batch_pids[$i]}
        entry="${batch_entries[$i]}"
        IFS='|' read -r model_arg model_csv dataset cov_bucket checkpoint run_path <<< "$entry"
        if wait "$pid"; then
            echo "  [OK]   Job $((i+1)): $model_csv / $dataset"
            echo "$run_path" >> "$MANIFEST"
            (( SUCCESS_COUNT++ )) || true
        else
            ec=$?
            echo "  [FAIL] Job $((i+1)): $model_csv / $dataset  (exit=$ec)"
            echo "${model_csv}|${dataset}|${checkpoint}|${run_path}|exit_code=${ec}" >> "$FAILED"
            (( FAIL_COUNT++ )) || true
        fi
    done

    unset batch_pids
    unset batch_entries
    batch_start=$(( batch_end + 1 ))
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "=== Rerun complete ==="
echo "  Succeeded: $SUCCESS_COUNT / $TOTAL"
echo "  Failed:    $FAIL_COUNT / $TOTAL"
echo ""
if [[ $FAIL_COUNT -gt 0 ]]; then
    echo "Failed jobs logged to: $FAILED"
    echo "Re-run failed jobs only by inspecting that file."
    exit 1
else
    echo "All jobs succeeded. Manifest: $MANIFEST"
    exit 0
fi
