#!/usr/bin/env bash
# Auto-generated rerun runner (GPU 0). Review manifest.csv first.
set -uo pipefail
cd "/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast"

echo "[GPU0] ze_moment_aleppo_2017 (expect RMSE~3.771)"
.venvs/moment/bin/python scripts/experiments/nocturnal_hypo_eval.py --model moment --dataset aleppo_2017 --config-dir configs/data/holdout_10pct --context-length 512 --forecast-length 96 --cuda-device 0 --output-dir experiments/nocturnal_forecasting/512ctx_96fh/moment/rebuttal_aleppo_2017_zeroshot --covariate-cols > scripts/analysis/rebuttal_neurips2026/outputs/reruns/logs/ze_moment_aleppo_2017.log 2>&1
echo "  -> exit $? : ze_moment_aleppo_2017"

echo "[GPU0] ze_moment_lynch_2022 (expect RMSE~3.483)"
.venvs/moment/bin/python scripts/experiments/nocturnal_hypo_eval.py --model moment --dataset lynch_2022 --config-dir configs/data/holdout_10pct --context-length 512 --forecast-length 96 --cuda-device 0 --output-dir experiments/nocturnal_forecasting/512ctx_96fh/moment/rebuttal_lynch_2022_zeroshot --covariate-cols > scripts/analysis/rebuttal_neurips2026/outputs/reruns/logs/ze_moment_lynch_2022.log 2>&1
echo "  -> exit $? : ze_moment_lynch_2022"

echo "[GPU0] ze_ttm_aleppo_2017 (expect RMSE~2.921)"
.venvs/ttm/bin/python scripts/experiments/nocturnal_hypo_eval.py --model ttm --dataset aleppo_2017 --config-dir configs/data/holdout_10pct --context-length 512 --forecast-length 96 --cuda-device 0 --output-dir experiments/nocturnal_forecasting/512ctx_96fh/ttm/rebuttal_aleppo_2017_zeroshot --covariate-cols > scripts/analysis/rebuttal_neurips2026/outputs/reruns/logs/ze_ttm_aleppo_2017.log 2>&1
echo "  -> exit $? : ze_ttm_aleppo_2017"

echo "[GPU0] ze_ttm_lynch_2022 (expect RMSE~3.001)"
.venvs/ttm/bin/python scripts/experiments/nocturnal_hypo_eval.py --model ttm --dataset lynch_2022 --config-dir configs/data/holdout_10pct --context-length 512 --forecast-length 96 --cuda-device 0 --output-dir experiments/nocturnal_forecasting/512ctx_96fh/ttm/rebuttal_lynch_2022_zeroshot --covariate-cols > scripts/analysis/rebuttal_neurips2026/outputs/reruns/logs/ze_ttm_lynch_2022.log 2>&1
echo "  -> exit $? : ze_ttm_lynch_2022"

echo "[GPU0] a4_ttm_brown_2019 (expect RMSE~2.566)"
.venvs/ttm/bin/python scripts/experiments/nocturnal_hypo_eval.py --model ttm --dataset brown_2019 --config-dir configs/data/holdout_10pct --context-length 512 --forecast-length 96 --cuda-device 0 --output-dir experiments/nocturnal_forecasting/512ctx_96fh/ttm/rebuttal_brown_2019_finetuned --checkpoint /data/home/cjrisi/nocturnal/trained_models/artifacts/ttm/2026-02-27_03:53_RID20260227_035316_193673_holdout_workflow/model.pt --covariate-cols > scripts/analysis/rebuttal_neurips2026/outputs/reruns/logs/a4_ttm_brown_2019.log 2>&1
echo "  -> exit $? : a4_ttm_brown_2019"

echo "GPU0 DONE"
