#!/usr/bin/env bash
# Auto-generated rerun runner (GPU 1). Review manifest.csv first.
set -uo pipefail
cd "/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast"

echo "[GPU1] ze_moment_brown_2019 (expect RMSE~3.556)"
.venvs/moment/bin/python scripts/experiments/nocturnal_hypo_eval.py --model moment --dataset brown_2019 --config-dir configs/data/holdout_10pct --context-length 512 --forecast-length 96 --cuda-device 1 --output-dir experiments/nocturnal_forecasting/512ctx_96fh/moment/rebuttal_brown_2019_zeroshot --covariate-cols > scripts/analysis/rebuttal_neurips2026/outputs/reruns/logs/ze_moment_brown_2019.log 2>&1
echo "  -> exit $? : ze_moment_brown_2019"

echo "[GPU1] ze_moment_tamborlane_2008 (expect RMSE~3.955)"
.venvs/moment/bin/python scripts/experiments/nocturnal_hypo_eval.py --model moment --dataset tamborlane_2008 --config-dir configs/data/holdout_10pct --context-length 512 --forecast-length 96 --cuda-device 1 --output-dir experiments/nocturnal_forecasting/512ctx_96fh/moment/rebuttal_tamborlane_2008_zeroshot --covariate-cols > scripts/analysis/rebuttal_neurips2026/outputs/reruns/logs/ze_moment_tamborlane_2008.log 2>&1
echo "  -> exit $? : ze_moment_tamborlane_2008"

echo "[GPU1] ze_ttm_brown_2019 (expect RMSE~2.843)"
.venvs/ttm/bin/python scripts/experiments/nocturnal_hypo_eval.py --model ttm --dataset brown_2019 --config-dir configs/data/holdout_10pct --context-length 512 --forecast-length 96 --cuda-device 1 --output-dir experiments/nocturnal_forecasting/512ctx_96fh/ttm/rebuttal_brown_2019_zeroshot --covariate-cols > scripts/analysis/rebuttal_neurips2026/outputs/reruns/logs/ze_ttm_brown_2019.log 2>&1
echo "  -> exit $? : ze_ttm_brown_2019"

echo "[GPU1] ze_ttm_tamborlane_2008 (expect RMSE~3.424)"
.venvs/ttm/bin/python scripts/experiments/nocturnal_hypo_eval.py --model ttm --dataset tamborlane_2008 --config-dir configs/data/holdout_10pct --context-length 512 --forecast-length 96 --cuda-device 1 --output-dir experiments/nocturnal_forecasting/512ctx_96fh/ttm/rebuttal_tamborlane_2008_zeroshot --covariate-cols > scripts/analysis/rebuttal_neurips2026/outputs/reruns/logs/ze_ttm_tamborlane_2008.log 2>&1
echo "  -> exit $? : ze_ttm_tamborlane_2008"

echo "[GPU1] a4_ttm_brown_2019 (expect RMSE~2.566)"
.venvs/ttm/bin/python scripts/experiments/nocturnal_hypo_eval.py --model ttm --dataset brown_2019 --config-dir configs/data/holdout_10pct --context-length 512 --forecast-length 96 --cuda-device 1 --output-dir experiments/nocturnal_forecasting/512ctx_96fh/ttm/rebuttal_brown_2019_finetuned --checkpoint /data/home/cjrisi/nocturnal/trained_models/artifacts/ttm/2026-02-27_03:53_RID20260227_035316_193673_holdout_workflow/model.pt --covariate-cols > scripts/analysis/rebuttal_neurips2026/outputs/reruns/logs/a4_ttm_brown_2019.log 2>&1
echo "  -> exit $? : a4_ttm_brown_2019"

echo "GPU1 DONE"
