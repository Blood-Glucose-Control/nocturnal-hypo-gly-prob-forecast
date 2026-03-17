# Archive old misconfigured eval outputs (recommended)
mkdir -p experiments/nocturnal_forecasting/_bad_runs_archive
mv experiments/nocturnal_forecasting/512ctx_96fh/chronos2/2026-03-15_2015_aleppo_2017_finetuned experiments/nocturnal_forecasting/_bad_runs_archive/
mv experiments/nocturnal_forecasting/512ctx_96fh/chronos2/2026-03-15_2019_aleppo_2017_finetuned experiments/nocturnal_forecasting/_bad_runs_archive/
mv experiments/nocturnal_forecasting/512ctx_96fh/chronos2/2026-03-15_2022_aleppo_2017_finetuned experiments/nocturnal_forecasting/_bad_runs_archive/
mv experiments/nocturnal_forecasting/288ctx_96fh/chronos2/2026-03-15_2025_aleppo_2017_finetuned experiments/nocturnal_forecasting/_bad_runs_archive/
mv experiments/nocturnal_forecasting/288ctx_96fh/chronos2/2026-03-15_2029_aleppo_2017_finetuned experiments/nocturnal_forecasting/_bad_runs_archive/


# 02 — IOB only (ctx=512, lr=1e-5)
python scripts/experiments/nocturnal_hypo_eval.py \
    --model chronos2 \
    --dataset aleppo_2017 \
    --config-dir configs/data/holdout_10pct \
    --model-config configs/models/chronos2/02_bg_iob.yaml \
    --context-length 512 \
    --forecast-length 96 \
    --covariate-cols iob \
    --cuda-device 0 \
    --checkpoint trained_models/artifacts/chronos2/2026-03-12_22:15_RID20260312_221524_1208721_holdout_workflow/model.pt \
    --output-dir experiments/nocturnal_forecasting/512ctx_96fh/chronos2/2026-03-15_2301_recheck_02_modelpt_aleppo_2017_finetuned

python scripts/experiments/nocturnal_hypo_eval.py \
    --model chronos2 \
    --dataset aleppo_2017 \
    --config-dir configs/data/holdout_10pct \
    --model-config configs/models/chronos2/02_bg_iob.yaml \
    --context-length 512 \
    --forecast-length 96 \
    --covariate-cols iob \
    --cuda-device 0 \
    --checkpoint trained_models/artifacts/chronos2/2026-03-12_22:15_RID20260312_221524_1208721_holdout_workflow/resumed_training/model.pt \
    --output-dir experiments/nocturnal_forecasting/512ctx_96fh/chronos2/2026-03-15_2302_recheck_02_resumed_aleppo_2017_finetuned


# 03 — IOB + insulin_availability (ctx=512, lr=5e-5)
python scripts/experiments/nocturnal_hypo_eval.py \
    --model chronos2 \
    --dataset brown_2019 \
    --config-dir configs/data/holdout_10pct \
    --model-config configs/models/chronos2/03_bg_iob_ia_high_lr.yaml \
    --context-length 512 \
    --forecast-length 96 \
    --covariate-cols iob insulin_availability \
    --cuda-device 0 \
    --checkpoint trained_models/artifacts/chronos2/2026-03-12_23:07_RID20260312_230758_1217900_holdout_workflow/model.pt \
    --output-dir experiments/nocturnal_forecasting/512ctx_96fh/chronos2/2026-03-15_2303_recheck_03_modelpt_brown_2019_finetuned

python scripts/experiments/nocturnal_hypo_eval.py \
    --model chronos2 \
    --dataset aleppo_2017 \
    --config-dir configs/data/holdout_10pct \
    --model-config configs/models/chronos2/03_bg_iob_ia_high_lr.yaml \
    --context-length 512 \
    --forecast-length 96 \
    --covariate-cols iob insulin_availability \
    --cuda-device 0 \
    --checkpoint trained_models/artifacts/chronos2/2026-03-12_23:07_RID20260312_230758_1217900_holdout_workflow/resumed_training/model.pt \
    --output-dir experiments/nocturnal_forecasting/512ctx_96fh/chronos2/2026-03-15_2304_recheck_03_resumed_aleppo_2017_finetuned


# 04 — IOB + insulin_availability (ctx=288, lr=1e-5)
python scripts/experiments/nocturnal_hypo_eval.py \
    --model chronos2 \
    --dataset aleppo_2017 \
    --config-dir configs/data/holdout_10pct \
    --model-config configs/models/chronos2/04_bg_iob_short_ctx.yaml \
    --context-length 288 \
    --forecast-length 96 \
    --covariate-cols iob insulin_availability \
    --cuda-device 0 \
    --checkpoint trained_models/artifacts/chronos2/2026-03-12_23:09_RID20260312_230932_1218491_holdout_workflow/model.pt \
    --output-dir experiments/nocturnal_forecasting/288ctx_96fh/chronos2/2026-03-15_2305_recheck_04_modelpt_aleppo_2017_finetuned

python scripts/experiments/nocturnal_hypo_eval.py \
    --model chronos2 \
    --dataset aleppo_2017 \
    --config-dir configs/data/holdout_10pct \
    --model-config configs/models/chronos2/04_bg_iob_short_ctx.yaml \
    --context-length 288 \
    --forecast-length 96 \
    --covariate-cols iob insulin_availability \
    --cuda-device 0 \
    --checkpoint trained_models/artifacts/chronos2/2026-03-12_23:09_RID20260312_230932_1218491_holdout_workflow/resumed_training/model.pt \
    --output-dir experiments/nocturnal_forecasting/288ctx_96fh/chronos2/2026-03-15_2306_recheck_04_resumed_aleppo_2017_finetuned


# ============================================================
# 03 step sweep (5k / 10k / 25k) — training + nocturnal eval
# ============================================================

# 1) Create temporary 03 configs with different fine_tune_steps
python - <<'PY'
from pathlib import Path
import yaml

base_path = Path("configs/models/chronos2/03_bg_iob_ia_high_lr.yaml")
base = yaml.safe_load(base_path.read_text())

for steps in [5000, 10000, 25000]:
    cfg = dict(base)
    cfg["fine_tune_steps"] = steps
    out = Path(f"configs/models/chronos2/_tmp_03_steps_{steps}.yaml")
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"wrote {out}")
PY


# 2) Train each sweep point (single dataset for quick compare)
CUDA_VISIBLE_DEVICES=0 MODEL_TYPE="chronos2" MODEL_CONFIG="configs/models/chronos2/_tmp_03_steps_5000.yaml" CONFIG_DIR="configs/data/holdout_10pct" DATASETS="aleppo_2017 brown_2019 lynch_2022" SKIP_TRAINING="false" OUTPUT_BASE_DIR="trained_models/artifacts/chronos2/sweep_03_steps5000_holdout_workflow" ./scripts/examples/run_holdout_generic_workflow.sh

CUDA_VISIBLE_DEVICES=0 MODEL_TYPE="chronos2" MODEL_CONFIG="configs/models/chronos2/_tmp_03_steps_10000.yaml" CONFIG_DIR="configs/data/holdout_10pct" DATASETS="aleppo_2017 brown_2019 lynch_2022" SKIP_TRAINING="false" OUTPUT_BASE_DIR="trained_models/artifacts/chronos2/sweep_03_steps10000_holdout_workflow" ./scripts/examples/run_holdout_generic_workflow.sh

CUDA_VISIBLE_DEVICES=0 MODEL_TYPE="chronos2" MODEL_CONFIG="configs/models/chronos2/_tmp_03_steps_25000.yaml" CONFIG_DIR="configs/data/holdout_10pct" DATASETS="aleppo_2017 brown_2019 lynch_2022" SKIP_TRAINING="false" OUTPUT_BASE_DIR="trained_models/artifacts/chronos2/sweep_03_steps25000_holdout_workflow" ./scripts/examples/run_holdout_generic_workflow.sh


# 3) Evaluate model.pt and resumed_training/model.pt for each sweep point

# 5k
python scripts/experiments/nocturnal_hypo_eval.py \
    --model chronos2 \
    --dataset aleppo_2017 \
    --config-dir configs/data/holdout_10pct \
    --model-config configs/models/chronos2/_tmp_03_steps_5000.yaml \
    --context-length 512 \
    --forecast-length 96 \
    --covariate-cols iob insulin_availability \
    --cuda-device 0 \
    --checkpoint trained_models/artifacts/chronos2/sweep_03_steps5000_holdout_workflow/model.pt \
    --output-dir experiments/nocturnal_forecasting/512ctx_96fh/chronos2/2026-03-15_2311_sweep03_5k_modelpt_aleppo_2017_finetuned

python scripts/experiments/nocturnal_hypo_eval.py \
    --model chronos2 \
    --dataset aleppo_2017 \
    --config-dir configs/data/holdout_10pct \
    --model-config configs/models/chronos2/_tmp_03_steps_5000.yaml \
    --context-length 512 \
    --forecast-length 96 \
    --covariate-cols iob insulin_availability \
    --cuda-device 0 \
    --checkpoint trained_models/artifacts/chronos2/sweep_03_steps5000_holdout_workflow/resumed_training/model.pt \
    --output-dir experiments/nocturnal_forecasting/512ctx_96fh/chronos2/2026-03-15_2312_sweep03_5k_resumed_aleppo_2017_finetuned

# 10k
python scripts/experiments/nocturnal_hypo_eval.py \
    --model chronos2 \
    --dataset aleppo_2017 \
    --config-dir configs/data/holdout_10pct \
    --model-config configs/models/chronos2/_tmp_03_steps_10000.yaml \
    --context-length 512 \
    --forecast-length 96 \
    --covariate-cols iob insulin_availability \
    --cuda-device 0 \
    --checkpoint trained_models/artifacts/chronos2/sweep_03_steps10000_holdout_workflow/model.pt \
    --output-dir experiments/nocturnal_forecasting/512ctx_96fh/chronos2/2026-03-15_2313_sweep03_10k_modelpt_aleppo_2017_finetuned

python scripts/experiments/nocturnal_hypo_eval.py \
    --model chronos2 \
    --dataset aleppo_2017 \
    --config-dir configs/data/holdout_10pct \
    --model-config configs/models/chronos2/_tmp_03_steps_10000.yaml \
    --context-length 512 \
    --forecast-length 96 \
    --covariate-cols iob insulin_availability \
    --cuda-device 0 \
    --checkpoint trained_models/artifacts/chronos2/sweep_03_steps10000_holdout_workflow/resumed_training/model.pt \
    --output-dir experiments/nocturnal_forecasting/512ctx_96fh/chronos2/2026-03-15_2314_sweep03_10k_resumed_aleppo_2017_finetuned

# 25k
python scripts/experiments/nocturnal_hypo_eval.py \
    --model chronos2 \
    --dataset aleppo_2017 \
    --config-dir configs/data/holdout_10pct \
    --model-config configs/models/chronos2/_tmp_03_steps_25000.yaml \
    --context-length 512 \
    --forecast-length 96 \
    --covariate-cols iob insulin_availability \
    --cuda-device 0 \
    --checkpoint trained_models/artifacts/chronos2/sweep_03_steps25000_holdout_workflow/model.pt \
    --output-dir experiments/nocturnal_forecasting/512ctx_96fh/chronos2/2026-03-15_2315_sweep03_25k_modelpt_aleppo_2017_finetuned

python scripts/experiments/nocturnal_hypo_eval.py \
    --model chronos2 \
    --dataset aleppo_2017 \
    --config-dir configs/data/holdout_10pct \
    --model-config configs/models/chronos2/_tmp_03_steps_25000.yaml \
    --context-length 512 \
    --forecast-length 96 \
    --covariate-cols iob insulin_availability \
    --cuda-device 0 \
    --checkpoint trained_models/artifacts/chronos2/sweep_03_steps25000_holdout_workflow/resumed_training/model.pt \
    --output-dir experiments/nocturnal_forecasting/512ctx_96fh/chronos2/2026-03-15_2316_sweep03_25k_resumed_aleppo_2017_finetuned


# ============================================================
# 03 step sweep extended (15k / 20k / 30k) — no step 7
# NOTE: resumed training is broken (restarts from base weights,
#       not the fine-tuned checkpoint). Skip step 7 for all runs.
# ============================================================

# 1) Create temp configs for 15k / 20k / 30k
python - <<'PY'
from pathlib import Path
import yaml

base_path = Path("configs/models/chronos2/03_bg_iob_ia_high_lr.yaml")
base = yaml.safe_load(base_path.read_text())

for steps in [15000, 20000, 30000]:
    cfg = dict(base)
    cfg["fine_tune_steps"] = steps
    out = Path(f"configs/models/chronos2/_tmp_03_steps_{steps}.yaml")
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"wrote {out}")
PY


# 2) Train — call Python directly so we can pass --skip-steps 7

python scripts/examples/example_holdout_generic_workflow.py \
    --model-type chronos2 \
    --datasets aleppo_2017 brown_2019 lynch_2022 \
    --config-dir configs/data/holdout_10pct \
    --model-config configs/models/chronos2/_tmp_03_steps_15000.yaml \
    --output-dir trained_models/artifacts/chronos2/sweep_03_steps15000_holdout_workflow \
    --skip-steps 7

python scripts/examples/example_holdout_generic_workflow.py \
    --model-type chronos2 \
    --datasets aleppo_2017 brown_2019 lynch_2022 \
    --config-dir configs/data/holdout_10pct \
    --model-config configs/models/chronos2/_tmp_03_steps_20000.yaml \
    --output-dir trained_models/artifacts/chronos2/sweep_03_steps20000_holdout_workflow \
    --skip-steps 7

python scripts/examples/example_holdout_generic_workflow.py \
    --model-type chronos2 \
    --datasets aleppo_2017 brown_2019 lynch_2022 \
    --config-dir configs/data/holdout_10pct \
    --model-config configs/models/chronos2/_tmp_03_steps_30000.yaml \
    --output-dir trained_models/artifacts/chronos2/sweep_03_steps30000_holdout_workflow \
    --skip-steps 7


# 3) Eval — model.pt only (no resumed since step 7 was skipped)

# 15k
python scripts/experiments/nocturnal_hypo_eval.py \
    --model chronos2 \
    --dataset brown_2019 \
    --config-dir configs/data/holdout_10pct \
    --model-config configs/models/chronos2/_tmp_03_steps_15000.yaml \
    --context-length 512 \
    --forecast-length 96 \
    --covariate-cols iob insulin_availability \
    --cuda-device 0 \
    --checkpoint trained_models/artifacts/chronos2/sweep_03_steps15000_holdout_workflow/model.pt \
    --output-dir experiments/nocturnal_forecasting/512ctx_96fh/chronos2/2026-03-16_1000_sweep03_15k_modelpt_brown_2019_finetuned

# 20k
python scripts/experiments/nocturnal_hypo_eval.py \
    --model chronos2 \
    --dataset brown_2019 \
    --config-dir configs/data/holdout_10pct \
    --model-config configs/models/chronos2/_tmp_03_steps_20000.yaml \
    --context-length 512 \
    --forecast-length 96 \
    --covariate-cols iob insulin_availability \
    --cuda-device 0 \
    --checkpoint trained_models/artifacts/chronos2/sweep_03_steps20000_holdout_workflow/model.pt \
    --output-dir experiments/nocturnal_forecasting/512ctx_96fh/chronos2/2026-03-16_1001_sweep03_20k_modelpt_brown_2019_finetuned

# 30k
python scripts/experiments/nocturnal_hypo_eval.py \
    --model chronos2 \
    --dataset brown_2019 \
    --config-dir configs/data/holdout_10pct \
    --model-config configs/models/chronos2/_tmp_03_steps_30000.yaml \
    --context-length 512 \
    --forecast-length 96 \
    --covariate-cols iob insulin_availability \
    --cuda-device 0 \
    --checkpoint trained_models/artifacts/chronos2/sweep_03_steps30000_holdout_workflow/model.pt \
    --output-dir experiments/nocturnal_forecasting/512ctx_96fh/chronos2/2026-03-16_1002_sweep03_30k_modelpt_brown_2019_finetuned