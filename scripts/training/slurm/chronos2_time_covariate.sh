#!/bin/bash
#SBATCH --job-name=chronos2_cov_cd
#SBATCH --output=logs/chronos2_cov_cd_%j.out
#SBATCH --error=logs/chronos2_cov_cd_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --partition=ALL

# ============================================================================
# Chronos-2 Covariate A/B/C/D Experiment
#
# Arm A: BG-only fine-tune (no covariates)
# Arm B: BG + time-of-day features (hour_sin, hour_cos as known future)
# Arm C: BG + IOB (past-only context, NOT known future)
# Arm D: BG + IOB (past-only) + time-of-day (known future)
#
# Override defaults:
#   sbatch --export=STEPS=10000 scripts/training/slurm/chronos2_time_covariate.sh
#   sbatch --export=ARM=CD scripts/training/slurm/chronos2_time_covariate.sh
# ============================================================================

# Configurable parameters (override via --export)
STEPS=${STEPS:-5000}
LR=${LR:-1e-5}
TIME_LIMIT=${TIME_LIMIT:-7200}
MAX_EVAL_EPISODES=${MAX_EVAL_EPISODES:-500}
ARM=${ARM:-CD}

# Project setup
PROJECT_ROOT=/u201/s6jindal/nocturnal-hypo-gly-prob-forecast
cd "$PROJECT_ROOT" || exit 1

# Create logs directory
mkdir -p logs

# Environment
echo "Setting up conda environment..."
eval "$(/opt/anaconda3/bin/conda shell.bash hook)"
conda activate chronos

echo "============================================"
echo "Chronos-2 Covariate A/B/C/D Experiment"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo ""
echo "Parameters:"
echo "  STEPS=$STEPS"
echo "  LR=$LR"
echo "  TIME_LIMIT=$TIME_LIMIT"
echo "  MAX_EVAL_EPISODES=$MAX_EVAL_EPISODES"
echo "  ARM=$ARM"
echo ""

# Verify GPU
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Verify time features and IOB exist in cache
echo "Verifying features in data..."
python -c "
from src.data.diabetes_datasets.data_loader import get_loader
loader = get_loader('brown_2019', use_cached=True)
sample = list(loader.processed_data.values())[0]
assert 'hour_sin' in sample.columns, 'hour_sin missing! Regenerate cache.'
assert 'hour_cos' in sample.columns, 'hour_cos missing! Regenerate cache.'
assert 'iob' in sample.columns, 'iob missing from data!'
print('All features verified in cache (hour_sin, hour_cos, iob).')
"
if [ $? -ne 0 ]; then
    echo "ERROR: Features not in cache. Regenerating..."
    rm -rf cache/data/brown_2019/processed/
    python -c "
from src.data.diabetes_datasets.data_loader import get_loader
loader = get_loader('brown_2019', use_cached=True)
print(f'Regenerated cache: {len(loader.processed_data)} patients')
"
fi

# Run experiment
echo ""
echo "Starting experiment..."
python scripts/chronos2_time_covariate_experiment.py \
    --steps "$STEPS" \
    --lr "$LR" \
    --time-limit "$TIME_LIMIT" \
    --max-eval-episodes "$MAX_EVAL_EPISODES" \
    --arm "$ARM"

EXIT_CODE=$?

echo ""
echo "============================================"
echo "Experiment finished with exit code: $EXIT_CODE"
echo "Date: $(date)"
echo "============================================"

exit $EXIT_CODE
