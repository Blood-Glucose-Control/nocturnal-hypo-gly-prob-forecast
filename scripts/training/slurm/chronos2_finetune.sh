#!/bin/bash
#
# CHRONOS-2 FINE-TUNING via Model Class
# ======================================
# Fine-tunes Chronos-2 on Brown 2019 using Chronos2Forecaster + DatasetRegistry.
# This mirrors the TTM workflow: registry → config → model.fit() → model.save().
#
# Usage:
#   sbatch scripts/training/slurm/chronos2_finetune.sh
#   sbatch --export=STEPS=5000 scripts/training/slurm/chronos2_finetune.sh
#
#SBATCH --job-name=chronos2_ft
#SBATCH --output=logs/chronos2_ft_%j.out
#SBATCH --error=logs/chronos2_ft_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --partition=HI

# Configuration (can be overridden with --export)
: ${STEPS:=15000}
: ${LR:=1e-5}
: ${TIME_LIMIT:=}
: ${OUTPUT_DIR:=models/chronos2_brown}
: ${DATASET:=brown_2019}
: ${CONFIG_DIR:=configs/data/holdout_5pct}

# Environment setup
PROJECT_ROOT="/u201/s6jindal/nocturnal-hypo-gly-prob-forecast"
cd "$PROJECT_ROOT" || { echo "Failed to cd to $PROJECT_ROOT"; exit 1; }

eval "$(/opt/anaconda3/bin/conda shell.bash hook)"
conda activate chronos

echo "========================================="
echo "Chronos-2 Fine-Tuning (Model Class)"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "Steps: $STEPS | LR: $LR | Dataset: $DATASET"
echo "Output: $OUTPUT_DIR"
echo "========================================="

mkdir -p logs

# Build command
CMD="python scripts/examples/example_chronos2_finetune.py \
    --steps $STEPS \
    --lr $LR \
    --output-dir $OUTPUT_DIR \
    --dataset $DATASET \
    --config-dir $CONFIG_DIR"

if [ -n "$TIME_LIMIT" ]; then
    CMD="$CMD --time-limit $TIME_LIMIT"
fi

echo "Command: $CMD"
eval $CMD 2>&1 | tee "logs/chronos2_ft_${SLURM_JOB_ID}.log"

EXIT_CODE=$?
echo ""
echo "Completed: $(date) | Exit: $EXIT_CODE | Duration: ${SECONDS}s"
exit $EXIT_CODE
