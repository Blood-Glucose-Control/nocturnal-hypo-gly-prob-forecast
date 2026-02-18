#!/bin/bash
#
# CHRONOS-2 EVALUATION via Model Class
# =====================================
# Loads a saved Chronos2Forecaster and evaluates on holdout data.
# Run this after chronos2_finetune.sh to verify parity with the
# validated experiment (1.890 RMSE).
#
# Usage:
#   sbatch --export=MODEL_DIR=models/chronos2_brown/20260218_123456 scripts/training/slurm/chronos2_evaluate.sh
#
#SBATCH --job-name=chronos2_eval
#SBATCH --output=logs/chronos2_eval_%j.out
#SBATCH --error=logs/chronos2_eval_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --partition=HI

# Configuration
: ${MODEL_DIR:?MODEL_DIR must be set (e.g. models/chronos2_brown/20260218_123456)}
: ${DATASET:=brown_2019}
: ${CONFIG_DIR:=configs/data/holdout_5pct}

# Environment setup
PROJECT_ROOT="/u201/s6jindal/nocturnal-hypo-gly-prob-forecast"
cd "$PROJECT_ROOT" || { echo "Failed to cd to $PROJECT_ROOT"; exit 1; }

eval "$(/opt/anaconda3/bin/conda shell.bash hook)"
conda activate chronos

echo "========================================="
echo "Chronos-2 Evaluation (Model Class)"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Model: $MODEL_DIR"
echo "Dataset: $DATASET"
echo "========================================="

python scripts/examples/example_chronos2_evaluate.py \
    --model-dir "$MODEL_DIR" \
    --dataset "$DATASET" \
    --config-dir "$CONFIG_DIR" \
    2>&1 | tee "logs/chronos2_eval_${SLURM_JOB_ID}.log"

EXIT_CODE=$?
echo "Completed: $(date) | Exit: $EXIT_CODE | Duration: ${SECONDS}s"
exit $EXIT_CODE
