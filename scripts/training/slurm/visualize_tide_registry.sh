#!/bin/bash
#SBATCH --partition=HI
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --job-name=tide_viz
#SBATCH --output=logs/tide_viz_%j.out
#SBATCH --error=logs/tide_viz_%j.err

# ============================================================
# TiDE Registry Visualization
# ============================================================
# Generates best/worst 30 plots for patient-level and temporal holdout
#
# USAGE:
#   sbatch scripts/training/slurm/visualize_tide_registry.sh
#   MODEL_PATH=models/tide_registry/best_1406592 sbatch scripts/training/slurm/visualize_tide_registry.sh

MODEL_PATH="${MODEL_PATH:-models/tide_registry/best_1406592}"

echo "=========================================="
echo "TiDE Registry Visualization"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Model: $MODEL_PATH"
echo ""

# Conda setup
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate chronos
echo "Conda env: $(conda info --envs | grep '*')"
echo "Python: $(which python)"

cd /u201/s6jindal/nocturnal-hypo-gly-prob-forecast

python scripts/visualize_tide_registry.py \
    --model-path "$MODEL_PATH" \
    --holdout-dir configs/data/holdout_5pct \
    --max-episodes 500

echo ""
echo "Visualization complete at $(date)"
