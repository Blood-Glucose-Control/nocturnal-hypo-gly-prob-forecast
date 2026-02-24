#!/bin/bash
#SBATCH --job-name=tide_reg
#SBATCH --partition=HI
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/tide_reg_%j.out
#SBATCH --error=logs/tide_reg_%j.err

# TiDE with Data Registry + Patient-Level Holdout
# Uses holdout_5pct config: 8 fully held-out patients + 5% temporal
#
# USAGE:
#   sbatch scripts/training/slurm/tide_registry.sh              # best config
#   sbatch --export=CONFIG=scaled scripts/training/slurm/tide_registry.sh

set -e

cd /u201/s6jindal/nocturnal-hypo-gly-prob-forecast || exit 1

eval "$(/opt/anaconda3/bin/conda shell.bash hook)"
conda activate chronos

# Default to best config if not overridden
CONFIG=${CONFIG:-best}
HOLDOUT_DIR=${HOLDOUT_DIR:-configs/data/holdout_5pct}
TIME_LIMIT=${TIME_LIMIT:-7200}

echo "=========================================="
echo "TiDE Registry Experiment: ${CONFIG}"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo ""
echo "Config: ${CONFIG}"
echo "Holdout dir: ${HOLDOUT_DIR}"
echo "Time limit: ${TIME_LIMIT}s"
echo ""
echo "Conda env: $(conda info --envs | grep '*')"
echo "Python: $(which python)"
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo ""

python scripts/tide_registry_experiment.py \
    --config "${CONFIG}" \
    --holdout-dir "${HOLDOUT_DIR}" \
    --time-limit "${TIME_LIMIT}" \
    --max-eval-episodes 500

echo ""
echo "Experiment complete at $(date)"
