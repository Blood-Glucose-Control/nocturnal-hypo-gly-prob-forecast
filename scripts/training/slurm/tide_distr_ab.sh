#!/bin/bash
#SBATCH --job-name=tide_distr
#SBATCH --partition=HI
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/tide_distr_%A_%a.out
#SBATCH --error=logs/tide_distr_%A_%a.err
#SBATCH --array=0-1

# TiDE Distribution A/B Experiment
# Array job: 0=student_t (baseline), 1=normal (treatment)

set -e

cd /u201/s6jindal/nocturnal-hypo-gly-prob-forecast || exit 1

# Activate conda environment
eval "$(/opt/anaconda3/bin/conda shell.bash hook)"
conda activate chronos

CONFIGS=("student_t" "normal")
CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}

echo "=========================================="
echo "TiDE Distribution A/B: ${CONFIG}"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}, Array Task: ${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo ""
echo "Conda env: $(conda info --envs | grep '*')"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo ""

echo "Starting TiDE distr_output=${CONFIG} experiment..."
python scripts/tide_distr_ab_experiment.py \
    --config "${CONFIG}" \
    --time-limit 7200 \
    --max-eval-episodes 500

echo ""
echo "Experiment complete at $(date)"
