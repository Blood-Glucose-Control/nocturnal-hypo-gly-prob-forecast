#!/bin/bash
#SBATCH --job-name=toto_eval
#SBATCH --output=logs/toto_eval_%j.out
#SBATCH --error=logs/toto_eval_%j.err
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --partition=HI

# Evaluate Toto zero-shot on nocturnal forecasting task.
#
# Usage:
#   sbatch scripts/training/slurm/toto_eval.sh
#
# Override defaults:
#   DATASET=tamborlane_2008 sbatch scripts/training/slurm/toto_eval.sh

: ${DATASET:="brown_2019"}
: ${CONFIG_DIR:="configs/data/holdout_10pct"}

if [ -n "$SLURM_SUBMIT_DIR" ]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
    PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
fi
cd "$PROJECT_ROOT" || exit 1

mkdir -p logs

echo "========================================="
echo "Toto Nocturnal Evaluation (SLURM)"
echo "========================================="
echo "Job ID:      $SLURM_JOB_ID"
echo "Node:        $SLURM_NODELIST"
echo "Dataset:     $DATASET"
echo "Config Dir:  $CONFIG_DIR"
echo "Started:     $(date)"
echo "========================================="

source ".venvs/toto/bin/activate"
echo "Python: $(which python)"

nvidia-smi --query-gpu=index,name,memory.total --format=csv
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

echo ""
echo "Running nocturnal_hypo_eval.py..."
echo ""

python scripts/experiments/nocturnal_hypo_eval.py \
    --model toto \
    --dataset "$DATASET" \
    --config-dir "$CONFIG_DIR" \
    --context-length 512 \
    --forecast-length 72 \
    --cuda-device 0

exit_code=$?

echo ""
echo "========================================="
echo "Completed: $(date)"
echo "Exit code: $exit_code"
echo "========================================="

exit $exit_code
