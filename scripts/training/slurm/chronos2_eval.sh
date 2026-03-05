#!/bin/bash
#SBATCH --job-name=chronos2_eval
#SBATCH --output=logs/chronos2_eval_%j.out
#SBATCH --error=logs/chronos2_eval_%j.err
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --partition=HI

# Evaluate fine-tuned Chronos-2 on nocturnal forecasting task.
#
# Usage:
#   CHECKPOINT=models/chronos2_brown_2019/<timestamp> sbatch scripts/training/slurm/chronos2_eval.sh
#
# Override defaults:
#   DATASET=aleppo_2017 CHECKPOINT=path/to/model sbatch scripts/training/slurm/chronos2_eval.sh

: ${DATASET:="brown_2019"}
: ${CONFIG_DIR:="configs/data/holdout_10pct"}
: ${CHECKPOINT:=""}
: ${COV:="iob"}

if [ -z "$CHECKPOINT" ]; then
    echo "ERROR: CHECKPOINT must be set. Usage:"
    echo "  CHECKPOINT=models/chronos2_brown_2019/<timestamp> sbatch $0"
    exit 1
fi

if [ -n "$SLURM_SUBMIT_DIR" ]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
    PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
fi
cd "$PROJECT_ROOT" || exit 1

mkdir -p logs

echo "========================================="
echo "Chronos-2 Nocturnal Evaluation (SLURM)"
echo "========================================="
echo "Job ID:      $SLURM_JOB_ID"
echo "Node:        $SLURM_NODELIST"
echo "Dataset:     $DATASET"
echo "Checkpoint:  $CHECKPOINT"
echo "Config Dir:  $CONFIG_DIR"
echo "Started:     $(date)"
echo "========================================="

source ".venvs/chronos2/bin/activate"
echo "Python: $(which python)"

nvidia-smi --query-gpu=index,name,memory.total --format=csv
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

echo ""
echo "Running nocturnal_hypo_eval.py..."
echo ""

python scripts/experiments/nocturnal_hypo_eval.py \
    --model chronos2 \
    --dataset "$DATASET" \
    --config-dir "$CONFIG_DIR" \
    --checkpoint "$CHECKPOINT" \
    --context-length 512 \
    --forecast-length 72 \
    --covariate-cols $COV \
    --holdout-patients-only

exit_code=$?

echo ""
echo "========================================="
echo "Completed: $(date)"
echo "Exit code: $exit_code"
echo "========================================="

exit $exit_code
