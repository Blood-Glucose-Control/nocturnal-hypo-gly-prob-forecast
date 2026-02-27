#!/bin/bash
#SBATCH --job-name=chronos2_ft
#SBATCH --output=logs/chronos2_ft_%j.out
#SBATCH --error=logs/chronos2_ft_%j.err
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --partition=HI

# Stage 1 Chronos-2 fine-tuning on a dataset.
#
# Usage:
#   sbatch scripts/training/slurm/chronos2_finetune.sh
#
# Override defaults:
#   DATASET=brown_2019 sbatch scripts/training/slurm/chronos2_finetune.sh
#   STEPS=5000 COV="iob cob" sbatch scripts/training/slurm/chronos2_finetune.sh

: ${DATASET:="brown_2019"}
: ${STEPS:="15000"}
: ${LR:="1e-5"}
: ${COV:="iob"}
: ${CONFIG_DIR:="configs/data/holdout_10pct"}
: ${OUTPUT_DIR:="models/chronos2_${DATASET}"}

if [ -n "$SLURM_SUBMIT_DIR" ]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
    PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
fi
cd "$PROJECT_ROOT" || exit 1

mkdir -p logs

echo "========================================="
echo "Chronos-2 Stage 1 Fine-Tuning (SLURM)"
echo "========================================="
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURM_NODELIST"
echo "Dataset:    $DATASET"
echo "Steps:      $STEPS"
echo "LR:         $LR"
echo "Covariates: $COV"
echo "Output:     $OUTPUT_DIR"
echo "Started:    $(date)"
echo "========================================="

source ".venvs/chronos2/bin/activate"
echo "Python: $(which python)"
echo "Python version: $(python --version)"

nvidia-smi --query-gpu=index,name,memory.total --format=csv
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-"expandable_segments:True"}

echo ""
echo "Running example_chronos2_finetune.py..."
echo ""

python scripts/examples/example_chronos2_finetune.py \
    --dataset "$DATASET" \
    --steps "$STEPS" \
    --lr "$LR" \
    --covariates $COV \
    --config-dir "$CONFIG_DIR" \
    --output-dir "$OUTPUT_DIR"

exit_code=$?

echo ""
echo "========================================="
echo "Completed: $(date)"
echo "Exit code: $exit_code"
echo "========================================="

exit $exit_code
