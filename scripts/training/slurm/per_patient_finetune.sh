#!/bin/bash
#SBATCH --job-name=stage2_ft
#SBATCH --output=logs/stage2_ft_%j.out
#SBATCH --error=logs/stage2_ft_%j.err
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --partition=HI

# Per-patient Stage 2 fine-tuning on a holdout patient.
# Defaults target bro_62 on brown_2019 with the Stage 1 Chronos2 checkpoint.
#
# Usage:
#   sbatch scripts/training/slurm/per_patient_finetune.sh
#
# Override defaults:
#   PATIENT=bro_164 sbatch scripts/training/slurm/per_patient_finetune.sh
#   CHECKPOINT=models/other/path sbatch scripts/training/slurm/per_patient_finetune.sh

: ${MODEL:="chronos2"}
: ${CHECKPOINT:="models/chronos2_stage1/20260224_234112"}
: ${DATASET:="brown_2019"}
: ${PATIENT:="bro_62"}
: ${TEST_DAYS:="14"}
: ${FT_STEPS:="5000"}
: ${LR:="1e-5"}
: ${COV:="iob"}

# Use SLURM_SUBMIT_DIR (directory where sbatch was run) — BASH_SOURCE
# doesn't work in SLURM because the script runs from /var/spool/.
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
    PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
fi
cd "$PROJECT_ROOT" || exit 1

mkdir -p logs

echo "========================================="
echo "Per-Patient Stage 2 Fine-Tuning (SLURM)"
echo "========================================="
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURM_NODELIST"
echo "Model:      $MODEL"
echo "Checkpoint: $CHECKPOINT"
echo "Dataset:    $DATASET"
echo "Patient:    $PATIENT"
echo "Started:    $(date)"
echo "========================================="

# Activate chronos2 venv
source ".venvs/${MODEL}/bin/activate"
echo "Python: $(which python)"
echo "Python version: $(python --version)"

# GPU info
nvidia-smi --query-gpu=index,name,memory.total --format=csv
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-"expandable_segments:True"}

echo ""
echo "Running per_patient_finetune.py..."
echo ""

python scripts/experiments/per_patient_finetune.py \
    --model "$MODEL" \
    --checkpoint "$CHECKPOINT" \
    --dataset "$DATASET" \
    --patient-id "$PATIENT" \
    --test-days "$TEST_DAYS" \
    --fine-tune-steps "$FT_STEPS" \
    --learning-rate "$LR" \
    --covariate-cols $COV

exit_code=$?

echo ""
echo "========================================="
echo "Completed: $(date)"
echo "Exit code: $exit_code"
echo "========================================="

exit $exit_code
