#!/bin/bash
#
# Submit per-patient Stage 2 fine-tuning for ALL holdout patients.
#
# This is a launcher script (runs on the login node) that submits one
# SLURM job per holdout patient via per_patient_finetune.sh.
#
# Usage:
#   source .venvs/chronos2/bin/activate   # needed for sbatch env inheritance
#   bash scripts/training/slurm/per_patient_finetune_all.sh
#
# Override defaults:
#   CHECKPOINT=models/other/path bash scripts/training/slurm/per_patient_finetune_all.sh
#   DATASET=lynch_2022 PATIENTS="lyn_270 lyn_400" bash scripts/training/slurm/per_patient_finetune_all.sh

: ${DATASET:="brown_2019"}
: ${CHECKPOINT:="models/chronos2_stage1/20260224_234112"}
: ${MODEL:="chronos2"}

# Brown 2019 holdout patients (all 8, ~191-204 days each)
# bro_62 already ran — included here for completeness, skip with PATIENTS override if needed
: ${PATIENTS:="bro_3 bro_31 bro_35 bro_62 bro_88 bro_134 bro_152 bro_164"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SINGLE_SCRIPT="$SCRIPT_DIR/per_patient_finetune.sh"

echo "========================================="
echo "Submitting Stage 2 jobs for ALL holdout patients"
echo "========================================="
echo "Dataset:    $DATASET"
echo "Checkpoint: $CHECKPOINT"
echo "Model:      $MODEL"
echo "Patients:   $PATIENTS"
echo ""

submitted=0
for patient in $PATIENTS; do
    # --export=ALL is SLURM default — passes all env vars including MODEL,
    # CHECKPOINT, DATASET. We override PATIENT per iteration.
    job_output=$(export PATIENT="$patient" MODEL="$MODEL" DATASET="$DATASET" \
        CHECKPOINT="$CHECKPOINT" && \
        sbatch --job-name="s2_${patient}" "$SINGLE_SCRIPT" 2>&1)

    if echo "$job_output" | grep -q "Submitted"; then
        job_id=$(echo "$job_output" | grep -oP '\d+')
        echo "  $patient -> Job $job_id"
        submitted=$((submitted + 1))
    else
        echo "  $patient -> FAILED: $job_output"
    fi
done

echo ""
echo "Submitted $submitted / $(echo $PATIENTS | wc -w) jobs"
echo "Monitor with: squeue -u $USER"
echo "Results in:   experiments/per_patient_finetune/chronos2/$DATASET/"
