#!/bin/bash
#
# Submit per-patient Stage 2 fine-tuning for ALL holdout patients.
#
# Two modes:
#   1. SINGLE JOB (default): Runs all patients sequentially in one SLURM job
#      using --all-patients flag. Produces a unified summary.csv.
#   2. FAN-OUT: Submits one SLURM job per patient for parallel execution.
#      Set MODE=fanout to use this mode.
#
# Usage:
#   # Single job (default) — submit one SLURM job that iterates over all patients:
#   bash scripts/training/slurm/per_patient_finetune_all.sh
#
#   # Fan-out — submit separate SLURM jobs per patient:
#   MODE=fanout bash scripts/training/slurm/per_patient_finetune_all.sh
#
# Override defaults:
#   CHECKPOINT=models/other/path bash scripts/training/slurm/per_patient_finetune_all.sh
#   DATASET=lynch_2022 PATIENTS="lyn_270 lyn_400" bash scripts/training/slurm/per_patient_finetune_all.sh

: ${DATASET:="brown_2019"}
: ${CHECKPOINT:="models/chronos2_stage1/20260224_234112"}
: ${MODEL:="chronos2"}
: ${CONFIG_DIR:="configs/data/holdout_10pct"}
: ${MODE:="single"}
: ${FT_STEPS:="5000"}
: ${LR:="1e-5"}
: ${COV:="iob"}
: ${TEST_DAYS:="14"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================="
echo "Stage 2 Fine-Tuning — ALL Holdout Patients"
echo "========================================="
echo "Mode:       $MODE"
echo "Dataset:    $DATASET"
echo "Checkpoint: $CHECKPOINT"
echo "Model:      $MODEL"
echo "Config dir: $CONFIG_DIR"
echo ""

if [ "$MODE" = "fanout" ]; then
    # ── Fan-out mode: one SLURM job per patient ────────────────────────────
    SINGLE_SCRIPT="$SCRIPT_DIR/per_patient_finetune.sh"

    # Read holdout patients from config YAML (override with PATIENTS env var)
    if [ -z "$PATIENTS" ]; then
        CONFIG_FILE="${CONFIG_DIR}/${DATASET}.yaml"
        if [ ! -f "$CONFIG_FILE" ]; then
            echo "ERROR: Config file not found: $CONFIG_FILE"
            exit 1
        fi
        PATIENTS=$(grep '^\s*- ' "$CONFIG_FILE" | sed 's/^\s*- //' | tr '\n' ' ')
        if [ -z "$PATIENTS" ]; then
            echo "ERROR: No holdout patients found in $CONFIG_FILE"
            exit 1
        fi
    fi

    echo "Patients: $PATIENTS"
    echo ""

    # Create shared batch output directory
    TIMESTAMP=$(date +"%Y-%m-%d_%H:%M")
    RUN_ID="RID$(date +"%Y%m%d_%H%M%S")_$$"
    BATCH_DIR="trained_models/artifacts/${MODEL}/${TIMESTAMP}_${RUN_ID}_per_patient_finetune"
    mkdir -p "$BATCH_DIR"
    echo "Batch output: $BATCH_DIR"
    echo ""

    submitted=0
    for patient in $PATIENTS; do
        job_output=$(export PATIENT="$patient" MODEL="$MODEL" DATASET="$DATASET" \
            CHECKPOINT="$CHECKPOINT" CONFIG_DIR="$CONFIG_DIR" \
            FT_STEPS="$FT_STEPS" LR="$LR" COV="$COV" TEST_DAYS="$TEST_DAYS" \
            OUTPUT_DIR="$BATCH_DIR" && \
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

else
    # ── Single-job mode: --all-patients in one SLURM submission ─────────────
    # Create a temporary SLURM script with the --all-patients flag
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=stage2_all_${DATASET}
#SBATCH --output=logs/stage2_all_%j.out
#SBATCH --error=logs/stage2_all_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --partition=HI

if [ -n "\$SLURM_SUBMIT_DIR" ]; then
    PROJECT_ROOT="\$SLURM_SUBMIT_DIR"
else
    PROJECT_ROOT="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")/../../.." && pwd)"
fi
cd "\$PROJECT_ROOT" || exit 1

mkdir -p logs

echo "========================================="
echo "Per-Patient Stage 2 — ALL Patients (SLURM)"
echo "========================================="
echo "Job ID:     \$SLURM_JOB_ID"
echo "Node:       \$SLURM_NODELIST"
echo "Model:      $MODEL"
echo "Checkpoint: $CHECKPOINT"
echo "Dataset:    $DATASET"
echo "Started:    \$(date)"
echo "========================================="

source ".venvs/${MODEL}/bin/activate"
echo "Python: \$(which python)"

nvidia-smi --query-gpu=index,name,memory.total --format=csv
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_ALLOC_CONF=\${PYTORCH_ALLOC_CONF:-"expandable_segments:True"}

echo ""
echo "Running per_patient_finetune.py --all-patients..."
echo ""

python scripts/experiments/per_patient_finetune.py \\
    --model "$MODEL" \\
    --checkpoint "$CHECKPOINT" \\
    --dataset "$DATASET" \\
    --all-patients \\
    --test-days $TEST_DAYS \\
    --fine-tune-steps $FT_STEPS \\
    --learning-rate $LR \\
    --covariate-cols $COV \\
    --config-dir "$CONFIG_DIR"

exit_code=\$?

echo ""
echo "========================================="
echo "Completed: \$(date)"
echo "Exit code: \$exit_code"
echo "========================================="

exit \$exit_code
EOF

    echo "Single-job mode: SLURM job submitted"
    echo "Monitor with: squeue -u $USER"
fi

echo ""
echo "Results in: trained_models/artifacts/$MODEL/"
