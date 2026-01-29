#!/bin/bash
#
# PRODUCTION MULTI-GPU TRAINING (DDP)
# =====================================
# Use this for distributed training across multiple GPUs
#
# Quick Start:
#   sbatch scripts/training/slurm/multi_gpu.sh
#
# Specify number of GPUs:
#   sbatch --export=NUM_GPUS=4 scripts/training/slurm/multi_gpu.sh
#
#SBATCH --job-name=ttm_train_multi
#SBATCH --output=logs/train_multi_%j.out
#SBATCH --error=logs/train_multi_%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=128GB
#SBATCH --gres=gpu:4
#SBATCH --partition=gpu
##SBATCH --mail-user=your.email@example.com
##SBATCH --mail-type=BEGIN,END,FAIL

# =============================================================================
# CONFIGURATION
# =============================================================================

# Default configuration (can be overridden with --export)
: ${NUM_GPUS:=4}
: ${CONFIG_PATH:="configs/models/ttm/fine_tune.yaml"}
: ${DATA_CONFIG:="configs/data/kaggle_bris_t1d.yaml"}
: ${OUTPUT_DIR:="trained_models/artifacts/ttm"}
: ${EXPERIMENT_NAME:="multi_gpu_training"}

# Distributed training settings
: ${MASTER_PORT:=29500}

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

echo "========================================="
echo "TTM Multi-GPU Distributed Training"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo ""
echo "Configuration:"
echo "  GPUs: $NUM_GPUS"
echo "  Model config: $CONFIG_PATH"
echo "  Data config: $DATA_CONFIG"
echo "  Output dir: $OUTPUT_DIR"
echo "  Experiment: $EXPERIMENT_NAME"
echo "  Master port: $MASTER_PORT"
echo "========================================="
echo ""

# Load modules if needed (uncomment and adjust for your cluster)
# module load cuda/11.8
# module load python/3.10
# module load nccl

# Activate virtual environment
echo "Activating environment..."
source /u6/cjrisi/nocturnal/.noctprob-venv/bin/activate

# Navigate to project root
cd /u6/cjrisi/nocturnal

# =============================================================================
# HARDWARE VERIFICATION
# =============================================================================

echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv
echo ""
echo "NCCL Version:"
python -c "import torch; print(torch.cuda.nccl.version())" 2>/dev/null || echo "NCCL not available"
echo ""

# =============================================================================
# DISTRIBUTED TRAINING SETUP
# =============================================================================

# Set environment variables for distributed training
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT=$MASTER_PORT
export WORLD_SIZE=$NUM_GPUS
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / NUM_GPUS))

# NCCL optimizations
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=^docker0,lo

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p "logs"

# =============================================================================
# TRAINING
# =============================================================================

echo "Starting distributed training with $NUM_GPUS GPUs..."
echo "Using torchrun for process coordination..."
echo ""

# Run distributed training with torchrun
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    scripts/examples/example_distributed_ttm.py \
    --output_dir "$OUTPUT_DIR/$EXPERIMENT_NAME" \
    2>&1 | tee "logs/train_${SLURM_JOB_ID}.log"

# Capture exit code
exit_code=$?

# =============================================================================
# COMPLETION
# =============================================================================

echo ""
echo "========================================="
echo "Training completed: $(date)"
echo "Exit code: $exit_code"
echo "Duration: $SECONDS seconds"
echo "========================================="

if [ $exit_code -eq 0 ]; then
    echo "✅ SUCCESS: Model saved to $OUTPUT_DIR/$EXPERIMENT_NAME"
    echo ""
    echo "Training efficiency metrics:"
    python -c "
import sys
duration_sec = $SECONDS
num_gpus = $NUM_GPUS
duration_hours = duration_sec / 3600
print(f'  Total time: {duration_hours:.2f} hours')
print(f'  GPU-hours: {duration_hours * num_gpus:.2f}')
print(f'  Effective speedup: {num_gpus}x (theoretical)')
"
else
    echo "❌ FAILED: Check logs/train_${SLURM_JOB_ID}.log for details"
fi

exit $exit_code
