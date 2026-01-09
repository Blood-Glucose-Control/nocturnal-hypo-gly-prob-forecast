#!/bin/bash
#
# ADAPTIVE RESOURCE ALLOCATION
# ==============================
# Automatically detects available GPUs and launches training
# with optimal configuration
#
# Quick Start:
#   sbatch scripts/training/slurm/adaptive_resources.sh
#
# Override detection:
#   sbatch --export=FORCE_NUM_GPUS=2 scripts/training/slurm/adaptive_resources.sh
#
#SBATCH --job-name=ttm_train_auto
#SBATCH --output=logs/train_auto_%j.out
#SBATCH --error=logs/train_auto_%j.err
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
: ${CONFIG_PATH:="configs/models/ttm/fine_tune.yaml"}
: ${DATA_CONFIG:="configs/data/kaggle_bris_t1d.yaml"}
: ${OUTPUT_DIR:="trained_models/artifacts/ttm"}
: ${EXPERIMENT_NAME:="adaptive_training"}

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

echo "========================================="
echo "TTM Adaptive Resource Training"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "========================================="
echo ""

# Activate virtual environment
echo "Activating environment..."
source /u6/cjrisi/nocturnal/.noctprob-venv/bin/activate

# Navigate to project root
cd /u6/cjrisi/nocturnal

# =============================================================================
# HARDWARE DETECTION
# =============================================================================

echo "Detecting hardware configuration..."
echo ""

# Detect number of GPUs
if [ -n "$FORCE_NUM_GPUS" ]; then
    NUM_GPUS=$FORCE_NUM_GPUS
    echo "🔧 Using forced GPU count: $NUM_GPUS"
else
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
    echo "🔍 Detected $NUM_GPUS GPUs automatically"
fi

echo ""
echo "GPU Details:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
echo ""

# =============================================================================
# RESOURCE CONFIGURATION
# =============================================================================

# Adjust resources based on GPU count
if [ "$NUM_GPUS" -eq 0 ]; then
    echo "⚠️  WARNING: No GPUs detected! Training will use CPU (very slow)"
    USE_CPU=true
    STRATEGY="cpu"
elif [ "$NUM_GPUS" -eq 1 ]; then
    echo "📱 Single GPU detected - using simple training"
    STRATEGY="single_gpu"
    export CUDA_VISIBLE_DEVICES=0
elif [ "$NUM_GPUS" -le 4 ]; then
    echo "🚀 Multiple GPUs detected - using DDP strategy"
    STRATEGY="multi_gpu_ddp"
else
    echo "🚀🚀 Large GPU allocation - using optimized DDP"
    STRATEGY="multi_gpu_ddp_optimized"
fi

# Set optimal thread count
if [ "$NUM_GPUS" -gt 1 ]; then
    export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / NUM_GPUS))
else
    export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
fi

echo ""
echo "Configuration selected:"
echo "  Strategy: $STRATEGY"
echo "  GPUs: $NUM_GPUS"
echo "  Threads per GPU: $OMP_NUM_THREADS"
echo "  Model config: $CONFIG_PATH"
echo "  Output dir: $OUTPUT_DIR"
echo "========================================="
echo ""

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "logs"

# =============================================================================
# TRAINING EXECUTION
# =============================================================================

echo "Starting training with $STRATEGY strategy..."
echo ""

# Set distributed training environment variables
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT=29500

case $STRATEGY in
    cpu)
        echo "Running CPU training (not recommended for large models)..."
        python scripts/examples/example_single_gpu_ttm.py \
            --output_dir "$OUTPUT_DIR/$EXPERIMENT_NAME" \
            --use_cpu \
            2>&1 | tee "logs/train_${SLURM_JOB_ID}.log"
        ;;
    
    single_gpu)
        echo "Running single GPU training..."
        python scripts/examples/example_single_gpu_ttm.py \
            --output_dir "$OUTPUT_DIR/$EXPERIMENT_NAME" \
            2>&1 | tee "logs/train_${SLURM_JOB_ID}.log"
        ;;
    
    multi_gpu_ddp|multi_gpu_ddp_optimized)
        echo "Running distributed training with torchrun..."
        echo "Command: torchrun --nproc_per_node=$NUM_GPUS"
        
        # NCCL optimizations
        export NCCL_DEBUG=INFO
        export NCCL_IB_DISABLE=0
        export NCCL_SOCKET_IFNAME=^docker0,lo
        
        torchrun \
            --nproc_per_node=$NUM_GPUS \
            --nnodes=1 \
            --node_rank=0 \
            --master_addr=$MASTER_ADDR \
            --master_port=$MASTER_PORT \
            scripts/examples/example_distributed_ttm.py \
            --output_dir "$OUTPUT_DIR/$EXPERIMENT_NAME" \
            2>&1 | tee "logs/train_${SLURM_JOB_ID}.log"
        ;;
    
    *)
        echo "❌ ERROR: Unknown strategy: $STRATEGY"
        exit 1
        ;;
esac

# Capture exit code
exit_code=$?

# =============================================================================
# COMPLETION AND REPORTING
# =============================================================================

echo ""
echo "========================================="
echo "Training completed: $(date)"
echo "Exit code: $exit_code"
echo "Duration: $SECONDS seconds ($(($SECONDS / 3600))h $(($SECONDS % 3600 / 60))m)"
echo "========================================="

if [ $exit_code -eq 0 ]; then
    echo "✅ SUCCESS: Model saved to $OUTPUT_DIR/$EXPERIMENT_NAME"
    echo ""
    
    # Calculate efficiency metrics
    if [ "$NUM_GPUS" -gt 0 ]; then
        duration_hours=$(echo "scale=2; $SECONDS / 3600" | bc)
        gpu_hours=$(echo "scale=2; $duration_hours * $NUM_GPUS" | bc)
        echo "Resource usage:"
        echo "  Total time: ${duration_hours}h"
        echo "  GPU-hours: ${gpu_hours}"
        echo "  Strategy used: $STRATEGY"
    fi
else
    echo "❌ FAILED: Check logs/train_${SLURM_JOB_ID}.log for details"
    echo ""
    echo "Troubleshooting tips:"
    echo "  1. Check GPU availability: nvidia-smi"
    echo "  2. Verify environment: source .noctprob-venv/bin/activate"
    echo "  3. Review error logs above"
    echo "  4. For distributed training issues, check NCCL configuration"
fi

exit $exit_code
