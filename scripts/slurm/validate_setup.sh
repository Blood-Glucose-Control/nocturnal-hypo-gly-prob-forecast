#!/bin/bash
# Test script to validate SLURM setup before submitting jobs

echo "SLURM Training Pipeline Validation"
echo "================================="

# Check if we're on a SLURM system
if ! command -v sbatch &> /dev/null; then
    echo "⚠️  WARNING: sbatch command not found. This doesn't appear to be a SLURM system."
    echo "   You can still test the training script directly with Python."
else
    echo "✅ SLURM system detected"
fi

# Check SLURM scripts exist and are executable
scripts=("scripts/slurm/train_cpu.sh" "scripts/slurm/train_gpu.sh" "scripts/slurm/submit_arima_job.sh")
for script in "${scripts[@]}"; do
    if [ -x "$script" ]; then
        echo "✅ $script is executable"
    elif [ -f "$script" ]; then
        echo "⚠️  $script exists but is not executable (run: chmod +x $script)"
    else
        echo "❌ $script not found"
    fi
done

# Check config files
echo ""
echo "Available training configs:"
if ls scripts/training/configs/*.yaml 1> /dev/null 2>&1; then
    for config in scripts/training/configs/*.yaml; do
        echo "  ✅ $(basename $config)"
    done
else
    echo "  ❌ No config files found in scripts/training/configs/"
fi

# Check training scripts
echo ""
echo "Training scripts:"
training_scripts=("scripts/training/train_statistical.py")
for script in "${training_scripts[@]}"; do
    if [ -f "$script" ]; then
        echo "  ✅ $script exists"
    else
        echo "  ❌ $script not found"
    fi
done

# Check virtual environment
echo ""
echo "Environment check:"
if [ -d ".noctprob-venv" ]; then
    echo "  ✅ Virtual environment found: .noctprob-venv"
    if [ -f ".noctprob-venv/bin/activate" ]; then
        echo "  ✅ Activation script exists"
    else
        echo "  ❌ Activation script not found"
    fi
else
    echo "  ⚠️  Virtual environment not found: .noctprob-venv"
    echo "     You may need to create and activate your Python environment"
fi

# Check logs directory
if [ -d "logs" ]; then
    echo "  ✅ Logs directory exists"
else
    echo "  ⚠️  Logs directory will be created when needed"
fi

# Check models directory
if [ -d "models" ]; then
    echo "  ✅ Models directory exists"
else
    echo "  ⚠️  Models directory will be created when needed"
fi

echo ""
echo "Validation complete!"
echo ""
echo "Next steps:"
echo "1. If you're on a SLURM cluster, submit a job with:"
echo "   ./scripts/slurm/submit_arima_job.sh"
echo ""
echo "2. For testing without SLURM, run directly:"
echo "   python scripts/training/train_statistical.py --config scripts/training/configs/arima_config.yaml"
echo ""
echo "3. Check available SLURM partitions with:"
echo "   sinfo"
echo ""
echo "4. Check your job queue with:"
echo "   squeue -u $USER"
