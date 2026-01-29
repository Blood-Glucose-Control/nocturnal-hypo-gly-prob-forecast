# SLURM Training Scripts

Quick reference for running TTM training jobs on your cluster during the hackathon.

## 🚀 Quick Start (Choose Your Path)

### Option 1: Just Want to Test? (Recommended First)
```bash
# Test that everything works (runs for ~1 hour, 1 epoch)
sbatch scripts/examples/run_single_gpu_ttm.sh
```

### Option 2: Production Training

#### Single GPU Training
```bash
# Basic: Uses default config
sbatch scripts/training/slurm/single_gpu.sh

# Custom: Specify your own config
sbatch --export=CONFIG_PATH=configs/models/ttm/custom.yaml scripts/training/slurm/single_gpu.sh
```

#### Multi-GPU Training
```bash
# Use 4 GPUs (default)
sbatch scripts/training/slurm/multi_gpu.sh

# Use 2 GPUs
sbatch --export=NUM_GPUS=2 scripts/training/slurm/multi_gpu.sh
```

#### Adaptive (Automatic)
```bash
# Automatically detects and uses all available GPUs
sbatch scripts/training/slurm/adaptive_resources.sh
```

## 📋 Script Comparison

| Script | When to Use | Time Limit | GPUs | Purpose |
|--------|-------------|------------|------|---------|
| `examples/run_single_gpu_ttm.sh` | **Testing framework** | 1h | 1 | Quick validation |
| `training/slurm/single_gpu.sh` | **Production, 1 GPU** | 24h | 1 | Full training runs |
| `training/slurm/multi_gpu.sh` | **Production, multi-GPU** | 48h | 4 (default) | Distributed training |
| `training/slurm/adaptive_resources.sh` | **Let it choose** | 48h | Auto-detect | Best for experimentation |

## ⚙️ Customization Options

All production scripts accept these environment variables:

```bash
# Example: Full customization
sbatch \
  --export=CONFIG_PATH=configs/models/ttm/my_config.yaml,\
DATA_CONFIG=configs/data/gluroo.yaml,\
EXPERIMENT_NAME=hackathon_exp_001,\
OUTPUT_DIR=trained_models/hackathon \
  scripts/training/slurm/adaptive_resources.sh
```

### Available Variables:
- `CONFIG_PATH`: Model configuration file (default: `configs/models/ttm/fine_tune.yaml`)
- `DATA_CONFIG`: Dataset configuration (default: `configs/data/kaggle_bris_t1d.yaml`)
- `OUTPUT_DIR`: Where to save models (default: `trained_models/artifacts/ttm`)
- `EXPERIMENT_NAME`: Name for this run (default: varies by script)
- `NUM_GPUS`: Force specific GPU count (multi_gpu.sh and adaptive only)

## 🔍 Monitoring Jobs

```bash
# Check job status
squeue -u $USER

# Watch output in real-time
tail -f logs/train_*.out

# Check specific job
tail -f logs/train_12345.out  # Replace 12345 with your job ID

# Cancel a job
scancel 12345
```
## For Quick Iteration
1. **Test first**: Run the example script to verify everything works
2. **Use adaptive**: Let it figure out resources while you focus on configs
3. **Monitor early**: Check the first few minutes of output to catch issues fast

## Resource Allocation
- **Experimenting with hyperparameters**: Use `single_gpu.sh` (faster iterations)
- **Final training runs**: Use `multi_gpu.sh` or `adaptive_resources.sh`
- **Limited GPU availability**: Use `single_gpu.sh` to maximize parallel experiments

## Debugging
If a job fails:
```bash
# Check error log
cat logs/train_*_JOBID.err  # Replace JOBID with your job ID

# Check full output
cat logs/train_JOBID.out

# Verify GPU access
srun --gres=gpu:1 --pty nvidia-smi

# Test environment
srun --gres=gpu:1 --pty bash
source .noctprob-venv/bin/activate
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 📝 Cluster-Specific Setup

**⚠️ IMPORTANT**: Before your first run, update these in ALL 4 scripts (3 production + 1 example):

1. **Partition name** (line with `#SBATCH --partition=`):
   ```bash
   # Find your cluster's GPU partition
   sinfo

   # Update in scripts - change 'gpu' to your partition name
   #SBATCH --partition=your_partition_name
   ```

2. **Email notifications** (optional, lines with `##SBATCH --mail-`):
   ```bash
   # Uncomment and add your email
   #SBATCH --mail-user=your.email@example.com
   #SBATCH --mail-type=BEGIN,END,FAIL
   ```

3. **Module loading** (if your cluster requires it):
   ```bash
   # Uncomment and adjust these lines in each script
   module load cuda/11.8
   module load python/3.10
   ```

## 🏗️ Directory Structure

After running jobs, you'll see:
```
nocturnal/
├── logs/                              # Job outputs
│   ├── train_12345.out               # Standard output
│   ├── train_12345.log               # Training log
│   └── example_ttm_12345.err         # Errors
├── trained_models/
│   └── artifacts/
│       └── ttm/
│           ├── single_gpu_training/   # From single_gpu.sh
│           ├── multi_gpu_training/    # From multi_gpu.sh
│           └── adaptive_training/     # From adaptive_resources.sh
```

## 🆘 Common Issues

### "invalid partition specified"
Your cluster uses a different name for the GPU partition.
- Fix: Run `sinfo`, find GPU partition name, update `#SBATCH --partition=` in scripts

### "No GPUs detected"
GPU allocation didn't work.
- Fix: Check `squeue -u $USER` shows `gpu:N` in TRES
- Try: `scontrol show partition` to see partition limits

### "ModuleNotFoundError"
Python environment not activated properly.
- Fix: Verify path in scripts points to your venv, e.g.: `/path/to/your/.noctprob-venv/bin/activate`

### "NCCL error" (multi-GPU only)
Distributed training communication issue.
- Check: All GPUs are visible: `nvidia-smi`
- Try: Reduce `NUM_GPUS` to test with fewer GPUs first

## 🎉 Hackathon Workflow Example

```bash
# Morning: Quick validation
sbatch scripts/examples/run_single_gpu_ttm.sh

# While that runs: Prepare configs for your experiments
cp configs/models/ttm/fine_tune.yaml configs/models/ttm/hackathon_v1.yaml
# ... edit hackathon_v1.yaml ...

# Launch experiment suite
for lr in 1e-4 1e-5 5e-5; do
  sbatch --export=CONFIG_PATH=configs/models/ttm/hackathon_v1.yaml,\
EXPERIMENT_NAME=lr_${lr} \
    scripts/training/slurm/single_gpu.sh
done

# Afternoon: Full training with best config
sbatch --export=EXPERIMENT_NAME=final_model scripts/training/slurm/multi_gpu.sh

# Monitor
watch -n 10 'squeue -u $USER'
```

Good luck with your hackathon! 🚀
