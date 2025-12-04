# MLflow Setup on SLURM Cluster

This README documents the setup process for running MLflow on a SLURM cluster with SSH tunneling for web UI access.

## Overview

We set up MLflow to run on the cluster's login node and access it via SSH port forwarding from a local machine. This approach provides flexibility for experimentation without requiring permanent infrastructure.

## Prerequisites

- Access to a SLURM cluster
- SSH key authentication configured
- MLflow installed in your conda/pip environment
- Local machine with SSH client

## Setup Steps

### 1. Create MLflow Directory

On the cluster, create a dedicated directory for MLflow:

```bash
mkdir -p /nocturnal/mlflow_experiments
cd /nocturnal/mlflow_experiments
```

### 2. Create MLflow Startup Script

Create the startup script with port checking:

```bash
# filepath: ./nocturnal/mlflow_experiments/start_mlflow_simple.sh
#!/bin/bash

PORT=${1:-5555}

# Simple port check using ss instead of netstat
if ss -tuln | grep -q ":$PORT "; then
    echo "Port $PORT is already in use!"
    echo "Try a different port: ./start_mlflow_simple.sh 5556"
    exit 1
fi

echo "Starting MLflow server on port $PORT"
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./artifacts \
    --host 0.0.0.0 \
    --port $PORT
```

Make it executable:
```bash
chmod +x start_mlflow_simple.sh
```

### 3. Configure SSH (if not already done)

Ensure your local machine has SSH config for the cluster:

```bash
# ~/.ssh/config
Host watGPU
    User [USERNAME]
    IdentityFile ~/.ssh/watgpu
    HostName watgpu.cs.uwaterloo.ca
```

## Usage Workflow

### Starting MLflow Server

1. **SSH into the cluster:**
   ```bash
   ssh watGPU
   ```

2. **Navigate to MLflow directory:**
   ```bash
   cd /u6/cjrisi/nocturnal/mlflow_experiments
   ```

3. **Start MLflow server:**
   ```bash
   ./start_mlflow_simple.sh 5555
   ```

   The script will:
   - Check if port 5555 is available
   - Start MLflow server on that port
   - Use SQLite database for experiment tracking
   - Store artifacts in local `./artifacts` directory

### Accessing MLflow UI

1. **Create SSH tunnel (from your local machine):**
   ```bash
   ssh -L 5555:localhost:5555 watGPU
   ```

2. **Open web browser:**
   Navigate to: `http://localhost:5555`

### Stopping MLflow Server

On the cluster, stop the server with `Ctrl+C` or:
```bash
pkill -f "mlflow server"
```

## Troubleshooting

### Port Already in Use
If you get a port conflict:
```bash
./start_mlflow_simple.sh 5556  # Try different port
ssh -L 5556:localhost:5556 watGPU  # Update tunnel accordingly
```

### SSH Tunnel Issues
- Verify basic SSH connection works: `ssh watGPU`
- Try alternative tunnel syntax: `ssh -L 5555:127.0.0.1:5555 watGPU`
- Check if MLflow is running: `ps aux | grep mlflow`

### Network Resolution Errors
- Ensure you're creating the tunnel from your local machine, not the cluster
- Verify SSH config is correct
- Try using full hostname instead of alias

## File Structure

```
/u6/cjrisi/nocturnal/mlflow_experiments/
├── start_mlflow_simple.sh    # Startup script
├── mlflow.db                 # SQLite database (created automatically)
├── artifacts/                # MLflow artifacts directory
└── README.md                 # This file
```

## Integration with Experiments

To use this MLflow server in your experiments, set the tracking URI:

```python
import mlflow

# Set tracking URI to your server
mlflow.set_tracking_uri("http://localhost:5555")

# Create or use experiment
mlflow.set_experiment("your_experiment_name")
```

## Future Migration

This setup uses local SQLite and file storage, making it easy to migrate to other MLflow backends (PostgreSQL, cloud storage, etc.) when needed.
