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
