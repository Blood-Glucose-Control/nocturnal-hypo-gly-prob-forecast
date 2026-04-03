#!/bin/bash

# Not tested and heavily vibe coded
# Killing the script process don't do shit
# The idea is to find the direct children of the bash script which is the main python process that spawns the worker processes.
# Stop all processes spawned by gluroo_data_processing_local.sh
# Usage: bash scripts/data_processing_scripts/loader_local_shells/gluroo_data_processing_local_stop.sh

SCRIPT_NAME="gluroo_data_processing_local.sh"

# Find the bash script PID
bash_pid=$(pgrep -f "$SCRIPT_NAME")
if [ -z "$bash_pid" ]; then
    echo "No running processes found for $SCRIPT_NAME."
    exit 0
fi
echo "Found bash script PID: $bash_pid"

# Find the main Python process (direct child of the bash script whose cmd is python)
main_python_pid=$(ps -o pid,ppid,cmd --no-headers | awk -v ppid="$bash_pid" '$2 == ppid && $3 ~ /python/ {print $1}')
if [ -z "$main_python_pid" ]; then
    echo "No child Python process found for bash PID $bash_pid. Killing bash script only."
    kill -9 "$bash_pid"
    exit 0
fi
echo "Found main Python PID: $main_python_pid"

# Find all worker PIDs (direct children of the main Python process)
worker_pids=$(ps -o pid,ppid --no-headers | awk -v ppid="$main_python_pid" '$2 == ppid {print $1}')
echo "Found worker PIDs: $worker_pids"

# Kill workers first, then main python, then bash
all_pids="$worker_pids $main_python_pid $bash_pid"
echo "Killing all PIDs: $all_pids"
kill -9 $all_pids 2>/dev/null

sleep 1

# Final check
still_running=$(pgrep -f "$SCRIPT_NAME")
if [ -n "$still_running" ]; then
    echo "WARNING: some processes could not be killed: $still_running"
    exit 1
else
    echo "All $SCRIPT_NAME processes stopped."
fi
