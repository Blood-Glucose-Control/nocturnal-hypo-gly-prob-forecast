#!/usr/bin/env python3
"""
Quick script to view the model registry
"""

import sys
from pathlib import Path

# Add the parent directory to path to import model_registry
sys.path.append(str(Path(__file__).parent))

from model_registry import ModelRegistry

def main():
    registry = ModelRegistry()
    print(registry.get_summary())
    
    # Show recent runs in detail
    import pandas as pd
    if registry.registry_path.exists():
        df = pd.read_csv(registry.registry_path)
        if len(df) > 0:
            print("\nRecent Run Details:")
            print("=" * 80)
            recent_runs = df.tail(3)
            for _, run in recent_runs.iterrows():
                print(f"Run: {run['run_id']}")
                print(f"  Experiment: {run['experiment_name']}")
                print(f"  Status: {run['status']}")
                print(f"  Batch Size: {run['batch_size']}")
                print(f"  Learning Rate: {run['learning_rate']}")
                print(f"  Node: {run['node_name']}")
                print(f"  Start Time: {run['start_time']}")
                if pd.notna(run['elapsed_time_seconds']):
                    elapsed_min = float(run['elapsed_time_seconds']) / 60
                    print(f"  Duration: {elapsed_min:.1f} minutes")
                print()

if __name__ == "__main__":
    main()