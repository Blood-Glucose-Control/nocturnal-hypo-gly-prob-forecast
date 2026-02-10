#!/usr/bin/env python3
# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)
# For commercial licensing, contact: christopher/cjrisi AT gluroo/uwaterloo DOT com/ca

"""
Utility script to view and manage data config registry.

Usage:
    # View summary
    python scripts/data_processing_scripts/view_data_config_registry.py

    # View recent entries
    python scripts/data_processing_scripts/view_data_config_registry.py --recent 5

    # Filter by dataset
    python scripts/data_processing_scripts/view_data_config_registry.py --dataset lynch_2022

    # Filter by branch
    python scripts/data_processing_scripts/view_data_config_registry.py --branch main

    # Export to CSV
    python scripts/data_processing_scripts/view_data_config_registry.py --export data_config_registry.csv
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import only what we need to avoid circular imports
# Direct import to avoid triggering __init__.py imports
import importlib.util

spec = importlib.util.spec_from_file_location(
    "data_config_registry",
    Path(__file__).parent.parent.parent / "src/data/versioning/data_config_registry.py",
)
data_config_registry_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_config_registry_module)
DataConfigRegistry = data_config_registry_module.DataConfigRegistry


def main():
    parser = argparse.ArgumentParser(
        description="View and manage data config registry",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--registry-path",
        default="configs/data/data_config_registry.json",
        help="Path to registry file",
    )

    parser.add_argument(
        "--recent",
        type=int,
        metavar="N",
        help="Show N most recent entries",
    )

    parser.add_argument(
        "--dataset",
        help="Filter by dataset name",
    )

    parser.add_argument(
        "--branch",
        help="Filter by git branch",
    )

    parser.add_argument(
        "--export",
        metavar="CSV_FILE",
        help="Export registry to CSV file",
    )

    parser.add_argument(
        "--entry-id",
        help="Show details for specific entry ID",
    )

    parser.add_argument(
        "--delete-entry",
        metavar="ENTRY_ID",
        help="Delete a specific entry by ID",
    )

    parser.add_argument(
        "--delete-dataset",
        metavar="DATASET",
        help="Delete all entries for a specific dataset",
    )

    parser.add_argument(
        "--delete-branch",
        metavar="BRANCH",
        help="Delete all entries from a specific branch",
    )

    parser.add_argument(
        "--clear-all",
        action="store_true",
        help="Clear all entries from the registry (use with caution!)",
    )

    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Confirm destructive operations (required for --clear-all)",
    )

    args = parser.parse_args()

    # Initialize registry
    registry = DataConfigRegistry(registry_path=args.registry_path)

    # Handle delete operations
    if args.delete_entry:
        success = registry.delete_entry(args.delete_entry)
        if success:
            print(f"Deleted entry: {args.delete_entry}")
        else:
            print(f"Entry ID '{args.delete_entry}' not found")
        return

    if args.delete_dataset:
        count = registry.delete_entries_by_dataset(args.delete_dataset)
        print(f"Deleted {count} entries for dataset '{args.delete_dataset}'")
        return

    if args.delete_branch:
        count = registry.delete_entries_by_branch(args.delete_branch)
        print(f"Deleted {count} entries from branch '{args.delete_branch}'")
        return

    if args.clear_all:
        if args.confirm:
            success = registry.clear_registry(confirm=True)
            if success:
                print("Registry cleared successfully")
        else:
            print("ERROR: --clear-all requires --confirm flag for safety")
            print("Usage: python view_data_config_registry.py --clear-all --confirm")
        return

    if args.export:
        # Export to CSV
        registry.export_to_csv(args.export)
        print(f"Registry exported to: {args.export}")
        return

    if args.entry_id:
        # Show specific entry
        entry = registry.get_entry(args.entry_id)
        if entry:
            print(json.dumps(entry, indent=2))
        else:
            print(f"Entry ID '{args.entry_id}' not found")
        return

    if args.dataset:
        # Filter by dataset
        entries = registry.get_entries_by_dataset(args.dataset)
        print(f"Found {len(entries)} entries for dataset '{args.dataset}'")
        for entry in entries:
            system_info = entry.get("system", {})
            print(f"\nEntry ID: {entry['entry_id']}")
            print(f"Timestamp: {system_info.get('timestamp', 'N/A')}")
            print(f"User: {system_info.get('user', 'N/A')}")
            print(f"Branch: {entry['git'].get('branch', 'N/A')}")
            print(f"Datasets: {', '.join(entry['config']['datasets'])}")
        return

    if args.branch:
        # Filter by branch
        entries = registry.get_entries_by_branch(args.branch)
        print(f"Found {len(entries)} entries for branch '{args.branch}'")
        for entry in entries:
            system_info = entry.get("system", {})
            print(f"\nEntry ID: {entry['entry_id']}")
            print(f"Timestamp: {system_info.get('timestamp', 'N/A')}")
            print(f"User: {system_info.get('user', 'N/A')}")
            print(f"Datasets: {', '.join(entry['config']['datasets'])}")
        return

    if args.recent:
        # Show recent entries
        entries = registry.get_recent_entries(args.recent)
        print(f"Most recent {len(entries)} entries:\n")
        for entry in entries:
            system_info = entry.get("system", {})
            print(f"Entry ID: {entry['entry_id']}")
            print(f"Timestamp: {system_info.get('timestamp', 'N/A')}")
            print(f"User: {system_info.get('user', 'N/A')}")
            print(f"Branch: {entry['git'].get('branch', 'N/A')}")
            print(f"Datasets: {', '.join(entry['config']['datasets'])}")
            print(f"Split Type: {entry['config']['split_type']}")
            print(f"Generated Files: {len(entry['outputs']['generated_files'])}")
            print("-" * 60)
        return

    # Default: show summary
    print(registry.get_summary())


if __name__ == "__main__":
    main()
