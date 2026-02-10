# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)
# For commercial licensing, contact: christopher/cjrisi AT gluroo/uwaterloo DOT com/ca

"""
Data Configuration Registry System for Holdout Config Generation

Maintains a JSON registry of all data configuration generation runs with:
- Git metadata (branch, commit, clean status)
- System metadata (user, timestamp)
- Configuration parameters used
- Generated output files
"""

import fcntl
import getpass
import json
import logging
import os
import subprocess
import tempfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

logger = logging.getLogger(__name__)


class DataConfigRegistry:
    """Registry for tracking data configuration generation events."""

    def __init__(self, registry_path: str = "configs/data/data_config_registry.json"):
        """Initialize data config registry with JSON file path.

        Args:
            registry_path: Path to the registry JSON file
        """
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_path = self.registry_path.with_suffix(".lock")
        self._initialize_registry()

    @contextmanager
    def _file_lock(self) -> Generator[None, None, None]:
        """Context manager for file-based locking to prevent concurrent access.

        Uses fcntl.flock for advisory locking on Unix systems.
        """
        lock_file = open(self._lock_path, "w")
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            lock_file.close()

    def _atomic_write_json(self, data: Dict[str, Any]) -> None:
        """Atomically write JSON data to the registry file.

        Writes to a temporary file first, then renames to the target path.
        This ensures the registry is never left in a partially-written state.

        Args:
            data: Dictionary to write as JSON
        """
        # Create temp file in the same directory to ensure same filesystem for atomic rename
        dir_path = self.registry_path.parent
        fd, temp_path = tempfile.mkstemp(suffix=".tmp", dir=dir_path)
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2)
            # Atomic rename (on POSIX systems)
            os.replace(temp_path, self.registry_path)
        except Exception:
            # Clean up temp file on failure
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise

    def _read_registry(self) -> Dict[str, Any]:
        """Read the registry JSON file.

        Returns:
            Dictionary containing registry data
        """
        with open(self.registry_path, "r") as f:
            return json.load(f)

    def _initialize_registry(self):
        """Create registry JSON file if it doesn't exist."""
        if not self.registry_path.exists():
            initial_data = {
                "schema_version": "1.0.0",
                "description": "Registry of data configuration generation runs",
                "entries": [],
            }
            with self._file_lock():
                # Double-check after acquiring lock (another process may have created it)
                if not self.registry_path.exists():
                    self._atomic_write_json(initial_data)
                    logger.info(f"Initialized new registry at {self.registry_path}")

    def _get_git_info(self) -> Dict[str, Any]:
        """Get git repository information.

        Returns:
            Dictionary containing git metadata
        """
        git_info = {
            "branch": None,
            "commit_hash": None,
            "commit_short_hash": None,
            "is_clean": None,
            "uncommitted_changes": None,
            "remote_url": None,
        }

        try:
            # Get current branch
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            )
            git_info["branch"] = result.stdout.strip()

            # Get commit hash
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            )
            git_info["commit_hash"] = result.stdout.strip()

            # Get short commit hash
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            )
            git_info["commit_short_hash"] = result.stdout.strip()

            # Check if working directory is clean
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=True,
            )
            is_clean = len(result.stdout.strip()) == 0
            git_info["is_clean"] = is_clean
            git_info["uncommitted_changes"] = (
                None if is_clean else result.stdout.strip().split("\n")
            )

            # Get remote URL
            result = subprocess.run(
                ["git", "config", "--get", "remote.origin.url"],
                capture_output=True,
                text=True,
                check=True,
            )
            git_info["remote_url"] = result.stdout.strip()

        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to get git info: {e}")
        except FileNotFoundError:
            logger.warning("Git not found in system PATH")

        return git_info

    def _get_system_info(self) -> Dict[str, str]:
        """Get system and user information.

        Returns:
            Dictionary containing system metadata
        """
        system_info = {
            "user": getpass.getuser(),
            "timestamp": datetime.now().isoformat(),
            "timestamp_utc": datetime.utcnow().isoformat(),
        }

        try:
            # Try to get hostname
            result = subprocess.run(
                ["hostname"], capture_output=True, text=True, check=True
            )
            system_info["hostname"] = result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            system_info["hostname"] = None

        return system_info

    def register_config_generation(
        self,
        datasets: List[str],
        split_type: str,
        temporal_pct: float,
        patient_pct: float,
        seed: int,
        output_dir: Path,
        min_train_samples: int,
        min_holdout_samples: int,
        min_train_patients: int,
        min_holdout_patients: int,
        generated_files: Optional[List[str]] = None,
        additional_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Register a data configuration generation event.

        Args:
            datasets: List of dataset names processed
            split_type: Type of split (hybrid, temporal, patient)
            temporal_pct: Temporal holdout percentage
            patient_pct: Patient holdout percentage
            seed: Random seed used
            output_dir: Output directory for configs
            min_train_samples: Minimum training samples required
            min_holdout_samples: Minimum holdout samples required
            min_train_patients: Minimum training patients required
            min_holdout_patients: Minimum holdout patients required
            generated_files: List of generated config file paths (optional)
            additional_metadata: Any additional metadata to store (optional)

        Returns:
            Unique entry ID for this registration
        """
        # Generate unique entry ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        entry_id = f"config_gen_{timestamp}"

        # Gather all metadata
        entry = {
            "entry_id": entry_id,
            "git": self._get_git_info(),
            "system": self._get_system_info(),
            "config": {
                "datasets": datasets,
                "split_type": split_type,
                "temporal_holdout_percentage": temporal_pct,
                "patient_holdout_percentage": patient_pct,
                "random_seed": seed,
                "output_directory": str(output_dir),
                "constraints": {
                    "min_train_samples": min_train_samples,
                    "min_holdout_samples": min_holdout_samples,
                    "min_train_patients": min_train_patients,
                    "min_holdout_patients": min_holdout_patients,
                },
            },
            "outputs": {
                "generated_files": generated_files or [],
                "output_directory": str(output_dir),
            },
        }

        # Add any additional metadata
        if additional_metadata:
            entry["additional_metadata"] = additional_metadata

        # Atomically update registry with file locking
        with self._file_lock():
            registry_data = self._read_registry()
            registry_data["entries"].append(entry)
            self._atomic_write_json(registry_data)

        logger.info(f"Registered config generation with ID: {entry_id}")
        logger.info(f"Git branch: {entry['git']['branch']}")
        logger.info(f"Git clean: {entry['git']['is_clean']}")
        logger.info(f"User: {entry['system']['user']}")

        return entry_id

    def get_entry(self, entry_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific registry entry by ID.

        Args:
            entry_id: The entry ID to retrieve

        Returns:
            Dictionary containing entry data or None if not found
        """
        with self._file_lock():
            registry_data = self._read_registry()

        for entry in registry_data["entries"]:
            if entry["entry_id"] == entry_id:
                return entry

        return None

    def get_recent_entries(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get the most recent N registry entries.

        Args:
            n: Number of recent entries to retrieve

        Returns:
            List of entry dictionaries
        """
        with self._file_lock():
            registry_data = self._read_registry()

        entries = registry_data["entries"]
        return entries[-n:] if len(entries) > n else entries

    def get_entries_by_dataset(self, dataset_name: str) -> List[Dict[str, Any]]:
        """Get all registry entries for a specific dataset.

        Args:
            dataset_name: Name of the dataset to filter by

        Returns:
            List of entry dictionaries containing the dataset
        """
        with self._file_lock():
            registry_data = self._read_registry()

        matching_entries = []
        for entry in registry_data["entries"]:
            if dataset_name in entry["config"]["datasets"]:
                matching_entries.append(entry)

        return matching_entries

    def get_entries_by_branch(self, branch_name: str) -> List[Dict[str, Any]]:
        """Get all registry entries for a specific git branch.

        Args:
            branch_name: Name of the git branch to filter by

        Returns:
            List of entry dictionaries from the specified branch
        """
        with self._file_lock():
            registry_data = self._read_registry()

        matching_entries = []
        for entry in registry_data["entries"]:
            if entry["git"]["branch"] == branch_name:
                matching_entries.append(entry)

        return matching_entries

    def get_summary(self) -> str:
        """Get a summary of the registry contents.

        Returns:
            Formatted string with registry summary
        """
        with self._file_lock():
            registry_data = self._read_registry()

        entries = registry_data["entries"]
        total_entries = len(entries)

        if total_entries == 0:
            return "Registry is empty"

        # Collect statistics
        datasets = set()
        branches = set()
        users = set()

        for entry in entries:
            datasets.update(entry["config"]["datasets"])
            if entry["git"]["branch"]:
                branches.add(entry["git"]["branch"])
            users.add(entry["system"]["user"])

        summary = f"""
Data Config Registry Summary
=============================
Total entries: {total_entries}
Unique datasets: {len(datasets)}
Git branches used: {len(branches)}
Users: {len(users)}

Datasets: {', '.join(sorted(datasets))}
Recent branches: {', '.join(sorted(branches))}
        """.strip()

        return summary

    def export_to_csv(self, output_path: str):
        """Export registry to a flattened CSV format.

        Args:
            output_path: Path for the output CSV file
        """
        import pandas as pd

        with self._file_lock():
            registry_data = self._read_registry()

        # Flatten entries for CSV export
        rows = []
        for entry in registry_data["entries"]:
            row = {
                "entry_id": entry["entry_id"],
                "timestamp": entry["system"]["timestamp"],
                "user": entry["system"]["user"],
                "git_branch": entry["git"]["branch"],
                "git_commit": entry["git"]["commit_short_hash"],
                "git_clean": entry["git"]["is_clean"],
                "datasets": ",".join(entry["config"]["datasets"]),
                "split_type": entry["config"]["split_type"],
                "temporal_pct": entry["config"]["temporal_holdout_percentage"],
                "patient_pct": entry["config"]["patient_holdout_percentage"],
                "seed": entry["config"]["random_seed"],
                "output_dir": entry["config"]["output_directory"],
                "num_generated_files": len(entry["outputs"]["generated_files"]),
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        logger.info(f"Exported registry to CSV: {output_path}")

    def delete_entry(self, entry_id: str) -> bool:
        """Delete a specific registry entry by ID.

        Args:
            entry_id: The entry ID to delete

        Returns:
            True if entry was deleted, False if not found
        """
        with self._file_lock():
            registry_data = self._read_registry()

            initial_count = len(registry_data["entries"])
            registry_data["entries"] = [
                entry
                for entry in registry_data["entries"]
                if entry["entry_id"] != entry_id
            ]

            if len(registry_data["entries"]) < initial_count:
                self._atomic_write_json(registry_data)
                logger.info(f"Deleted entry: {entry_id}")
                return True
            else:
                logger.warning(f"Entry ID '{entry_id}' not found")
                return False

    def delete_entries_by_dataset(self, dataset_name: str) -> int:
        """Delete all registry entries for a specific dataset.

        Args:
            dataset_name: Name of the dataset to filter by

        Returns:
            Number of entries deleted
        """
        with self._file_lock():
            registry_data = self._read_registry()

            initial_count = len(registry_data["entries"])
            registry_data["entries"] = [
                entry
                for entry in registry_data["entries"]
                if dataset_name not in entry["config"]["datasets"]
            ]

            deleted_count = initial_count - len(registry_data["entries"])
            if deleted_count > 0:
                self._atomic_write_json(registry_data)
                logger.info(
                    f"Deleted {deleted_count} entries for dataset '{dataset_name}'"
                )
            else:
                logger.info(f"No entries found for dataset '{dataset_name}'")

        return deleted_count

    def delete_entries_by_branch(self, branch_name: str) -> int:
        """Delete all registry entries from a specific git branch.

        Args:
            branch_name: Name of the git branch to filter by

        Returns:
            Number of entries deleted
        """
        with self._file_lock():
            registry_data = self._read_registry()

            initial_count = len(registry_data["entries"])
            registry_data["entries"] = [
                entry
                for entry in registry_data["entries"]
                if entry["git"]["branch"] != branch_name
            ]

            deleted_count = initial_count - len(registry_data["entries"])
            if deleted_count > 0:
                self._atomic_write_json(registry_data)
                logger.info(
                    f"Deleted {deleted_count} entries from branch '{branch_name}'"
                )
            else:
                logger.info(f"No entries found for branch '{branch_name}'")

        return deleted_count

    def clear_registry(self, confirm: bool = False) -> bool:
        """Clear all entries from the registry.

        Args:
            confirm: Must be True to actually clear the registry (safety check)

        Returns:
            True if registry was cleared, False if not confirmed
        """
        if not confirm:
            logger.warning(
                "Registry clear operation requires confirm=True parameter. "
                "This will delete all entries permanently."
            )
            return False

        with self._file_lock():
            registry_data = self._read_registry()
            entry_count = len(registry_data["entries"])
            registry_data["entries"] = []
            self._atomic_write_json(registry_data)

        logger.info(f"Cleared registry - deleted {entry_count} entries")
        return True
