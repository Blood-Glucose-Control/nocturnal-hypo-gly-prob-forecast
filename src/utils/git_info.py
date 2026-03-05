# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)
# For commercial licensing, contact: christopher/cjrisi AT gluroo/uwaterloo DOT com/ca

"""Git information utilities for reproducibility tracking."""

import subprocess


def get_git_commit_hash() -> str:
    """Get the current git commit hash.

    Useful for tracking which version of code was used for experiments
    and ensuring reproducibility.

    Returns:
        Short git commit hash (7 characters), or 'unknown' if not in a git repository
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def get_git_branch() -> str:
    """Get the current git branch name.

    Returns:
        Current branch name, or 'unknown' if not in a git repository
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def is_git_dirty() -> bool:
    """Check if there are uncommitted changes in the git repository.

    Returns:
        True if there are uncommitted changes, False otherwise.
        Returns False if not in a git repository.
    """
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        )
        return bool(result.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
