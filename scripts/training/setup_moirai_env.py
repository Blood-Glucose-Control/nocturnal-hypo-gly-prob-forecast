#!/usr/bin/env python
"""
Cross-platform script to set up the Moirai model environment.

Usage:
    python scripts/setup_moirai_env.py

Creates a virtual environment at .venvs/moirai and installs moirai dependencies.
"""

import os
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd, description):
    """Run command and print output."""
    print(f"\n{'='*70}")
    print(f"  {description}")
    print(f"{'='*70}")
    print(f"  Command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"\n❌ Failed: {description}")
        sys.exit(1)
    print(f"✅ Done: {description}")


def main():
    repo_root = Path(__file__).parent.parent
    venvs_dir = repo_root / ".venvs"
    venv_path = venvs_dir / "moirai"

    print(f"\n{'='*70}")
    print("  MOIRAI ENVIRONMENT SETUP")
    print(f"{'='*70}")
    print(f"  Repo root: {repo_root}")
    print(f"  Venv path: {venv_path}")
    print(f"  Python: {sys.executable}")

    # Create venvs directory
    venvs_dir.mkdir(exist_ok=True)

    # Create venv if it doesn't exist
    if not venv_path.exists():
        print("\n  Creating virtual environment...")
        run_cmd(
            [sys.executable, "-m", "venv", str(venv_path)],
            "Create virtual environment"
        )
    else:
        print(f"\n  Virtual environment already exists at {venv_path}")

    # Determine python path for the venv
    if os.name == "nt":  # Windows
        python_exe = venv_path / "Scripts" / "python.exe"
    else:  # Unix/Linux/macOS
        python_exe = venv_path / "bin" / "python"

    # Upgrade pip using python -m (works better on Windows)
    run_cmd(
        [str(python_exe), "-m", "pip", "install", "--upgrade", "pip"],
        "Upgrade pip"
    )

    # Install project with moirai dependencies
    print("\n  Installing project with [moirai] dependencies...")
    run_cmd(
        [str(python_exe), "-m", "pip", "install", "-e", f"{repo_root}[moirai]"],
        "Install project with moirai dependencies"
    )

    # Check installation
    print(f"\n{'='*70}")
    print("  VERIFYING INSTALLATION")
    print(f"{'='*70}")

    run_cmd(
        [str(python_exe), "-c", "from uni2ts.model.moirai import MoiraiModule; print('✅ uni2ts/Moirai imported successfully')"],
        "Verify uni2ts import"
    )

    # Print activation instructions
    if os.name == "nt":  # Windows
        activate_cmd = f"{venv_path}\\Scripts\\Activate.ps1"
        print(f"\n{'='*70}")
        print("  SETUP COMPLETE!")
        print(f"{'='*70}")
        print(f"\n  To activate the environment on Windows PowerShell:")
        print(f"    .venvs\\moirai\\Scripts\\Activate.ps1")
        print(f"\n  Or on Windows CMD:")
        print(f"    .venvs\\moirai\\Scripts\\activate.bat")
    else:
        activate_cmd = f"source {venv_path}/bin/activate"
        print(f"\n{'='*70}")
        print("  SETUP COMPLETE!")
        print(f"{'='*70}")
        print(f"\n  To activate the environment:")
        print(f"    {activate_cmd}")

    print(f"\n  Python: {python_exe}")
    print(f"  To verify setup, run:")
    print(f"    {activate_cmd}")
    print(f"    python -c 'from uni2ts.model.moirai import MoiraiModule'")
    print(f"\n  To test Moirai training:")
    print(f"    {activate_cmd}")
    print(f"    python scripts/examples/train_moirai_quick_test.py")
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
