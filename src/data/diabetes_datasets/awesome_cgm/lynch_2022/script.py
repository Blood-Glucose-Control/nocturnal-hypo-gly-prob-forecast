import sys
import os
import logging
from pathlib import Path

script_path = Path(__file__).resolve()
# Go up 6 levels to reach the project root
project_root = script_path.parent.parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO)

try:
    from src.data.diabetes_datasets.data_loader import get_loader
except ImportError as e:
    print(f"Import failed: {e}")
    print(f"project_root used: {project_root}")
    raise

def _expected_sas_dir() -> Path:
    # Expected by lynch_2022.py; cache path is namespaced under awesome_cgm/lynch_2022
    return project_root / "cache" / "data" / "awesome_cgm" / "lynch_2022" / "raw" / "IOBP2 RCT Public Dataset" / "Data Tables in SAS"

def main():
    # Optional quick setup check:
    if "--check-only" in sys.argv:
        sas_dir = _expected_sas_dir()
        if sas_dir.exists():
            print(f"✓ Found SAS tables directory:\n  {sas_dir}")
            sys.exit(0)
        else:
            print("✗ Missing Lynch 2022 SAS tables directory.")
            print(f"Please place the SAS .sas7bdat files at:\n  {sas_dir}")
            print("The folder should contain the SAS tables (e.g., CGM, demographics, etc.).")
            sys.exit(2)

    # Preflight check before invoking the loader to avoid a long stack trace
    sas_dir = _expected_sas_dir()
    if not sas_dir.exists():
        print("✗ Missing Lynch 2022 SAS tables directory.")
        print(f"Please place the SAS .sas7bdat files at:\n  {sas_dir}")
        print("Tip: run this script with '--check-only' to re-validate after placing the files.")
        return

    try:
        print("Creating Lynch2022DataLoader...")
        loader = get_loader(
            data_source_name="lynch_2022",
            dataset_type="train",
            use_cached=True,
            num_validation_days=20,
        )
        print("✓ Created")
        print(f"Dataset name: {loader.dataset_name}")
        print(f"Num patients: {loader.num_patients}")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback; traceback.print_exc()

if __name__ == "__main__":
    main()