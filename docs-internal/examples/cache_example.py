"""Example script demonstrating centralized cache usage."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.data.cache_manager import get_cache_manager
from src.data.diabetes_datasets.data_loader import get_loader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_PREVIEW_FILES = 3
MAX_PREVIEW_SUBDIRS = 3


def _iter_sorted_dirs(path: Path) -> list[Path]:
    return sorted(
        (entry for entry in path.iterdir() if entry.is_dir()), key=lambda p: p.name
    )


def _iter_sorted_files(path: Path) -> list[Path]:
    return sorted(
        (entry for entry in path.iterdir() if entry.is_file()), key=lambda p: p.name
    )


def _find_child_dir(parent: Path, child_name: str) -> Path | None:
    expected_path = parent / child_name
    if expected_path.is_dir():
        return expected_path

    lower_child_name = child_name.lower()
    for entry in _iter_sorted_dirs(parent):
        if entry.name.lower() == lower_child_name:
            return entry
    return None


def _print_patient_preview(processed_data: dict[str, pd.DataFrame]) -> None:
    first_patient_id = sorted(processed_data.keys())[0]
    first_patient_df = processed_data[first_patient_id]

    print(f"   ✓ Example patient: {first_patient_id}")
    print(f"   ✓ Patient frame shape: {first_patient_df.shape}")
    print("   ✓ Head:")
    print(first_patient_df.head(3).to_string())


def _print_directory_summary(dir_path: Path, indent: str = "      ") -> None:
    files = _iter_sorted_files(dir_path)
    subdirs = _iter_sorted_dirs(dir_path)
    print(f"{indent}📁 {dir_path.name}/ ({len(subdirs)} dirs, {len(files)} files)")

    for file_path in files[:MAX_PREVIEW_FILES]:
        print(f"{indent}   📄 {file_path.name}")
    if len(files) > MAX_PREVIEW_FILES:
        print(f"{indent}   ... {len(files) - MAX_PREVIEW_FILES} more files")

    for subdir in subdirs[:MAX_PREVIEW_SUBDIRS]:
        nested_file_count = len(_iter_sorted_files(subdir))
        nested_dir_count = len(_iter_sorted_dirs(subdir))
        print(
            f"{indent}   📁 {subdir.name}/ ({nested_dir_count} dirs, {nested_file_count} files)"
        )
    if len(subdirs) > MAX_PREVIEW_SUBDIRS:
        print(f"{indent}   ... {len(subdirs) - MAX_PREVIEW_SUBDIRS} more directories")


def _show_cache_structure(cache_root: Path) -> None:
    print("2. Cache directory structure (readable summary):")
    if not cache_root.is_dir():
        print("   No cache directory found yet")
        return

    dataset_dirs = _iter_sorted_dirs(cache_root)
    if not dataset_dirs:
        print("   Cache directory exists but has no dataset folders yet")
        return

    for dataset_dir in dataset_dirs:
        print(f"   📁 {dataset_dir.name}/")

        raw_dir = _find_child_dir(dataset_dir, "raw")
        if raw_dir is not None:
            _print_directory_summary(raw_dir, indent="      ")
        else:
            print("      raw/ (missing)")

        processed_dir = _find_child_dir(dataset_dir, "processed")
        if processed_dir is not None:
            _print_directory_summary(processed_dir, indent="      ")
        else:
            print("      processed/ (missing)")


def main() -> None:
    print("=== Centralized Cache System Demo ===\n")

    cache_manager = get_cache_manager()
    cache_root = Path(cache_manager.cache_root)
    print(f"Cache root directory: {cache_root}\n")

    print("1. Loading Aleppo T1D dataset...")
    loader = get_loader(
        data_source_name="aleppo_2017",
        use_cached=True,
    )
    processed_data = loader.processed_data
    if not isinstance(processed_data, dict) or not processed_data:
        raise ValueError(
            "Expected non-empty dict[str, DataFrame] in loader.processed_data."
        )

    print("   ✓ Dataset loaded successfully!")
    print(f"   ✓ Dataset name: {loader.dataset_name}")
    print(f"   ✓ Number of patients: {len(processed_data)}")
    _print_patient_preview(processed_data)
    print()

    _show_cache_structure(cache_root)
    print()

    print("3. Cache management options:")
    print("   - Clear specific dataset: cache_manager.clear_cache('aleppo_2017')")
    print("   - Clear all cache: cache_manager.clear_cache()")
    print("   - Check cache info: cache_manager.get_dataset_cache_path('aleppo_2017')")
    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    main()
