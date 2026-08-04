"""
Regenerate Tamborlane 2008 processed data after fixing the datetime bug.

This script reprocesses the Tamborlane dataset to use the correct DeviceDtTm
timestamps instead of artificially aligning all patients to midnight.
"""

import logging
import sys
from pathlib import Path

# Add src to path
repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root / "src"))

from src.data.diabetes_datasets.tamborlane_2008.tamborlane_2008 import (  # noqa: E402
    Tamborlane2008DataLoader,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def main():
    """Regenerate Tamborlane processed data."""
    logger.info("=" * 80)
    logger.info("Regenerating Tamborlane 2008 processed data")
    logger.info("=" * 80)

    # Load with use_cached=False to force reprocessing
    loader = Tamborlane2008DataLoader(
        use_cached=False,  # Force reprocessing
        parallel=True,  # Use parallel processing for speed
        max_workers=8,
        extract_features=True,
    )

    # Access processed_data to trigger processing
    processed_data = loader.processed_data

    logger.info(f"\nSuccessfully processed {len(processed_data)} patients")

    # Verify a sample patient has correct time-of-day variation
    sample_patient = list(processed_data.keys())[0]
    sample_df = processed_data[sample_patient]

    logger.info(f"\nSample patient {sample_patient}:")
    logger.info(
        f"  Date range: {sample_df['datetime'].min()} to {sample_df['datetime'].max()}"
    )
    logger.info("  First 5 timestamps:")
    for ts in sample_df["datetime"].head(5):
        logger.info(f"    {ts}")

    logger.info("\n" + "=" * 80)
    logger.info("Tamborlane data regeneration complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
