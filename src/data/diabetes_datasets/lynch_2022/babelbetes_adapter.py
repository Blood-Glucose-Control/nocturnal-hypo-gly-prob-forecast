# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)
# For commercial licensing, contact: christopher/cjrisi AT gluroo/uwaterloo DOT com/ca

"""
BabelBetes-compatible loader for the Lynch 2022 / IOBP2 RCT dataset.

Replicates the IOBP2 class logic from BabelBetes v0.2.3 (nudgebg/babelbetes)
as a standalone implementation, reading from the pipe-separated "Data Tables"
txt files rather than SAS files. This allows cross-validation of our pipeline
against the BabelBetes reference implementation without requiring the babelbetes
package (which has transitive deps incompatible with this Linux environment).

Reference: https://github.com/nudgebg/babelbetes/blob/main/babelbetes/studies/iobp2.py

Key decisions replicated from BabelBetes:
  - CGM source: CGMVal column in IOBP2DeviceiLet.txt
  - Sentinel clamping: CGMVal <= 39 → 40, CGMVal >= 401 → 400 (mg/dL)
  - Deduplication: drop duplicate (PtID, DeviceDtTm) CGM rows
  - Insulin: BasalDelivPrev + BolusDelivPrev + MealBolusDelivPrev
  - Timestamp shift: insulin delivery timestamps shifted -5 min (previous-step semantics)
  - Units: CGM returned in mg/dL (caller converts to mmol/L)
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_ILET_FILENAME = "IOBP2DeviceiLet.txt"
_DEMO_FILENAME = "IOBP2DiabScreening.txt"

# mg/dL sentinel values used by the iLet device
_CGM_SENTINEL_LOW = 39    # sensor below range → clamp to 40
_CGM_SENTINEL_HIGH = 401  # sensor above range → clamp to 400


class IOBP2Adapter:
    """
    BabelBetes-compatible adapter for IOBP2 / Lynch 2022 data.

    Reads from pipe-separated txt files and exposes the same interface
    as the BabelBetes IOBP2 class: extract_cgm_history(),
    extract_bolus_event_history(), extract_basal_event_history().

    Args:
        study_path: Path containing the "Data Tables" directory with txt files.

    Example:
        >>> adapter = IOBP2Adapter("/path/to/IOBP2 RCT Public Dataset")
        >>> cgm = adapter.extract_cgm_history()  # per-patient CGM DataFrames
        >>> insulin = adapter.extract_insulin_event_history()  # per-patient insulin
    """

    def __init__(self, study_path: Path | str):
        self.study_path = Path(study_path)
        self._tables_dir = self.study_path / "Data Tables"
        if not self._tables_dir.exists():
            raise FileNotFoundError(
                f"Expected 'Data Tables' directory at {self._tables_dir}"
            )
        self._ilet: pd.DataFrame | None = None
        self._demo: pd.DataFrame | None = None

    def _load_ilet(self) -> pd.DataFrame:
        if self._ilet is None:
            path = self._tables_dir / _ILET_FILENAME
            self._ilet = pd.read_csv(path, sep="|", low_memory=False)
            self._ilet["DeviceDtTm"] = pd.to_datetime(
                self._ilet["DeviceDtTm"], errors="coerce"
            )
            self._ilet["CGMVal"] = pd.to_numeric(
                self._ilet["CGMVal"], errors="coerce"
            )
            logger.info("Loaded %s with shape %s", _ILET_FILENAME, self._ilet.shape)
        return self._ilet

    def _load_demo(self) -> pd.DataFrame:
        if self._demo is None:
            path = self._tables_dir / _DEMO_FILENAME
            self._demo = pd.read_csv(path, sep="|", low_memory=False)
            logger.info("Loaded %s with shape %s", _DEMO_FILENAME, self._demo.shape)
        return self._demo

    def extract_cgm_history(self) -> dict[str, pd.DataFrame]:
        """
        Extract per-patient CGM history in mg/dL.

        Replicates BabelBetes IOBP2.extract_cgm_history():
          - Filters rows with valid CGMVal
          - Clamps sensor-low (≤39) to 40 and sensor-high (≥401) to 400
          - Deduplicates (PtID, DeviceDtTm)

        Returns:
            Dict mapping patient_id → DataFrame with columns [datetime, cgm_mgdl]
            where datetime is the CGM observation time (NOT shifted).
        """
        ilet = self._load_ilet()
        cgm = ilet.dropna(subset=["CGMVal"]).copy()

        # Sentinel clamping
        cgm.loc[cgm["CGMVal"] <= _CGM_SENTINEL_LOW, "CGMVal"] = 40.0
        cgm.loc[cgm["CGMVal"] >= _CGM_SENTINEL_HIGH, "CGMVal"] = 400.0

        # Deduplication
        cgm = cgm.drop_duplicates(subset=["PtID", "DeviceDtTm"])
        cgm = cgm[["PtID", "DeviceDtTm", "CGMVal"]].rename(
            columns={"DeviceDtTm": "datetime", "CGMVal": "cgm_mgdl"}
        )

        result: dict[str, pd.DataFrame] = {}
        for pt_id, group in cgm.groupby("PtID"):
            result[str(int(pt_id))] = (
                group[["datetime", "cgm_mgdl"]]
                .sort_values("datetime")
                .reset_index(drop=True)
            )
        logger.info("Extracted CGM history for %d patients", len(result))
        return result

    def extract_insulin_event_history(self) -> dict[str, pd.DataFrame]:
        """
        Extract per-patient insulin delivery history with -5 min timestamp shift.

        Replicates BabelBetes IOBP2 insulin extraction:
          - dose = BasalDelivPrev + BolusDelivPrev + MealBolusDelivPrev
          - timestamp shifted -5 min (delivery of the *previous* 5-min step)

        Returns:
            Dict mapping patient_id → DataFrame with columns [datetime, dose_units]
            where datetime is the delivery start time (original timestamp − 5 min).
        """
        ilet = self._load_ilet()
        ins = ilet[["PtID", "DeviceDtTm"]].copy()

        for col in ["BasalDelivPrev", "BolusDelivPrev", "MealBolusDelivPrev"]:
            if col in ilet.columns:
                ins[col] = pd.to_numeric(ilet[col], errors="coerce").fillna(0.0)
            else:
                ins[col] = 0.0

        ins["dose_units"] = (
            ins["BasalDelivPrev"] + ins["BolusDelivPrev"] + ins["MealBolusDelivPrev"]
        )
        # Shift back 5 min: delivery was for [T-5, T)
        ins["datetime"] = ins["DeviceDtTm"] - pd.Timedelta(minutes=5)

        result: dict[str, pd.DataFrame] = {}
        for pt_id, group in ins.groupby("PtID"):
            result[str(int(pt_id))] = (
                group[["datetime", "dose_units"]]
                .sort_values("datetime")
                .reset_index(drop=True)
            )
        logger.info("Extracted insulin history for %d patients", len(result))
        return result

    def patient_ids(self) -> list[str]:
        """Return sorted list of all patient IDs in the dataset."""
        ilet = self._load_ilet()
        return sorted(str(int(pid)) for pid in ilet["PtID"].dropna().unique())
