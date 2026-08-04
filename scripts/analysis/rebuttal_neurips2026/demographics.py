# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)
"""Per-patient demographics loader for the subgroup analysis (A5).

IMPORTANT (reviewer-flagged): use the patient's ACTUAL age at enrollment /
randomization, NOT `DiagAge` (age at T1D diagnosis). Both are available; we also
derive diabetes duration = age_enroll - diag_age where diag age exists.

True-age / sex source columns per dataset (JAEB public releases):
    aleppo_2017 (Replace-BG): HPtRoster.AgeAsOfEnrollDt + HScreening.Gender/DiagAge
    brown_2019  (DCLP3, UTF-16): DiabScreening_a.AgeAtEnrollment/Gender/DiagAge
    lynch_2022  (IOBP2): IOBP2PtRoster.AgeAsofEnrollDt + IOBP2DiabScreening.Sex/DiagAge
    tamborlane_2008 (JDRF): tblAPtSummary.AgeAsOfRandDt/Gender  (no DiagAge)

Join key: episode pid is "<prefix>_<PtID>" (ale_/bro_/lyn_/tam_); we build the
same pid from PtID so demographics merge onto the per-episode frames.
"""

from __future__ import annotations

import functools
from pathlib import Path

import pandas as pd

RAW = Path("/data/shared/cache/data")

_PREFIX = {
    "aleppo_2017": "ale",
    "brown_2019": "bro",
    "lynch_2022": "lyn",
    "tamborlane_2008": "tam",
}


def _read(path: Path, sep: str = "|") -> pd.DataFrame:
    """Read a JAEB table, trying common encodings (Brown screening is UTF-16)."""
    last: Exception | None = None
    for enc in ("utf-8", "utf-16", "latin-1"):
        try:
            return pd.read_csv(path, sep=sep, encoding=enc)
        except Exception as e:  # noqa: BLE001
            last = e
    raise last  # type: ignore[misc]


def _norm_sex(s: pd.Series) -> pd.Series:
    m = s.astype(str).str.strip().str.upper().str[0]
    return m.where(m.isin(["M", "F"]))


@functools.lru_cache(maxsize=None)
def load_demographics(dataset: str) -> pd.DataFrame:
    """Return per-patient demographics: pid, ptid, sex (M/F), age, diag_age, duration."""
    pfx = _PREFIX[dataset]
    if dataset == "aleppo_2017":
        base = RAW / "aleppo_2017/raw/Data Tables"
        roster = _read(base / "HPtRoster.txt")[["PtID", "AgeAsOfEnrollDt"]]
        scr = _read(base / "HScreening.txt")[["PtID", "Gender", "DiagAge"]]
        df = roster.merge(scr, on="PtID", how="left")
        df = df.rename(
            columns={"AgeAsOfEnrollDt": "age", "Gender": "sex", "DiagAge": "diag_age"}
        )
    elif dataset == "brown_2019":
        base = (
            RAW
            / "brown_2019/raw/DCLP3 Public Dataset - Release 3 - 2022-08-04/Data Files"
        )
        df = _read(base / "DiabScreening_a.txt")[
            ["PtID", "AgeAtEnrollment", "Gender", "DiagAge"]
        ]
        df = df.rename(
            columns={"AgeAtEnrollment": "age", "Gender": "sex", "DiagAge": "diag_age"}
        )
    elif dataset == "lynch_2022":
        base = RAW / "lynch_2022/raw/IOBP2 RCT Public Dataset/Data Tables"
        roster = _read(base / "IOBP2PtRoster.txt")[["PtID", "AgeAsofEnrollDt"]]
        scr = _read(base / "IOBP2DiabScreening.txt")[["PtID", "Sex", "DiagAge"]]
        df = roster.merge(scr, on="PtID", how="left")
        df = df.rename(
            columns={"AgeAsofEnrollDt": "age", "Sex": "sex", "DiagAge": "diag_age"}
        )
    elif dataset == "tamborlane_2008":
        base = RAW / "tamborlane_2008/raw"
        df = _read(base / "tblAPtSummary.csv", sep=",")[
            ["PtID", "Gender", "AgeAsOfRandDt"]
        ]
        df = df.rename(columns={"AgeAsOfRandDt": "age", "Gender": "sex"})
        df["diag_age"] = pd.NA
    else:
        raise ValueError(dataset)

    df["sex"] = _norm_sex(df["sex"])
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["diag_age"] = pd.to_numeric(df["diag_age"], errors="coerce")
    df["duration"] = df["age"] - df["diag_age"]
    df["ptid"] = df["PtID"].astype(str)
    df["pid"] = pfx + "_" + df["ptid"]
    return df[["pid", "ptid", "sex", "age", "diag_age", "duration"]].drop_duplicates(
        "pid"
    )
