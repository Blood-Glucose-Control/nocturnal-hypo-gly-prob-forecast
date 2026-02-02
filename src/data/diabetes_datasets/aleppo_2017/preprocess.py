# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)
# For commercial licensing, contact: christopher/cjrisi AT gluroo/uwaterloo DOT com/ca

import pandas as pd
import os
import logging
from src.utils.os_helper import get_project_root
import sqlite3
from pathlib import Path

"""
Note from the readme:
Data is de-identified
Useful tables:
- HDeviceBGM: One record per bgm or ketone reading
- HDeviceCGM: One record per cgm reading

- HDeviceBolus: One record per bolus reading from a pump
- HDeviceBasal: One record per basal reading from a pump

- HHypoEvent: One record per hypoglycemic event submitted (This can be labels for hypoglycemia prediction?)
- HInsulin: One record per Insulin type per patient
- HMedication: One record per medical conditions form (exgo var potentially)
- HDeviceWizard: One record per pump wizard reading
    This table points to HDeviceUploads and HDeviceBolus
    - BgInput: Blood glucose as inputted into wizard in mg
    - CarbInput: Carbohydrates as inputted into wizard in mg
    - InsulinOnBoard
    - InsulinCarbRatio
    - InsulinSensitivity



## For data processing:
- HAEDeviceProblems: We can squash the entire day if there is an device issue?
- HDeviceIssue: Same as above (see DevIssueType column)
- HDeviceUploads: This is technically a categorical data (we can assign and id to each device, hoping ttm will pick that up, CGM/BGM)


- HQuestHypoFear: One record per Hypoglycemia Fear Survey submitted (Probably not useful for data but probably interested to look into)
- HQuestHypoUnaware: Same as above
"""

logger = logging.getLogger(__name__)


repo = get_project_root()
CACHE_DIR = repo / "cache" / "data" / "aleppo_2017"
DATA_TABLES = CACHE_DIR / "raw" / "Data Tables"
DB_PATH = CACHE_DIR / "aleppo.db"

# SQLite param cap (commonly 999). Keep a margin.
SQLITE_MAX_VARS = 999
MARGIN = 10


# The SQL query to convert the raw data to a database by merging 4 tables and sort by pid and date.
# Note that each table only have a day offset from enrollment date so we here use 2020-01-01 as the base date.
# Full outer join takes way too long to process.
template_query = """
WITH
  params AS (
    SELECT '2020-01-01' AS base_date
  )
SELECT
    pid, date, tableType,
    bolusType, normalBolus, expectedNormalBolus, extendedBolus, expectedExtendedBolus,
    bgInput, foodG, iob, cr, isf,
    bgMgdl,
    rate, basalDurationMins, suprBasalType, suprRate
FROM (
    -- Bolus data
    SELECT
        HDeviceBolus.PtID as pid,
        datetime(julianday((SELECT base_date FROM params)) + CAST(HDeviceBolus.DeviceDtTmDaysFromEnroll AS INTEGER) + (julianday(HDeviceBolus.DeviceTm) - julianday('00:00:00'))) AS date,
        'bolus' as tableType,
        HDeviceBolus.BolusType as bolusType,
        Normal as normalBolus,
        ExpectedNormal as expectedNormalBolus,
        Extended as extendedBolus,
        ExpectedExtended as expectedExtendedBolus,
        NULL as bgInput,
        NULL as foodG,
        NULL as iob,
        NULL as cr,
        NULL as isf,
        NULL as bgMgdl,
        NULL as rate,
        NULL as basalDurationMins,
        NULL as suprBasalType,
        NULL as suprRate
    FROM HDeviceBolus

    UNION ALL

    -- Wizard data
    SELECT
        HDeviceWizard.PtID as pid,
        datetime(julianday((SELECT base_date FROM params)) + CAST(HDeviceWizard.DeviceDtTmDaysFromEnroll AS INTEGER) + (julianday(HDeviceWizard.DeviceTm) - julianday('00:00:00'))) AS date,
        'wizard' as tableType,
        NULL as bolusType,
        NULL as normalBolus,
        NULL as expectedNormalBolus,
        NULL as extendedBolus,
        NULL as expectedExtendedBolus,
        HDeviceWizard.BgInput as bgInput,
        HDeviceWizard.CarbInput as foodG, --  Readme says this is in mg but I think it is in grams
        HDeviceWizard.InsulinOnBoard as iob,
        HDeviceWizard.InsulinCarbRatio as cr,
        HDeviceWizard.InsulinSensitivity as isf,
        NULL as bgMgdl,
        NULL as rate,
        NULL as basalDurationMins,
        NULL as suprBasalType,
        NULL as suprRate
    FROM HDeviceWizard

    UNION ALL

    -- CGM data
    SELECT
        HDeviceCGM.PtID as pid,
        datetime(julianday((SELECT base_date FROM params)) + CAST(HDeviceCGM.DeviceDtTmDaysFromEnroll AS INTEGER) + (julianday(HDeviceCGM.DeviceTm) - julianday('00:00:00'))) AS date,
        'cgm' as tableType,
        NULL as bolusType,
        NULL as normalBolus,
        NULL as expectedNormalBolus,
        NULL as extendedBolus,
        NULL as expectedExtendedBolus,
        NULL as bgInput,
        NULL as foodG,
        NULL as iob,
        NULL as cr,
        NULL as isf,
        HDeviceCGM.GlucoseValue as bgMgdl,
        NULL as rate,
        NULL as basalDurationMins,
        NULL as suprBasalType,
        NULL as suprRate
    FROM HDeviceCGM
    WHERE HDeviceCGM.GlucoseValue IS NOT NULL
    AND HDeviceCGM.RecordType = 'CGM'

    UNION ALL

    -- Basal data
    SELECT
        HDeviceBasal.PtID as pid,
        datetime(julianday((SELECT base_date FROM params)) + CAST(HDeviceBasal.DeviceDtTmDaysFromEnroll AS INTEGER) + (julianday(HDeviceBasal.DeviceTm) - julianday('00:00:00'))) AS date,
        'basal' as tableType,
        NULL as bolusType,
        NULL as normalBolus,
        NULL as expectedNormalBolus,
        NULL as extendedBolus,
        NULL as expectedExtendedBolus,
        NULL as bgInput,
        NULL as foodG,
        NULL as iob,
        NULL as cr,
        NULL as isf,
        NULL as bgMgdl,
        HDeviceBasal.Rate as rate,
        HDeviceBasal.Duration / 60000 as basalDurationMins,
        HDeviceBasal.SuprBasalType as suprBasalType,
        HDeviceBasal.SuprRate as suprRate
    FROM HDeviceBasal
)
ORDER BY pid, date ASC;
"""


def _raw_to_db(raw_folder_path: Path):
    """
    Args:
        raw_folder_path (Path): Path to the `raw` folder for the dataset. This should be the `Data Tables` folder.
        output_csv_path (str): Path to the CSV file for the processed data.
        Create a database from the raw data.
    """
    con = sqlite3.connect(DB_PATH)
    # cur = con.cursor()
    # cur.execute("PRAGMA journal_mode=WAL;")
    # cur.execute("PRAGMA synchronous=OFF;")
    # cur.execute("PRAGMA temp_store=MEMORY;")

    if not os.path.exists(raw_folder_path):
        raise FileNotFoundError(
            "Raw data does not exist, please follow the instructions in the README.md to download the data and place it in the correct cache directory."
        )

    for f in sorted(DATA_TABLES.glob("*.txt")):
        table = f.stem
        logger.info(f"Importing {f.name} -> {table}")

        # Stream in chunks to control memory and SQL variables
        first = True
        for df in pd.read_csv(
            f, sep="|", dtype=str, low_memory=False, chunksize=50_000
        ):
            ncols = len(df.columns)
            # rows per insert so (rows * cols) <= SQLITE_MAX_VARS - MARGIN
            safe_rows = max(1, (SQLITE_MAX_VARS - MARGIN) // max(1, ncols))

            df.to_sql(
                table,
                con,
                if_exists="replace" if first else "append",
                index=False,
                chunksize=safe_rows,
                method=None,
            )
            first = False

    con.commit()
    con.close()


# This can probbaly be in a utils file
def query_db(query: str, db_path: Path = DB_PATH):
    # Open the connection first
    con = sqlite3.connect(db_path)
    # cur = con.cursor()
    df = pd.read_sql_query(query, con)
    con.close()
    return df


# Convert the data to a CSV file by patients.
def convert_to_csv(df):
    logger.info("Converting data to CSV")
    project_root = get_project_root()
    # TODO: Probably shouldn't hardcode this.
    data_dir = project_root / "cache" / "data" / "aleppo_2017" / "interim"
    os.makedirs(data_dir, exist_ok=True)
    for pid in df["pid"].unique():
        df_pid = df[df["pid"] == pid]
        df_pid.to_csv(data_dir / f"p{pid}_full.csv", index=False)
        logger.info(f"Done processing pid {pid}")


def create_aleppo_csv(raw_folder_path: Path):
    """
    Args:
        raw_folder_path (Path): Path to the `raw` folder for the dataset. This should be the `Data Tables` folder.
    """

    # Import the raw data to a database.
    _raw_to_db(raw_folder_path)

    # Query the database to get the data.
    raw_df = query_db(template_query)

    # Save the data to a CSV file.
    convert_to_csv(raw_df)
