# This is the script for processing Lynch2022 data into the common format.
# Author: Neo Kok (ported to Python by ChatGPT)
# Date: 10/24/2024

import os
import pandas as pd
import pyreadstat
from pathlib import Path

# --- File paths ---
project_root = Path(__file__).parents[5]
base_path = project_root / "cache" / "data" / "awesome_cgm" / "lynch_2022" / "raw" / "IOBP2 RCT Public Dataset" / "Data Tables in SAS"
file_data = base_path / "iobp2devicecgm.sas7bdat"
file_demo = base_path / "iobp2diabscreening.sas7bdat"
file_age  = base_path / "iobp2ptroster.sas7bdat"

# --- Read SAS files ---
data, _ = pyreadstat.read_sas7bdat(str(file_data))
demo, _ = pyreadstat.read_sas7bdat(str(file_demo))
age,  _ = pyreadstat.read_sas7bdat(str(file_age))

# --- Select necessary columns ---
data = data[["PtID", "DeviceDtTm", "Value"]]
demo = demo[["PtID", "InsModPump", "Sex"]]
age  = age[["PtID", "AgeAsofEnrollDt"]]

# --- Merge ---
data = data.merge(demo, on="PtID", how="left")
data = data.merge(age, on="PtID", how="left")

# --- Rename and transform ---
df_final = (
    data.rename(columns={
        "PtID": "id",
        "DeviceDtTm": "time",
        "Value": "gl",
        "AgeAsofEnrollDt": "age",
        "Sex": "sex",
        "InsModPump": "insulinModality"
    })
    .assign(
        time=lambda df: pd.to_datetime(df["time"], errors="coerce"),
        type=1,
        device="Dexcom G6",
        dataset="lynch2022",
        insulinModality=lambda df: df["insulinModality"].notna().astype(int)
    )
    .dropna(subset=["time", "gl"])
    .sort_values(["id", "time"])
)

# --- Create pseudoID per subject ---
group_ids = {old: new_id for new_id, old in enumerate(df_final["id"].unique(), start=1001)}
df_final["id"] = df_final["id"].map(group_ids)

# --- Reorder columns ---
df_final = df_final[["id", "time", "gl", "age", "sex", "insulinModality", "type", "device", "dataset"]]

# --- Save CSV ---
processed_dir = project_root / "cache" / "data" / "awesome_cgm" / "lynch_2022" / "processed"
processed_dir.mkdir(parents=True, exist_ok=True)
output_path = processed_dir / "lynch2022.csv"
df_final.to_csv(output_path, index=False)

print(f"✅ Processed dataset saved to {output_path}")
