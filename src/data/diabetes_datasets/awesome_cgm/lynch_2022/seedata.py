import pandas as pd
import os
from pathlib import Path

project_root = Path(__file__).parents[5]
base_path = project_root / "cache" / "data" / "awesome_cgm" / "lynch_2022" / "raw" / "IOBP2 RCT Public Dataset" / "Data Tables in SAS"

df = pd.read_sas(base_path / 'iobp2devicecgm.sas7bdat')
df2 = pd.read_sas(base_path / 'iobp2deviceilet.sas7bdat')

print(df.columns)
print(df2.columns)
print(df.head())
print(df2.head())

# --- New: print first few values for each column in df2 ---
print("\nFirst few values per column in df2:")
for col in df2.columns:
    s = df2[col]
    sample = s.dropna().head(5).tolist()
    if not sample:
        sample = s.head(5).tolist()
    print(f"- {col}: {sample}")
