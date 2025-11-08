# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)
# For commercial licensing, contact: [Add your contact information]

import pandas as pd


def mg_dl_to_mmol_l(df: pd.DataFrame, bgl_col: str = "bg-0:00") -> pd.DataFrame:
    return (df[bgl_col] / 18.018).round(2)
