import pandas as pd
def mg_dl_to_mmol_l(df: pd.DataFrame, bgl_col: str = "bg-0:00") -> pd.DataFrame:
    return (df[bgl_col] / 18.0).round(2)