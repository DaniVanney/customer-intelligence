from pathlib import Path

import pandas as pd


def load_retail_excel(path: Path) -> pd.DataFrame:
    """Load every Excel sheet and combine them row-wise."""
    sheets = pd.read_excel(path, sheet_name=None)
    return pd.concat(sheets.values(), ignore_index=True)


def save_parquet(data: pd.DataFrame, path: Path) -> None:
    """Save tabular data as Parquet, creating its directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(path, index=False)