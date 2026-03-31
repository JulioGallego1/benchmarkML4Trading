from __future__ import annotations
import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)

def read_file(file_path: str) -> pd.DataFrame:
    """Read a data file into a pandas DataFrame.

    Currently supports Parquet. Raises an informative error if the file cannot be 
    found or the extension is unknown.

    Parameters
    ----------
    file_path : str
        Absolute or relative path to the file on disk.

    Returns
    -------
    pd.DataFrame
        The loaded DataFrame.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext == ".parquet":
        return pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}. Supported extension is .parquet.")
    
def load_price_data(path: str) -> pd.DataFrame:
    """Load a price time series from a pipeline-managed Parquet file.

    Expects columns ``["Date", "Close"]`` guaranteed by the download step:
    Date as timezone-naive datetime64, Close as numeric, sorted ascending,
    no duplicates.

    Raises
    ------
    ValueError
        If the expected columns are missing from the file.
    """
    df = read_file(path)

    missing = {"Date", "Close"} - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns {missing} in {path}.")

    logger.debug(f"Loaded {len(df)} records from {path}.")
    return df[["Date", "Close"]]