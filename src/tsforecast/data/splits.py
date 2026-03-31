from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def make_time_splits(df: pd.DataFrame, 
                     test_start: str,
                     test_days: int,
                     val_days: int,
                     context_length: int,
                     date_col: str = 'Date',
                     verbose: bool = False
                    ) -> Dict[str, Tuple[int, int]]:
    """Compute train/validation/test index ranges for a time series.

    Given a DataFrame sorted by date, this function identifies integer
    index ranges corresponding to a training period, a validation period
    and a test period. The splits respect temporal ordering and include
    ``context_length`` extra rows at the beginning of the validation and
    test sets to ensure that windows can be formed without running
    off the start of the series.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least a date column. Must be sorted by date.
    test_start : str
        The first date to include in the test prediction window.
    test_days : int
        Number of target days to predict in the test set.
    val_days : int
        Number of target days to predict in the validation set.
    context_length : int
        Number of past observations used as input by the model. This
        quantity must be available immediately before each prediction
        window.
    date_col : str, optional
        Name of the date column. Defaults to ``"Date"``.
    verbose : bool, optional
        If True, print information about the splits.

    Returns
    -------
    Dict[str, Tuple[int, int]]
        Dictionary with keys ``"train"``, ``"val"`` and ``"test"`` mapping
        to tuples ``(start, end)``. The indices follow Python slicing
        semantics such that ``df.iloc[start:end]`` returns the raw rows
        (including the ``context_length`` overlap) used to form windows.

    Raises
    ------
    ValueError
        If there is not enough history prior to the validation set to
        accommodate the specified ``context_length``.
    """

    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found in DataFrame.")
    
    date_arr = df[date_col].to_numpy(dtype='datetime64[ns]')

    # Validate that test_start lies within the observed date range
    first_date = date_arr[0]
    last_date = date_arr[-1]
    test_start_dt = np.datetime64(pd.to_datetime(test_start), "ns")
    if test_start_dt < first_date or test_start_dt > last_date:
        raise ValueError(
            f"test_start={test_start!r} is outside the available date range "
            f"[{first_date}, {last_date}]."
        )

    test_start_idx = int(np.searchsorted(date_arr, test_start_dt))
    test_end_idx = test_start_idx + test_days

    if test_end_idx > len(df):
        available = len(df) - test_start_idx
        logger.warning(
            f"Requested test_days={test_days} but only {available} days available "
            f"from test_start. Clamping to {available}."
        )
        test_end_idx = len(df)

    val_end_idx = test_start_idx
    val_start_idx = val_end_idx - val_days

    if val_start_idx < context_length:
        raise ValueError(f"Not enough history for validation set. "
                         f"val_start_idx={val_start_idx} < context_length={context_length}.")

    val_context_start = val_start_idx - context_length

    splits = {
        "train": (0, val_context_start),
        "val": (val_context_start, val_end_idx),
        "test": (test_start_idx - context_length, test_end_idx)
    }

    if verbose:
        for name, (start, end) in splits.items():
            start_idx = max(start, 0)
            end_idx = min(end, len(df))
            rng = end_idx - start_idx
            d_start = df.iloc[start_idx][date_col].date() if start_idx < len(df) else "N/A"
            d_end = df.iloc[end_idx - 1][date_col].date() if end_idx > 0 else "N/A"
            print(
                f"{name.upper():<5}: idx [{start} -> {end}] | {d_start} -> {d_end} | rows (including context)={rng}"
            )

    return splits