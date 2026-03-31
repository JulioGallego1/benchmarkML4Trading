from __future__ import annotations
from typing import Tuple

import numpy as np
import pandas as pd

def generate_windows_mimo(
        series: np.ndarray,
        dates: np.ndarray,
        start: int,
        end: int,
        context_length: int,
        horizon: int,
        stride: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate sliding windows of past context and future targets (MIMO).

        Given a 1-D sequence of numerical values and corresponding dates,
        this function constructs input-output pairs for supervised learning. Each
        input consists of ``context_length`` consecutive past values and each
        output consists of ``horizon`` consecutive future values. The windows
        are moved forward in time by ``stride`` steps. Optionally, the
        function also returns the ``anchor`` (the last observed value prior
        to each prediction) and the date corresponding to the first predicted
        value, which can be useful for computing directional accuracy or
        aligning predictions back to calendar dates.

        Parameters
        ----------
        series : np.ndarray
            One-dimensional array of values from which to construct windows.
            Example: closing prices or returns.
        dates : np.ndarray
            Array of datetime64 objects aligned with ``series``. Used only
            to return start dates. Must be the same length as ``series``.
        start : int
            Inclusive start index for the raw slice of ``series`` to use.
            Note: if ``start`` is negative it will be clamped to 0.
        end : int
            Exclusive end index for the raw slice of ``series`` to use.
        context_length : int
            Number of past observations in each input window.
        horizon : int
            Number of future observations to predict for each window.
        stride : int, optional
            Step size between successive windows. Defaults to 1.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            ``(X, Y, anchors, start_dates)`` where:

            * ``X`` has shape ``(n_windows, context_length)``
            * ``Y`` has shape ``(n_windows, horizon)``
            * ``anchors`` has shape ``(n_windows,)`` containing the last
            observed value prior to each prediction window
            * ``start_dates`` has shape ``(n_windows,)`` containing the
            ``datetime64[ns]`` for the first predicted value

        Notes
        -----
        If the computed range does not contain enough rows to form any
        windows, empty arrays are returned for all outputs. The inputs and
        targets are returned with dtype ``float32`` to keep memory usage
        manageable when datasets are large.
        """
    if series.ndim != 1:
        raise ValueError("Input series must be one-dimensional.")
    if dates.ndim != 1 or len(dates) != len(series):
        raise ValueError("Dates array must be one-dimensional and the same length as series.")
    
    n = len(series)
    start = max(start, 0)
    end = min(end, n)

    t_start = start + context_length
    t_end = end - horizon + 1

    if t_end <= t_start or context_length <= 0 or horizon <= 0:
        # Return empty arrays if no windows can be formed
        return (
            np.empty((0, context_length), dtype=np.float32),
            np.empty((0, horizon), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype="datetime64[ns]"),
        )

    # Vectorized window generation using numpy stride tricks (NumPy >= 1.20).
    # Each window covers context_length + horizon consecutive values; we then
    # split into X (past) and Y (future) without any Python-level loop.
    window_size = context_length + horizon
    # sliding_window_view returns a read-only view of shape (n_full, window_size)
    # where n_full = len(series) - window_size + 1.  We index by the positions
    # corresponding to t_start..t_end with the given stride.
    full_windows = np.lib.stride_tricks.sliding_window_view(
        series.astype(np.float64), window_size
    )
    # t_start is the first valid "prediction start"; in full_windows coordinates
    # the window starting at position p has its context ending at p + context_length - 1
    # and its target starting at p + context_length.  So the window index equals
    # t - context_length where t is the prediction start index.
    window_indices = np.arange(t_start - context_length, t_end - context_length, stride)
    selected = full_windows[window_indices]          # shape (n_windows, window_size)

    X = selected[:, :context_length].astype(np.float32)
    Y = selected[:, context_length:].astype(np.float32)

    # Anchors: last observed value before each prediction window (index t-1)
    t_positions = window_indices + context_length     # t values in original series
    anchors_arr = series[t_positions - 1].astype(np.float32)

    # Start dates: date of the first predicted step (index t)
    start_dates_arr = dates[t_positions].astype("datetime64[ns]")

    return X, Y, anchors_arr, start_dates_arr

