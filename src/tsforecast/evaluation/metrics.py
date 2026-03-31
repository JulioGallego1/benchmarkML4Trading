from __future__ import annotations

import numpy as np
from typing import Union

def _check_shapes(y_true: np.ndarray, y_pred: np.ndarray):
    """Validate that y_true and y_pred have matching shapes."""
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shapes of y_true {y_true.shape} and y_pred {y_pred.shape} do not match.")

def mae(y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]) -> float:
    """Compute Mean Absolute Error (MAE) between true and predicted values.

    Parameters
    ----------
    y_true : np.ndarray or list
        Array of true target values.
    y_pred : np.ndarray or list
        Array of predicted values.

    Returns
    -------
    float
        The mean absolute error.

    Notes
    -----
    MAE is a common regression metric that measures the average magnitude of
    errors in a set of predictions, without considering their direction. It is
    calculated as the average of the absolute differences between predicted and
    true values.
    """
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)
    
    _check_shapes(y_true, y_pred)

    return float(np.mean(np.abs(y_true - y_pred)))

def rmse(y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]) -> float:
    """Compute Root Mean Squared Error (RMSE) between true and predicted values.

    Parameters
    ----------
    y_true : np.ndarray or list
        Array of true target values.
    y_pred : np.ndarray or list
        Array of predicted values.

    Returns
    -------
    float
        The root mean squared error.

    Notes
    -----
    RMSE is a common regression metric that measures the square root of the
    average of squared differences between predicted and true values. It gives
    higher weight to larger errors compared to MAE.
    """
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)
    
    _check_shapes(y_true, y_pred)
    
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mape(y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list], eps: float = 1e-8) -> float:
    """Mean Absolute Percentage Error.

    Calculates the average absolute difference between predicted and true
    values relative to the true values, expressed as a percentage. A
    small epsilon is used to avoid division by zero when the true
    values are close to zero. The returned value is scaled by 100 so
    that a result of 1.5 corresponds to a 1.5% error.

    Parameters
    ----------
    y_true : array-like
        Ground truth values.
    y_pred : array-like
        Predicted values.
    eps : float, optional
        Small constant to avoid division by zero. Defaults to 1e-8.

    Returns
    -------
    float
        The mean absolute percentage error (in percent).
    """
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)

    _check_shapes(y_true, y_pred)

    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def smape(y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list], eps: float = 1e-8) -> float:
    """Symmetric Mean Absolute Percentage Error.

    This metric is similar to MAPE but symmetric and bounded between 0
    and 200%. It mitigates the issue of extremely large errors when the
    true values are close to zero by normalising the absolute error by
    the sum of the absolute true and predicted values.

    Parameters
    ----------
    y_true : array-like
        Ground truth values.
    y_pred : array-like
        Predicted values.
    eps : float, optional
        Small constant to avoid division by zero. Defaults to 1e-8.

    Returns
    -------
    float
        The symmetric mean absolute percentage error (in percent).
    """
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)

    _check_shapes(y_true, y_pred)

    denom = np.maximum((np.abs(y_true) + np.abs(y_pred)), eps)
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom) * 100.0)

def directional_accuracy(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    anchors: Union[np.ndarray, list],
) -> float:
    """Directional Accuracy (per-window net direction).

    Computes the percentage of forecast windows in which the model correctly
    predicted the **net direction** of the price series over the full horizon.

    Direction is the sign of ``last_price - anchor``, where ``anchor`` is the
    last observed price before the forecast window.

    Parameters
    ----------
    y_true : array-like
        Ground truth prices with shape ``(n_windows, horizon)`` or
        ``(horizon,)`` for a single window.
    y_pred : array-like
        Predicted prices with the same shape as ``y_true``.
    anchors : array-like
        Shape ``(n_windows,)``. Last observed price before each forecast
        window.

    Returns
    -------
    float
        The percentage of windows (0–100) where the predicted direction
        matches the actual direction.

    Notes
    -----
    This metric differs from step-wise MDA (Mean Directional Accuracy).
    It evaluates whether the model correctly identified the *overall trend*
    for the forecast horizon, which is more meaningful for trading strategies.
    """
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)

    _check_shapes(y_true, y_pred)

    if y_true.ndim == 1:
        # Single window: treat as (1, H)
        y_true = y_true[np.newaxis, :]
        y_pred = y_pred[np.newaxis, :]

    anchors = np.array(anchors, dtype=np.float64).ravel()
    dir_true = np.sign(y_true[:, -1] - anchors)
    dir_pred = np.sign(y_pred[:, -1] - anchors)

    correct = dir_true == dir_pred
    return float(np.mean(correct) * 100.0)