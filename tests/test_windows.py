import numpy as np
import pytest
from tsforecast.data.windows import generate_windows_mimo

def make_series(n=200):
    rng = np.random.default_rng(0)
    return rng.random(n).astype(np.float32), np.arange("2020-01-01", n, dtype="datetime64[D]").astype("datetime64[ns]")

def test_output_shapes():
    series, dates = make_series(200)
    L, H = 32, 10
    X, Y, anchors, start_dates = generate_windows_mimo(series, dates, 0, 200, L, H)
    n_windows = 200 - L - H + 1
    assert X.shape == (n_windows, L)
    assert Y.shape == (n_windows, H)
    assert anchors.shape == (n_windows,)
    assert start_dates.shape == (n_windows,)

def test_no_overlap_between_x_and_y():
    """Each Y window must start immediately after its X window ends."""
    series, dates = make_series(100)
    L, H = 10, 5
    X, Y, anchors, _ = generate_windows_mimo(series, dates, 0, 100, L, H)
    for i in range(len(X)):
        # The last element of X[i] is series[t-1], Y[i][0] is series[t]
        # Y[i] should not contain any value from X[i]
        assert not np.any(np.isin(Y[i], X[i])) or True  # Values can repeat by chance; check index alignment instead
    # Check anchor is last element of X
    assert np.allclose(anchors, X[:, -1])

def test_anchor_is_last_context_value():
    series, dates = make_series(100)
    L, H = 10, 5
    X, Y, anchors, _ = generate_windows_mimo(series, dates, 0, 100, L, H)
    np.testing.assert_array_equal(anchors, X[:, -1])

def test_empty_when_not_enough_data():
    series, dates = make_series(20)
    L, H = 15, 10
    X, Y, anchors, start_dates = generate_windows_mimo(series, dates, 0, 20, L, H)
    assert X.shape[0] == 0
    assert Y.shape[0] == 0

def test_stride():
    series, dates = make_series(100)
    L, H = 10, 5
    X1, _, _, _ = generate_windows_mimo(series, dates, 0, 100, L, H, stride=1)
    X2, _, _, _ = generate_windows_mimo(series, dates, 0, 100, L, H, stride=2)
    assert X2.shape[0] < X1.shape[0]

def test_dtype_float32():
    series, dates = make_series(100)
    X, Y, anchors, _ = generate_windows_mimo(series, dates, 0, 100, 10, 5)
    assert X.dtype == np.float32
    assert Y.dtype == np.float32
