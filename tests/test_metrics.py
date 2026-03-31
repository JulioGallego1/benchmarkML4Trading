import numpy as np
import pytest
from tsforecast.evaluation.metrics import mae, rmse, mape, smape, directional_accuracy

def test_mae_perfect():
    y = np.array([1.0, 2.0, 3.0])
    assert mae(y, y) == pytest.approx(0.0)

def test_mae_known():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 3.0, 4.0])
    assert mae(y_true, y_pred) == pytest.approx(1.0)

def test_rmse_known():
    y_true = np.array([0.0, 0.0, 0.0])
    y_pred = np.array([1.0, 1.0, 1.0])
    assert rmse(y_true, y_pred) == pytest.approx(1.0)

def test_mape_known():
    y_true = np.array([100.0, 200.0])
    y_pred = np.array([110.0, 180.0])
    expected = (10/100 + 20/200) / 2 * 100  # 10.0
    assert mape(y_true, y_pred) == pytest.approx(expected, rel=1e-4)

def test_smape_symmetric():
    y_true = np.array([1.0, 2.0])
    y_pred = np.array([2.0, 1.0])
    # smape should be symmetric: swap y_true and y_pred -> same result
    assert smape(y_true, y_pred) == pytest.approx(smape(y_pred, y_true), rel=1e-5)

def test_directional_accuracy_all_correct():
    # Raw prices. Window 1: anchor=100, last price=103 (up). Window 2: up. Window 3: down.
    # Predictions match actual net direction in all 3 windows.
    anchors = np.array([100.0, 100.0, 100.0])
    y_true = np.array([[101.0, 103.0], [102.0, 104.0], [99.0, 97.0]])
    y_pred = np.array([[102.0, 105.0], [101.0, 103.0], [98.0, 96.0]])
    da = directional_accuracy(y_true, y_pred, anchors)
    assert da == pytest.approx(100.0)

def test_directional_accuracy_all_wrong():
    # anchor=100, y_true last=103 (up), y_pred last=96 (down) -> wrong
    anchors = np.array([100.0])
    y_true = np.array([[101.0, 103.0]])  # net up
    y_pred = np.array([[98.0, 96.0]])    # net down
    da = directional_accuracy(y_true, y_pred, anchors)
    assert da == pytest.approx(0.0)

def test_directional_accuracy_half_correct():
    # Window 1: anchor=100, y_true last=103 (up), y_pred last=105 (up) -> correct
    # Window 2: anchor=100, y_true last=103 (up), y_pred last=96 (down) -> wrong
    anchors = np.array([100.0, 100.0])
    y_true = np.array([[101.0, 103.0], [101.0, 103.0]])
    y_pred = np.array([[102.0, 105.0], [98.0, 96.0]])
    da = directional_accuracy(y_true, y_pred, anchors)
    assert da == pytest.approx(50.0)

def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        mae(np.array([1.0, 2.0]), np.array([1.0]))
