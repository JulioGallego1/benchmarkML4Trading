import numpy as np
import pandas as pd
import pytest
from tsforecast.data.splits import make_time_splits

def make_df(n=1200):
    dates = pd.date_range("2018-01-01", periods=n, freq="B")
    return pd.DataFrame({"Date": dates, "Close": np.random.rand(n)})

def test_non_overlapping_splits():
    df = make_df()
    splits = make_time_splits(df, test_start="2022-01-01", test_days=63, val_days=63, context_length=32)
    train_end = splits["train"][1]
    val_start = splits["val"][0]
    val_end = splits["val"][1]
    test_start = splits["test"][0]
    # Train ends before val starts (val_start includes context overlap before val window)
    # The actual prediction portion of val starts at val_start + context_length
    assert train_end <= val_start + 32  # train ends where val prediction starts (minus context)
    assert val_end <= test_start + 32   # val ends before test prediction starts (minus context)

def test_context_preserved_in_val():
    df = make_df()
    L = 48
    splits = make_time_splits(df, test_start="2022-01-01", test_days=63, val_days=63, context_length=L)
    val_start, val_end = splits["val"]
    # val slice must be at least context_length + 1 rows to form any window
    assert val_end - val_start >= L + 1

def test_context_preserved_in_test():
    df = make_df()
    L = 48
    splits = make_time_splits(df, test_start="2022-01-01", test_days=63, val_days=63, context_length=L)
    test_start_idx, test_end_idx = splits["test"]
    assert test_end_idx - test_start_idx >= L + 1

def test_insufficient_history_raises():
    df = make_df(100)
    with pytest.raises(ValueError):
        make_time_splits(df, test_start="2018-01-02", test_days=63, val_days=63, context_length=200)

def test_train_ends_before_val_prediction_window():
    df = make_df()
    L = 32
    splits = make_time_splits(df, test_start="2022-01-01", test_days=21, val_days=21, context_length=L)
    train_start, train_end = splits["train"]
    val_start, val_end = splits["val"]
    # train end == val start (val_start is where context of val begins)
    assert train_end == val_start
