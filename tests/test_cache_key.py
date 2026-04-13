"""Tests for cache key consistency including strategy and step (Issue 8)."""
import numpy as np
import pytest


def test_cache_key_dot_ticker_matches_safe_name():
    """Cache key for 'AIR.PA' must use '_' not '.', matching safe_name output."""
    from tsforecast.data.cache import WindowCache
    from tsforecast.data.download_close_prices import safe_name

    cache = WindowCache("data/processed")
    key = cache.key("AIR.PA", "bull", 96, 21)
    # The ticker portion of the key must match what safe_name produces (lowercased).
    assert safe_name("AIR.PA").lower() in key
    assert "air.pa" not in key


def test_cache_key_slash_ticker_matches_safe_name():
    """Cache key for 'BRK/A' must use '_' not '/', matching safe_name output."""
    from tsforecast.data.cache import WindowCache
    from tsforecast.data.download_close_prices import safe_name

    cache = WindowCache("data/processed")
    key = cache.key("BRK/A", "bull", 96, 21)
    assert safe_name("BRK/A").lower() in key
    assert "brk/a" not in key


def test_cache_key_includes_strategy():
    from tsforecast.data.cache import WindowCache

    cache = WindowCache("data/processed")
    key_mimo = cache.key("AAPL", "bear", 96, 21, "mimo", 0)
    key_rec = cache.key("AAPL", "bear", 96, 21, "recursive", 16)
    assert key_mimo != key_rec, "mimo and recursive keys must differ"


def test_cache_key_includes_step():
    from tsforecast.data.cache import WindowCache

    cache = WindowCache("data/processed")
    key_step8 = cache.key("AAPL", "bear", 96, 63, "recursive", 8)
    key_step16 = cache.key("AAPL", "bear", 96, 63, "recursive", 16)
    assert key_step8 != key_step16, "Different steps must produce different keys"


def test_cache_key_includes_h_total():
    from tsforecast.data.cache import WindowCache

    cache = WindowCache("data/processed")
    key_h21 = cache.key("AAPL", "bear", 96, 21)
    key_h63 = cache.key("AAPL", "bear", 96, 63)
    assert key_h21 != key_h63


def test_cache_roundtrip(tmp_path):
    from tsforecast.data.cache import WindowCache

    cache = WindowCache(tmp_path)
    rng = np.random.default_rng(0)

    def rand(shape):
        return rng.random(shape).astype(np.float32)

    n_tr, n_v, n_te, L, H = 50, 10, 10, 16, 4
    dates = (np.arange(n_tr, dtype="int64") * int(1e9) * 86400).view("datetime64[ns]")

    cache.save(
        "TEST", "bear", L, H, "mimo", 0,
        rand((n_tr, L)), rand((n_tr, H)), rand(n_tr), dates,
        rand((n_v, L)), rand((n_v, H)), rand(n_v), dates[:n_v],
        rand((n_te, L)), rand((n_te, H)), rand(n_te), dates[:n_te],
    )
    assert cache.exists("TEST", "bear", L, H, "mimo", 0)
    data = cache.load("TEST", "bear", L, H, "mimo", 0)
    assert data["X_train"].shape == (n_tr, L)
    assert data["Y_test"].shape == (n_te, H)


def test_cache_mimo_recursive_separate_files(tmp_path):
    from tsforecast.data.cache import WindowCache

    cache = WindowCache(tmp_path)
    rng = np.random.default_rng(1)

    def rand(shape):
        return rng.random(shape).astype(np.float32)

    n, L, H = 20, 16, 4
    dates = (np.arange(n, dtype="int64") * int(1e9) * 86400).view("datetime64[ns]")

    args = ("TEST", "bear", L, H)

    cache.save(*args, "mimo", 0,
               rand((n, L)), rand((n, H)), rand(n), dates,
               rand((n, L)), rand((n, H)), rand(n), dates,
               rand((n, L)), rand((n, H)), rand(n), dates)

    cache.save(*args, "recursive", 8,
               rand((n, L)), rand((n, H)), rand(n), dates,
               rand((n, L)), rand((n, H)), rand(n), dates,
               rand((n, L)), rand((n, H)), rand(n), dates)

    # Both exist independently
    assert cache.exists(*args, "mimo", 0)
    assert cache.exists(*args, "recursive", 8)

    # They are different files
    path_mimo = cache._path(*args, "mimo", 0)
    path_rec = cache._path(*args, "recursive", 8)
    assert path_mimo != path_rec
