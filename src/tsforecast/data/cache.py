from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class WindowCache:
    """Disk cache for windowed datasets.

    Each cache entry stores all three splits (train/val/test) for a given
    (ticker, regime, L, H_total, strategy, step) combination as a
    single compressed .npz file under ``cache_dir``.

    Usage::

        cache = WindowCache("data/processed")
        if cache.exists(ticker, regime, L, H, strategy, step):
            data = cache.load(ticker, regime, L, H, strategy, step)
        else:
            # ... compute windows ...
            cache.save(ticker, regime, L, H, strategy, step, X_train, ...)
    """

    _VALID_STRATEGIES = {"mimo", "recursive"}

    def __init__(self, cache_dir: str | Path = "data/processed") -> None:
        self.cache_dir = Path(cache_dir)

    def key(
        self,
        ticker: str,
        regime: str,
        L: int,
        H_total: int,
        strategy: str = "mimo",
        step: int = 0,
    ) -> str:
        """Return a deterministic filename stem based on params."""
        self._validate_common_params(L, H_total, strategy, step)

        ticker_s = self._sanitize_part(ticker)
        regime_s = self._sanitize_part(regime)
        strategy_s = self._sanitize_part(strategy)

        step_part = f"_step{step}" if strategy_s == "recursive" else ""
        return f"{ticker_s}_{regime_s}_L{L}_H{H_total}_{strategy_s}{step_part}_v2".lower()

    def _path(
        self,
        ticker: str,
        regime: str,
        L: int,
        H_total: int,
        strategy: str = "mimo",
        step: int = 0,
    ) -> Path:
        return self.cache_dir / f"{self.key(ticker, regime, L, H_total, strategy, step)}.npz"

    def exists(
        self,
        ticker: str,
        regime: str,
        L: int,
        H_total: int,
        strategy: str = "mimo",
        step: int = 0,
    ) -> bool:
        return self._path(ticker, regime, L, H_total, strategy, step).is_file()

    def save(
        self,
        ticker: str,
        regime: str,
        L: int,
        H_total: int,
        strategy: str,
        step: int,
        X_train,
        Y_train,
        anchors_train,
        dates_train,
        X_val,
        Y_val,
        anchors_val,
        dates_val,
        X_test,
        Y_test,
        anchors_test,
        dates_test,
    ) -> Path:
        """Save all splits to a .npz file.

        ``dates_*`` arrays are converted to int64 before saving so that numpy's
        .npz format can handle them.  They are restored to ``datetime64[ns]``
        by :meth:`load`.

        Returns
        -------
        Path
            The path of the saved cache file.
        """
        self._validate_common_params(L, H_total, strategy, step)
        self._validate_split_lengths("train", X_train, Y_train, anchors_train, dates_train)
        self._validate_split_lengths("val", X_val, Y_val, anchors_val, dates_val)
        self._validate_split_lengths("test", X_test, Y_test, anchors_test, dates_test)

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self._path(ticker, regime, L, H_total, strategy, step)
        lock_path = path.with_suffix(".lock")
        # np.savez_compressed appends ".npz" when the path doesn't already end
        # with ".npz".  We pass a stem name and numpy creates <stem>.npz.
        tmp_stem = path.parent / ("." + path.stem + ".tmp")
        tmp_actual = Path(str(tmp_stem) + ".npz")  # the file numpy will actually create

        # Acquire a file-system lock to prevent concurrent writes
        self._acquire_lock(lock_path)
        try:
            # Write to a temporary file first, then rename atomically
            np.savez_compressed(
                tmp_stem,
                X_train=np.asarray(X_train),
                Y_train=np.asarray(Y_train),
                anchors_train=np.asarray(anchors_train),
                dates_train=np.asarray(dates_train, dtype="datetime64[ns]").view("int64"),
                X_val=np.asarray(X_val),
                Y_val=np.asarray(Y_val),
                anchors_val=np.asarray(anchors_val),
                dates_val=np.asarray(dates_val, dtype="datetime64[ns]").view("int64"),
                X_test=np.asarray(X_test),
                Y_test=np.asarray(Y_test),
                anchors_test=np.asarray(anchors_test),
                dates_test=np.asarray(dates_test, dtype="datetime64[ns]").view("int64"),
            )
            # Atomic rename: either the full file exists or nothing
            os.replace(tmp_actual, path)
        except Exception:
            tmp_actual.unlink(missing_ok=True)
            raise
        finally:
            self._release_lock(lock_path)

        return path

    # ------------------------------------------------------------------
    # Lock helpers
    # ------------------------------------------------------------------

    _LOCK_TIMEOUT_SECONDS: float = 60.0
    _LOCK_RETRY_INTERVAL: float = 0.5
    _LOCK_MAX_RETRIES: int = 10

    def _acquire_lock(self, lock_path: Path) -> None:
        """Wait until a .lock file can be created, or steal a stale one."""
        for _ in range(self._LOCK_MAX_RETRIES):
            if lock_path.exists():
                age = time.time() - lock_path.stat().st_mtime
                if age > self._LOCK_TIMEOUT_SECONDS:
                    logger.warning(
                        f"Stale lock detected at {lock_path} (age={age:.0f}s). Removing."
                    )
                    lock_path.unlink(missing_ok=True)
                else:
                    time.sleep(self._LOCK_RETRY_INTERVAL)
                    continue
            try:
                lock_path.touch(exist_ok=False)
                return
            except FileExistsError:
                time.sleep(self._LOCK_RETRY_INTERVAL)
        # Last attempt: proceed anyway to avoid deadlock
        lock_path.touch(exist_ok=True)

    def _release_lock(self, lock_path: Path) -> None:
        lock_path.unlink(missing_ok=True)

    def load(
        self,
        ticker: str,
        regime: str,
        L: int,
        H_total: int,
        strategy: str = "mimo",
        step: int = 0,
    ) -> dict:
        """Load a cached window dataset.

        Returns
        -------
        dict
            Keys: ``X_train``, ``Y_train``, ``anchors_train``, ``dates_train``,
            ``X_val``, ``Y_val``, ``anchors_val``, ``dates_val``,
            ``X_test``, ``Y_test``, ``anchors_test``, ``dates_test``.

        Raises
        ------
        FileNotFoundError
            If the cache entry does not exist.  Call :meth:`exists` first to
            avoid this.
        """
        self._validate_common_params(L, H_total, strategy, step)

        path = self._path(ticker, regime, L, H_total, strategy, step)
        if not path.exists():
            raise FileNotFoundError(
                f"Cache miss for key '{self.key(ticker, regime, L, H_total, strategy, step)}'. "
                f"Expected file: {path}"
            )

        required_keys = {
            "X_train",
            "Y_train",
            "anchors_train",
            "dates_train",
            "X_val",
            "Y_val",
            "anchors_val",
            "dates_val",
            "X_test",
            "Y_test",
            "anchors_test",
            "dates_test",
        }

        try:
            raw = np.load(path, allow_pickle=False)
        except Exception as exc:
            logger.warning(
                f"Cache file {path} is corrupted ({exc}). Removing and treating as cache miss."
            )
            path.unlink(missing_ok=True)
            raise FileNotFoundError(
                f"Cache miss (corrupted file removed) for key "
                f"'{self.key(ticker, regime, L, H_total, strategy, step)}'."
            ) from exc

        with raw as data:
            missing = required_keys.difference(data.files)
            if missing:
                raise ValueError(
                    f"Corrupted or incomplete cache file: {path}. Missing keys: {sorted(missing)}"
                )

            result = {
                "X_train": data["X_train"],
                "Y_train": data["Y_train"],
                "anchors_train": data["anchors_train"],
                "dates_train": data["dates_train"].view("datetime64[ns]"),
                "X_val": data["X_val"],
                "Y_val": data["Y_val"],
                "anchors_val": data["anchors_val"],
                "dates_val": data["dates_val"].view("datetime64[ns]"),
                "X_test": data["X_test"],
                "Y_test": data["Y_test"],
                "anchors_test": data["anchors_test"],
                "dates_test": data["dates_test"].view("datetime64[ns]"),
            }

        self._validate_split_lengths(
            "train",
            result["X_train"],
            result["Y_train"],
            result["anchors_train"],
            result["dates_train"],
        )
        self._validate_split_lengths(
            "val",
            result["X_val"],
            result["Y_val"],
            result["anchors_val"],
            result["dates_val"],
        )
        self._validate_split_lengths(
            "test",
            result["X_test"],
            result["Y_test"],
            result["anchors_test"],
            result["dates_test"],
        )

        return result

    def _validate_common_params(
        self,
        L: int,
        H_total: int,
        strategy: str,
        step: int,
    ) -> None:
        """Validate cache parameters shared across methods."""
        if not isinstance(L, int) or L <= 0:
            raise ValueError(f"L must be a positive int, got {L!r}")
        if not isinstance(H_total, int) or H_total <= 0:
            raise ValueError(f"H_total must be a positive int, got {H_total!r}")
        if not isinstance(strategy, str) or strategy.lower() not in self._VALID_STRATEGIES:
            raise ValueError(
                f"Invalid strategy {strategy!r}. Expected one of: {sorted(self._VALID_STRATEGIES)}"
            )
        if not isinstance(step, int) or step < 0:
            raise ValueError(f"step must be an int >= 0, got {step!r}")
        if strategy.lower() != "recursive" and step != 0:
            raise ValueError(
                f"step only applies when strategy='recursive'. Got strategy={strategy!r}, step={step!r}"
            )

    def _validate_split_lengths(
        self,
        split_name: str,
        X,
        Y,
        anchors,
        dates,
    ) -> None:
        """Validate that all arrays in a split have the same number of samples."""
        len_x = len(X)
        len_y = len(Y)
        len_anchors = len(anchors)
        len_dates = len(dates)

        if not (len_x == len_y == len_anchors == len_dates):
            raise ValueError(
                f"Inconsistent lengths in {split_name} split: "
                f"len(X)={len_x}, len(Y)={len_y}, len(anchors)={len_anchors}, len(dates)={len_dates}"
            )

    def _sanitize_part(self, value: str) -> str:
        """Sanitize a filename component."""
        text = str(value).strip()
        text = text.replace(".", "_")
        text = text.replace("/", "_")
        text = text.replace("\\", "_")
        text = text.replace(" ", "_")
        return text