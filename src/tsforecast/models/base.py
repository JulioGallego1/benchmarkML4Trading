from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class BaseModel(ABC):
    """Abstract interface for forecasting models. Subclasses must implement: fit, predict, save, load."""

    _VALID_STRATEGIES: frozenset[str] = frozenset({"mimo", "recursive"})

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        Y_val: np.ndarray | None = None,
    ) -> None: ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def save(self, path: Path) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "BaseModel": ...

    # Shared validation used by all PyTorch subclasses

    def _validate_strategy(self, strategy: str, step: int, context_length: int, horizon: int) -> None:
        if strategy not in self._VALID_STRATEGIES:
            raise ValueError(f"Invalid strategy {strategy!r}. Expected one of: {sorted(self._VALID_STRATEGIES)}.")
        if strategy == "recursive" and (not isinstance(step, int) or step <= 0):
            raise ValueError(f"step must be a positive integer, got {step!r}.")
        if not isinstance(context_length, int) or context_length <= 0:
            raise ValueError(f"context_length must be a positive integer, got {context_length!r}.")
        if not isinstance(horizon, int) or horizon <= 0:
            raise ValueError(f"horizon must be a positive integer, got {horizon!r}.")

    def _validate_fit_arrays(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_val: np.ndarray | None,
        Y_val: np.ndarray | None,
    ) -> None:
        X_train, Y_train = np.asarray(X_train), np.asarray(Y_train)
        if X_train.ndim != 2:
            raise ValueError(f"X_train must be 2D with shape (N, L), got {X_train.shape}.")
        if Y_train.ndim != 2:
            raise ValueError(f"Y_train must be 2D with shape (N, H), got {Y_train.shape}.")
        if X_train.shape[0] != Y_train.shape[0]:
            raise ValueError(
                f"X_train and Y_train must have the same number of samples, "
                f"got {X_train.shape[0]} and {Y_train.shape[0]}."
            )
        if X_train.shape[1] != self.context_length:
            raise ValueError(f"X_train must have shape (N, {self.context_length}), got {X_train.shape}.")
        if (X_val is None) != (Y_val is None):
            raise ValueError("X_val and Y_val must either both be provided or both be None.")
        if X_val is not None and Y_val is not None:
            X_val, Y_val = np.asarray(X_val), np.asarray(Y_val)
            if X_val.ndim != 2:
                raise ValueError(f"X_val must be 2D with shape (N, L), got {X_val.shape}.")
            if Y_val.ndim != 2:
                raise ValueError(f"Y_val must be 2D with shape (N, H), got {Y_val.shape}.")
            if X_val.shape[0] != Y_val.shape[0]:
                raise ValueError(
                    f"X_val and Y_val must have the same number of samples, "
                    f"got {X_val.shape[0]} and {Y_val.shape[0]}."
                )
            if X_val.shape[1] != self.context_length:
                raise ValueError(f"X_val must have shape (N, {self.context_length}), got {X_val.shape}.")
