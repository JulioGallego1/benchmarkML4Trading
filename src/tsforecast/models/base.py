from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class BaseModel(ABC):
    """Abstract interface for forecasting models. Subclasses must implement: fit, predict, save, load."""

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        Y_val: np.ndarray | None = None,
        ticker_ids_train: np.ndarray | None = None,
        ticker_ids_val: np.ndarray | None = None,
    ) -> None:
        """Fit the model.

        Parameters
        ----------
        X_train, Y_train : np.ndarray
            Training windows.
        X_val, Y_val : np.ndarray | None
            Validation windows (used for early stopping).
        ticker_ids_train, ticker_ids_val : np.ndarray | None
            Integer ticker IDs, shape (N,).  Provided only in pooled mode so the
            model can condition predictions on asset identity.  Must be None in
            per-ticker mode — subclasses must default to identity-agnostic
            behaviour when these are absent.
        """
        ...

    @abstractmethod
    def predict(self, X: np.ndarray, ticker_ids: np.ndarray | None = None) -> np.ndarray:
        """Return predictions of shape (n_windows, horizon).

        Parameters
        ----------
        X : np.ndarray
            Context windows, shape (N, L).
        ticker_ids : np.ndarray | None
            Integer ticker IDs, shape (N,).  Required in pooled mode when the
            model was trained with ticker conditioning; None in per-ticker mode.
        """
        ...

    @abstractmethod
    def save(self, path: Path) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "BaseModel": ...
