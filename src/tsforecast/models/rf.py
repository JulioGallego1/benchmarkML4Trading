from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from tsforecast.models.base import BaseModel


class RandomForestModel(BaseModel):
    def __init__(
        self,
        n_estimators: int = 200,
        max_features: str = "sqrt",
        max_depth: int | None = None,
        min_samples_leaf: int = 1,
        random_state: int = 2024,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self._model: RandomForestRegressor | None = None

    # BaseModel interface

    def fit(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        Y_val: np.ndarray | None = None,
    ) -> None:
        # RF has no iterative training loop, so validation data is not needed
        # for early stopping. We concatenate it with training data to maximize
        # the amount of data available for fitting.
        if X_val is not None and Y_val is not None:
            X_train = np.concatenate([X_train, X_val], axis=0)
            Y_train = np.concatenate([Y_train, Y_val], axis=0)

        self._validate_init_params()
        self._validate_fit_inputs(X_train, Y_train)

        self._model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_features=self.max_features,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self._model.fit(X_train, Y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"X must be a 2D array of shape (N, L), got shape {X.shape}.")
        if X.shape[0] == 0:
            raise ValueError("X must contain at least one sample.")

        return self._model.predict(X).astype(np.float32)

    def save(self, path: Path) -> None:
        if self._model is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        path = Path(path)
        if path.suffix == "":
            path.mkdir(parents=True, exist_ok=True)
            dest = path / "model.joblib"
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            dest = path

        joblib.dump(
            {
                "model": self._model,
            },
            dest,
        )

    @classmethod
    def load(cls, path: Path) -> "RandomForestModel":
        path = Path(path)
        src = path / "model.joblib" if path.is_dir() else path
        saved = joblib.load(src)

        if not isinstance(saved, dict) or "model" not in saved:
            raise TypeError("Saved file is not in the expected format.")
        fitted = saved["model"]

        if not isinstance(fitted, RandomForestRegressor):
            raise TypeError(
                f"Loaded object is not a RandomForestRegressor. Got {type(fitted).__name__}."
            )

        instance = cls(
            n_estimators=fitted.n_estimators,
            max_features=fitted.max_features,
            max_depth=fitted.max_depth,
            min_samples_leaf=fitted.min_samples_leaf,
            random_state=fitted.random_state,
        )
        instance._model = fitted
        return instance

    # Validation helpers

    def _validate_init_params(self) -> None:
        if not isinstance(self.n_estimators, int) or self.n_estimators <= 0:
            raise ValueError(
                f"n_estimators must be a positive integer, got {self.n_estimators!r}."
            )
        if isinstance(self.max_features, str) and not self.max_features.strip():
            raise ValueError("max_features must not be an empty string.")

    def _validate_fit_inputs(self, X_train: np.ndarray, Y_train: np.ndarray) -> None:
        X_train = np.asarray(X_train)
        Y_train = np.asarray(Y_train)

        if X_train.ndim != 2:
            raise ValueError(
                f"X_train must be a 2D array with shape (N, L), got shape {X_train.shape}."
            )
        if Y_train.ndim != 2:
            raise ValueError(
                f"Y_train must be a 2D array with shape (N, H), got shape {Y_train.shape}."
            )
        if X_train.shape[0] == 0:
            raise ValueError("X_train must contain at least one sample.")
        if Y_train.shape[0] == 0:
            raise ValueError("Y_train must contain at least one sample.")
        if X_train.shape[0] != Y_train.shape[0]:
            raise ValueError(
                f"X_train and Y_train must have the same number of samples, got "
                f"{X_train.shape[0]} and {Y_train.shape[0]}."
            )
        if X_train.shape[1] == 0:
            raise ValueError("X_train must contain at least one feature.")
        if Y_train.shape[1] == 0:
            raise ValueError("Y_train must contain at least one target step.")
