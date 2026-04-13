from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from tsforecast.models.base import BaseModel
from tsforecast.training.callbacks import Checkpoint, EarlyStopping
from tsforecast.training.engine import fit_pytorch
from tsforecast.training.reproducibility import set_seed


class _RevIN(nn.Module):
    """Reversible Instance Normalization applied per-instance along the time axis."""

    def __init__(self, num_features: int = 1, eps: float = 1e-5, affine: bool = True) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def _get_statistics(self, x: torch.Tensor) -> None:
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        self._get_statistics(x)
        x = (x - self.mean) / self.stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps)
        x = x * self.stdev + self.mean
        return x


class _LSTMNet(nn.Module):
    """RevIN → stacked LSTM → Linear head → RevIN inverse. Input: (B, L). Output: (B, H, 1)."""

    def __init__(self, output_horizon: int, hidden_size: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.revin = _RevIN(num_features=1)
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, output_horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)                  # (B, L, 1)
        x = self.revin.normalize(x)          # stats stored for denorm
        out, _ = self.lstm(x)                # (B, L, hidden)
        pred = self.head(out[:, -1, :])      # (B, H)
        pred = pred.unsqueeze(-1)            # (B, H, 1)
        return self.revin.denormalize(pred)  # (B, H, 1) raw prices


class LSTMModel(BaseModel):
    """LSTM forecaster supporting MIMO and recursive strategies."""

    def __init__(
        self,
        context_length: int,
        horizon: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        lr: float = 1e-3,
        max_epochs: int = 50,
        batch_size: int = 32,
        patience: int = 8,
        random_state: int = 2024,
        device: str | None = None,
        strategy: str = "mimo",
        step: int = 16,
    ) -> None:
        self.context_length = context_length
        self.horizon = horizon
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.random_state = random_state
        self.strategy = strategy
        self.step = step
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self._net: _LSTMNet | None = None

    def fit(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        Y_val: np.ndarray | None = None,
    ) -> None:
        self._validate_strategy(self.strategy, self.step, self.context_length, self.horizon)
        if not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise ValueError(f"batch_size must be a positive integer, got {self.batch_size!r}.")
        self._validate_fit_arrays(X_train, Y_train, X_val, Y_val)

        set_seed(self.random_state)
        train_horizon = self.step if self.strategy == "recursive" else self.horizon

        if Y_train.shape[1] < train_horizon:
            raise ValueError(f"Y_train must have at least {train_horizon} target steps, got shape {Y_train.shape}.")
        if Y_val is not None and Y_val.shape[1] < train_horizon:
            raise ValueError(f"Y_val must have at least {train_horizon} target steps, got shape {Y_val.shape}.")

        Y_tr = Y_train[:, :train_horizon]
        Y_v = Y_val[:, :train_horizon] if Y_val is not None else None

        def make_loader(X, Y, shuffle):
            X_t = torch.tensor(X, dtype=torch.float32)
            Y_t = torch.tensor(Y, dtype=torch.float32).unsqueeze(-1)
            return DataLoader(TensorDataset(X_t, Y_t), batch_size=self.batch_size, shuffle=shuffle)

        train_loader = make_loader(X_train, Y_tr, shuffle=True)
        val_loader = make_loader(X_val, Y_v, shuffle=False) if X_val is not None and Y_v is not None else None

        self._net = _LSTMNet(train_horizon, self.hidden_size, self.num_layers, self.dropout)
        self._net.to(self.device)
        optimizer = AdamW(self._net.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

        with tempfile.TemporaryDirectory() as tmp_dir:
            self.history = fit_pytorch(
                model=self._net,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                criterion=criterion,
                scheduler=scheduler,
                early_stopping=EarlyStopping(patience=self.patience),
                checkpoint=Checkpoint(path=Path(tmp_dir) / "best.pt"),
                max_epochs=self.max_epochs,
                device=self.device,
            )

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._net is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"X must be a 2D array of shape (N, L), got shape {X.shape}.")
        if X.shape[1] != self.context_length:
            raise ValueError(f"X must have shape (N, {self.context_length}), got {X.shape}.")

        if self.strategy == "recursive":
            return self.predict_recursive(X, total_horizon=self.horizon, step=self.step)

        self._net.eval()
        results = []
        with torch.no_grad():
            for i in range(0, len(X), self.batch_size):
                x_b = torch.tensor(X[i : i + self.batch_size], dtype=torch.float32).to(self.device)
                preds = self._net(x_b)  # (B, H, 1) raw prices (forward includes denorm)
                results.append(preds.squeeze(-1).cpu().numpy())
        return np.concatenate(results, axis=0).astype(np.float32)

    def predict_recursive(self, X: np.ndarray, total_horizon: int, step: int) -> np.ndarray:
        """Iterative forecasting: predict ``step`` steps at a time until ``total_horizon``."""
        if self._net is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"X must be a 2D array of shape (N, L), got shape {X.shape}.")
        if X.shape[1] != self.context_length:
            raise ValueError(f"X must have shape (N, {self.context_length}), got {X.shape}.")
        if total_horizon <= 0:
            raise ValueError(f"total_horizon must be > 0, got {total_horizon}.")
        if step <= 0:
            raise ValueError(f"step must be > 0, got {step}.")

        self._net.eval()
        all_results = []
        with torch.no_grad():
            for i in range(0, len(X), self.batch_size):
                x_b = torch.tensor(X[i : i + self.batch_size], dtype=torch.float32).to(self.device)
                batch_results = []
                remaining = total_horizon
                while remaining > 0:
                    block_size = min(step, remaining)
                    # forward() normalizes x_b, runs LSTM, then denormalizes — all with
                    # the same per-instance stats from this step's context.  The
                    # normalize/denormalize pair is always self-consistent inside one call.
                    preds = self._net(x_b)           # (B, step, 1) raw prices
                    block = preds.squeeze(-1)[:, :block_size]  # (B, block_size)
                    batch_results.append(block.cpu().numpy())
                    x_b = torch.cat([x_b[:, block_size:], block], dim=1)
                    remaining -= block_size
                all_results.append(np.concatenate(batch_results, axis=1))
        return np.concatenate(all_results, axis=0).astype(np.float32)

    def save(self, path: Path) -> None:
        if self._net is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self._net.state_dict(), path / "model_state.pt")
        config = {
            "context_length": self.context_length,
            "horizon": self.horizon,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "lr": self.lr,
            "max_epochs": self.max_epochs,
            "batch_size": self.batch_size,
            "patience": self.patience,
            "random_state": self.random_state,
            "device": str(self.device),
            "strategy": self.strategy,
            "step": self.step,
        }
        (path / "config.json").write_text(json.dumps(config, indent=2))

    @classmethod
    def load(cls, path: Path) -> "LSTMModel":
        path = Path(path)
        config = json.loads((path / "config.json").read_text())
        required_keys = {
            "context_length", "horizon", "hidden_size", "num_layers", "dropout",
            "lr", "max_epochs", "batch_size", "patience", "random_state",
            "device", "strategy", "step",
        }
        missing = required_keys.difference(config)
        if missing:
            raise ValueError(f"Missing required keys in config.json: {sorted(missing)}")

        instance = cls(**config)
        train_horizon = instance.step if instance.strategy == "recursive" else instance.horizon
        instance._net = _LSTMNet(
            train_horizon, instance.hidden_size,
            instance.num_layers, instance.dropout,
        )
        instance._net.load_state_dict(torch.load(path / "model_state.pt", map_location=instance.device))
        instance._net.to(instance.device)
        instance._net.eval()
        return instance
