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
    """Internal nn.Module: RevIN → stacked LSTM → Linear head → RevIN inverse.

    Input shape: (batch, L). Output shape: (batch, H, 1).
    When ``num_tickers > 1``, a learned ticker embedding is concatenated to
    each timestep after RevIN normalisation, widening the LSTM input to ``1 + E``.
    """

    def __init__(
        self,
        input_len: int,
        output_horizon: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        num_tickers: int = 1,
        ticker_embedding_dim: int = 4,
    ) -> None:
        super().__init__()
        self.num_tickers = num_tickers
        self.ticker_embedding_dim = ticker_embedding_dim if num_tickers > 1 else 0

        self.revin = _RevIN(num_features=1)

        if num_tickers > 1:
            self.ticker_emb = nn.Embedding(num_tickers, ticker_embedding_dim)

        lstm_input_size = 1 + self.ticker_embedding_dim
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, output_horizon)

    def forward(self, x: torch.Tensor, ticker_ids: torch.Tensor | None = None) -> torch.Tensor:
        """
        Parameters
        ----------
        x          : (batch, L)  — normalised return context
        ticker_ids : (batch,)    — long tensor of ticker indices, or None
        """
        x = x.unsqueeze(-1)                  # (batch, L, 1)
        x = self.revin.normalize(x)          # (batch, L, 1) — stats stored for denorm

        if self.num_tickers > 1:
            if ticker_ids is None:
                # Fallback: treat as ticker 0 (neutral embedding).
                ticker_ids = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            emb = self.ticker_emb(ticker_ids)                    # (batch, E)
            emb = emb.unsqueeze(1).expand(-1, x.size(1), -1)    # (batch, L, E)
            x = torch.cat([x, emb], dim=-1)                      # (batch, L, 1+E)

        out, _ = self.lstm(x)                # (batch, L, hidden)
        last_hidden = out[:, -1, :]          # (batch, hidden)
        pred = self.head(last_hidden)        # (batch, H)
        pred = pred.unsqueeze(-1)            # (batch, H, 1)
        pred = self.revin.denormalize(pred)  # (batch, H, 1)
        return pred


class LSTMModel(BaseModel):
    """LSTM forecaster supporting MIMO and recursive strategies.

    Ticker-aware pooled mode
    ------------------------
    Pass ``num_tickers > 1`` to enable the learned embedding table inside
    :class:`_LSTMNet`.  In :meth:`fit` supply ``ticker_ids_train`` /
    ``ticker_ids_val`` (shape ``(N,)``, dtype int64) alongside the usual X/Y
    arrays.  In :meth:`predict` supply ``ticker_ids`` of the same length as X.

    Per-ticker mode
    ---------------
    Leave ``num_tickers=1`` (default).  The ``ticker_ids_*`` arguments are
    always ``None`` in this mode and the model behaves exactly as before.
    """

    _VALID_STRATEGIES = {"mimo", "recursive"}

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
        num_tickers: int = 1,
        ticker_embedding_dim: int = 4,
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
        self.num_tickers = num_tickers
        self.ticker_embedding_dim = ticker_embedding_dim
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._net: _LSTMNet | None = None
        self._train_horizon: int = step if strategy == "recursive" else horizon

    # Internal helpers                                                         
    def _build_net(self, train_horizon: int) -> _LSTMNet:
        return _LSTMNet(
            input_len=self.context_length,
            output_horizon=train_horizon,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            num_tickers=self.num_tickers,
            ticker_embedding_dim=self.ticker_embedding_dim,
        )

    def _make_loader(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        shuffle: bool,
        ticker_ids: np.ndarray | None = None,
    ) -> DataLoader:
        X_t = torch.tensor(X, dtype=torch.float32)
        Y_t = torch.tensor(Y, dtype=torch.float32).unsqueeze(-1)  # (N, H, 1)
        if ticker_ids is not None:
            T_t = torch.tensor(ticker_ids, dtype=torch.long)      # (N,)
            dataset = TensorDataset(X_t, T_t, Y_t)
        else:
            dataset = TensorDataset(X_t, Y_t)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)


    # BaseModel interface                                                      
    def fit(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        Y_val: np.ndarray | None = None,
        ticker_ids_train: np.ndarray | None = None,
        ticker_ids_val: np.ndarray | None = None,
    ) -> None:
        self._validate_strategy_and_step()
        self._validate_fit_inputs(X_train, Y_train, X_val, Y_val)

        set_seed(self.random_state)

        # For recursive strategy, train with step-sized horizon.
        train_horizon = self.step if self.strategy == "recursive" else self.horizon
        self._train_horizon = train_horizon

        if Y_train.shape[1] < train_horizon:
            raise ValueError(
                f"Y_train must have at least {train_horizon} target steps, got shape {Y_train.shape}."
            )
        if Y_val is not None and Y_val.shape[1] < train_horizon:
            raise ValueError(
                f"Y_val must have at least {train_horizon} target steps, got shape {Y_val.shape}."
            )

        # Slice targets to train_horizon if needed.
        Y_tr = Y_train[:, :train_horizon]
        Y_v = Y_val[:, :train_horizon] if Y_val is not None else None

        train_loader = self._make_loader(X_train, Y_tr, shuffle=True,
                                         ticker_ids=ticker_ids_train)
        val_loader = (
            self._make_loader(X_val, Y_v, shuffle=False, ticker_ids=ticker_ids_val)
            if X_val is not None and Y_v is not None
            else None
        )

        self._net = self._build_net(train_horizon)
        self._net.to(self.device)
        optimizer = AdamW(self._net.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
        early_stopping = EarlyStopping(patience=self.patience)

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint = Checkpoint(path=Path(tmp_dir) / "best.pt")
            self.history = fit_pytorch(
                model=self._net,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                criterion=criterion,
                scheduler=scheduler,
                early_stopping=early_stopping,
                checkpoint=checkpoint,
                max_epochs=self.max_epochs,
                device=self.device,
            )

    def predict(self, X: np.ndarray, ticker_ids: np.ndarray | None = None) -> np.ndarray:
        if self._net is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"X must be a 2D array of shape (N, L), got shape {X.shape}.")
        if X.shape[1] != self.context_length:
            raise ValueError(
                f"X must have shape (N, {self.context_length}), got {X.shape}."
            )

        if self.strategy == "recursive":
            return self.predict_recursive(
                X, total_horizon=self.horizon, step=self.step, ticker_ids=ticker_ids
            )

        self._net.eval()
        X_t = torch.tensor(X, dtype=torch.float32)
        T_t = torch.tensor(ticker_ids, dtype=torch.long) if ticker_ids is not None else None
        results = []
        with torch.no_grad():
            for i in range(0, len(X_t), self.batch_size):
                x_b = X_t[i : i + self.batch_size].to(self.device)
                t_b = T_t[i : i + self.batch_size].to(self.device) if T_t is not None else None
                preds = self._net(x_b, t_b)      # (B, H, 1)
                results.append(preds.squeeze(-1).cpu().numpy())
        return np.concatenate(results, axis=0).astype(np.float32)

    def predict_recursive(
        self,
        X: np.ndarray,
        total_horizon: int,
        step: int,
        ticker_ids: np.ndarray | None = None,
    ) -> np.ndarray:
        """Iterative forecasting: predict ``step`` steps at a time until ``total_horizon``.

        Parameters
        ----------
        X : np.ndarray
            Context windows of shape (N, L).
        total_horizon : int
            Total number of future steps to predict.
        step : int
            Number of steps predicted per iteration (must match training horizon).
        ticker_ids : np.ndarray | None
            Integer ticker IDs of shape (N,).  Constant across iterations
            because the same asset is being predicted.

        Returns
        -------
        np.ndarray
            Predictions of shape (N, total_horizon).
        """
        if self._net is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"X must be a 2D array of shape (N, L), got shape {X.shape}.")
        if X.shape[1] != self.context_length:
            raise ValueError(
                f"X must have shape (N, {self.context_length}), got {X.shape}."
            )
        if total_horizon <= 0:
            raise ValueError(f"total_horizon must be > 0, got {total_horizon}.")
        if step <= 0:
            raise ValueError(f"step must be > 0, got {step}.")

        self._net.eval()

        N = len(X)
        all_results = []
        with torch.no_grad():
            for i in range(0, N, self.batch_size):
                x_b = torch.tensor(X[i : i + self.batch_size], dtype=torch.float32).to(self.device)
                t_b = (
                    torch.tensor(ticker_ids[i : i + self.batch_size], dtype=torch.long).to(self.device)
                    if ticker_ids is not None
                    else None
                )
                batch_results = []
                remaining = total_horizon
                while remaining > 0:
                    block_size = min(step, remaining)
                    preds = self._net(x_b, t_b)          # (B, step, 1)
                    preds_2d = preds.squeeze(-1)          # (B, step)
                    block = preds_2d[:, :block_size]      # (B, block_size)
                    batch_results.append(block.cpu().numpy())
                    # Roll context window forward.
                    x_b = torch.cat([x_b[:, block_size:], block], dim=1)  # (B, L)
                    remaining -= block_size
                all_results.append(np.concatenate(batch_results, axis=1))

        return np.concatenate(all_results, axis=0).astype(np.float32)  # (N, total_horizon)


    # Persistence                                                              
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
            "num_tickers": self.num_tickers,
            "ticker_embedding_dim": self.ticker_embedding_dim,
        }
        (path / "config.json").write_text(json.dumps(config, indent=2))

    @classmethod
    def load(cls, path: Path) -> "LSTMModel":
        path = Path(path)
        config = json.loads((path / "config.json").read_text())

        required_keys = {
            "context_length",
            "horizon",
            "hidden_size",
            "num_layers",
            "dropout",
            "lr",
            "max_epochs",
            "batch_size",
            "patience",
            "random_state",
            "device",
            "strategy",
            "step",
        }
        missing = required_keys.difference(config)
        if missing:
            raise ValueError(f"Missing required keys in config.json: {sorted(missing)}")

        # num_tickers / ticker_embedding_dim are absent in checkpoints saved
        # before the ticker-aware update — default to the single-ticker case.
        config.setdefault("num_tickers", 1)
        config.setdefault("ticker_embedding_dim", 4)

        instance = cls(**config)
        train_horizon = instance.step if instance.strategy == "recursive" else instance.horizon
        instance._net = instance._build_net(train_horizon)
        instance._net.load_state_dict(
            torch.load(path / "model_state.pt", map_location=instance.device)
        )
        instance._net.to(instance.device)
        instance._net.eval()
        return instance


    # Validation helpers                                                       
    def _validate_strategy_and_step(self) -> None:
        if self.strategy not in self._VALID_STRATEGIES:
            raise ValueError(
                f"Invalid strategy {self.strategy!r}. Expected one of: {sorted(self._VALID_STRATEGIES)}."
            )
        if self.strategy == "recursive":
            if not isinstance(self.step, int) or self.step <= 0:
                raise ValueError(f"step must be a positive integer, got {self.step!r}.")
        if not isinstance(self.context_length, int) or self.context_length <= 0:
            raise ValueError(
                f"context_length must be a positive integer, got {self.context_length!r}."
            )
        if not isinstance(self.horizon, int) or self.horizon <= 0:
            raise ValueError(f"horizon must be a positive integer, got {self.horizon!r}.")
        if not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise ValueError(f"batch_size must be a positive integer, got {self.batch_size!r}.")

    def _validate_fit_inputs(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_val: np.ndarray | None,
        Y_val: np.ndarray | None,
    ) -> None:
        X_train = np.asarray(X_train)
        Y_train = np.asarray(Y_train)

        if X_train.ndim != 2:
            raise ValueError(f"X_train must be 2D with shape (N, L), got {X_train.shape}.")
        if Y_train.ndim != 2:
            raise ValueError(f"Y_train must be 2D with shape (N, H), got {Y_train.shape}.")
        if X_train.shape[0] != Y_train.shape[0]:
            raise ValueError(
                f"X_train and Y_train must have the same number of samples, got "
                f"{X_train.shape[0]} and {Y_train.shape[0]}."
            )
        if X_train.shape[1] != self.context_length:
            raise ValueError(
                f"X_train must have shape (N, {self.context_length}), got {X_train.shape}."
            )

        if (X_val is None) != (Y_val is None):
            raise ValueError("X_val and Y_val must either both be provided or both be None.")

        if X_val is not None and Y_val is not None:
            X_val = np.asarray(X_val)
            Y_val = np.asarray(Y_val)

            if X_val.ndim != 2:
                raise ValueError(f"X_val must be 2D with shape (N, L), got {X_val.shape}.")
            if Y_val.ndim != 2:
                raise ValueError(f"Y_val must be 2D with shape (N, H), got {Y_val.shape}.")
            if X_val.shape[0] != Y_val.shape[0]:
                raise ValueError(
                    f"X_val and Y_val must have the same number of samples, got "
                    f"{X_val.shape[0]} and {Y_val.shape[0]}."
                )
            if X_val.shape[1] != self.context_length:
                raise ValueError(
                    f"X_val must have shape (N, {self.context_length}), got {X_val.shape}."
                )
