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
from transformers import PatchTSTConfig, PatchTSTForPrediction, set_seed

from tsforecast.models.base import BaseModel
from tsforecast.training.callbacks import Checkpoint, EarlyStopping


class PatchTSTModel(BaseModel):
    """PatchTST forecaster supporting MIMO and recursive strategies."""

    def __init__(
        self,
        context_length: int,
        horizon: int,
        patch_length: int = 16,
        patch_stride: int = 8,
        d_model: int = 128,
        num_attention_heads: int = 4,
        num_hidden_layers: int = 3,
        ffn_dim: int = 256,
        dropout: float = 0.2,
        lr: float = 1e-4,
        max_epochs: int = 100,
        batch_size: int = 32,
        patience: int = 10,
        random_state: int = 2024,
        output_dir: str = "runs/patchtst_tmp",
        strategy: str = "mimo",
        step: int = 16,
        use_revin: bool = True,
    ) -> None:
        self.context_length = context_length
        self.horizon = horizon
        self.patch_length = patch_length
        self.patch_stride = patch_stride
        self.d_model = d_model
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.lr = lr
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.random_state = random_state
        self.output_dir = output_dir
        self.strategy = strategy
        self.step = step
        self.use_revin = use_revin
        self._model: PatchTSTForPrediction | None = None

    def fit(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        Y_val: np.ndarray | None = None,
    ) -> None:
        self._validate_strategy(self.strategy, self.step, self.context_length, self.horizon)
        self._validate_fit_arrays(X_train, Y_train, X_val, Y_val)

        set_seed(self.random_state)
        train_horizon = self.step if self.strategy == "recursive" else self.horizon

        if Y_train.shape[1] < train_horizon:
            raise ValueError(f"Y_train must have at least {train_horizon} target steps, got shape {Y_train.shape}.")
        if Y_val is not None and Y_val.shape[1] < train_horizon:
            raise ValueError(f"Y_val must have at least {train_horizon} target steps, got shape {Y_val.shape}.")

        Y_tr = Y_train[:, :train_horizon]
        Y_v = Y_val[:, :train_horizon] if Y_val is not None else None

        config = PatchTSTConfig(
            num_input_channels=1,
            context_length=self.context_length,
            prediction_length=train_horizon,
            patch_length=self.patch_length,
            stride=self.patch_stride,
            d_model=self.d_model,
            num_attention_heads=self.num_attention_heads,
            num_hidden_layers=self.num_hidden_layers,
            ffn_dim=self.ffn_dim,
            dropout=self.dropout,
            scaling="std" if self.use_revin else None,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        backbone = PatchTSTForPrediction(config).to(device)

        # X needs the channel dim: (N, L) → (N, L, 1)
        def make_loader(X, Y, shuffle):
            X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
            Y_t = torch.tensor(Y, dtype=torch.float32).unsqueeze(-1)
            return DataLoader(TensorDataset(X_t, Y_t), batch_size=self.batch_size, shuffle=shuffle)

        train_loader = make_loader(X_train, Y_tr, shuffle=True)
        val_loader = make_loader(X_val, Y_v, shuffle=False) if X_val is not None and Y_v is not None else None

        optimizer = AdamW(backbone.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
        early_stopping = EarlyStopping(patience=self.patience)
        criterion = nn.MSELoss()

        train_losses: list[float] = []
        val_losses: list[float] = []
        best_epoch = 0
        stopped_early = False

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint = Checkpoint(path=Path(tmp_dir) / "best.pt")

            for epoch in range(self.max_epochs):
                backbone.train()
                total_train = 0.0
                for x_b, y_b in train_loader:
                    x_b, y_b = x_b.to(device), y_b.to(device)
                    optimizer.zero_grad()
                    out = backbone(past_values=x_b)
                    loss = criterion(out.prediction_outputs, y_b)
                    loss.backward()
                    optimizer.step()
                    total_train += loss.item()
                train_losses.append(total_train / len(train_loader))

                val_loss = float("inf")
                if val_loader is not None:
                    backbone.eval()
                    total_val = 0.0
                    with torch.no_grad():
                        for x_b, y_b in val_loader:
                            x_b, y_b = x_b.to(device), y_b.to(device)
                            out = backbone(past_values=x_b)
                            total_val += criterion(out.prediction_outputs, y_b).item()
                    val_loss = total_val / len(val_loader)
                val_losses.append(val_loss)

                scheduler.step(val_loss)
                if checkpoint.update(backbone, val_loss):
                    best_epoch = epoch
                if early_stopping.step(val_loss):
                    stopped_early = True
                    break

            checkpoint.load_best(backbone)

        self.history = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_epoch": best_epoch,
            "stopped_early": stopped_early,
        }
        self._model = backbone

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"X must be a 2D array of shape (N, L), got shape {X.shape}.")
        if X.shape[1] != self.context_length:
            raise ValueError(f"X must have shape (N, {self.context_length}), got {X.shape}.")

        if self.strategy == "recursive":
            return self.predict_recursive(X, total_horizon=self.horizon, step=self.step)

        device = next(self._model.parameters()).device
        self._model.eval()
        results = []
        with torch.no_grad():
            for i in range(0, len(X), self.batch_size):
                batch = torch.tensor(X[i : i + self.batch_size], dtype=torch.float32).unsqueeze(-1).to(device)
                out = self._model(past_values=batch).prediction_outputs  # (B, H, 1)
                results.append(out.squeeze(-1).cpu().numpy())
        return np.concatenate(results, axis=0).astype(np.float32)

    def predict_recursive(self, X: np.ndarray, total_horizon: int, step: int) -> np.ndarray:
        """Iterative forecasting: predict ``step`` steps at a time until ``total_horizon``."""
        if self._model is None:
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

        device = next(self._model.parameters()).device
        self._model.eval()
        all_results = []
        with torch.no_grad():
            for i in range(0, len(X), self.batch_size):
                x_b = torch.tensor(X[i : i + self.batch_size], dtype=torch.float32)
                batch_results = []
                remaining = total_horizon
                while remaining > 0:
                    block_size = min(step, remaining)
                    preds = self._model(past_values=x_b.unsqueeze(-1).to(device)).prediction_outputs
                    preds = preds.squeeze(-1).cpu()                    # (B, step)
                    block = preds[:, :block_size]
                    batch_results.append(block.numpy())
                    x_b = torch.cat([x_b[:, block_size:], block], dim=1)
                    remaining -= block_size
                all_results.append(np.concatenate(batch_results, axis=1))
        return np.concatenate(all_results, axis=0).astype(np.float32)

    def save(self, path: Path) -> None:
        if self._model is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() before save().")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(path)
        hparams = {
            "context_length": self.context_length,
            "horizon": self.horizon,
            "patch_length": self.patch_length,
            "patch_stride": self.patch_stride,
            "d_model": self.d_model,
            "num_attention_heads": self.num_attention_heads,
            "num_hidden_layers": self.num_hidden_layers,
            "ffn_dim": self.ffn_dim,
            "dropout": self.dropout,
            "lr": self.lr,
            "max_epochs": self.max_epochs,
            "batch_size": self.batch_size,
            "patience": self.patience,
            "random_state": self.random_state,
            "output_dir": self.output_dir,
            "strategy": self.strategy,
            "step": self.step,
            "use_revin": self.use_revin,
        }
        (path / "hparams.json").write_text(json.dumps(hparams, indent=2))

    @classmethod
    def load(cls, path: Path) -> "PatchTSTModel":
        path = Path(path)
        hparams = json.loads((path / "hparams.json").read_text())
        required_keys = {
            "context_length", "horizon", "patch_length", "patch_stride", "d_model",
            "num_attention_heads", "num_hidden_layers", "ffn_dim", "dropout", "lr",
            "max_epochs", "batch_size", "patience", "random_state", "output_dir",
            "strategy", "step", "use_revin",
        }
        missing = required_keys.difference(hparams)
        if missing:
            raise ValueError(f"Missing required hyperparameters in hparams.json: {sorted(missing)}")
        hparams.pop("num_tickers", None)
        hparams.pop("ticker_embedding_dim", None)

        instance = cls(**hparams)
        instance._model = PatchTSTForPrediction.from_pretrained(path)
        return instance
