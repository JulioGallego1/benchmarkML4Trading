from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, TensorDataset
from transformers import (
    PatchTSTConfig,
    PatchTSTForPrediction,
    set_seed,
)

from tsforecast.models.base import BaseModel
from tsforecast.training.callbacks import Checkpoint, EarlyStopping
from tsforecast.training.engine import fit_pytorch

class _PatchTSTAdapter(nn.Module):
    """Thin wrapper so fit_pytorch's 2-tensor convention can drive PatchTSTForPrediction.

    fit_pytorch unpacks each batch as (*inputs, Y) and calls model(*inputs).
    For the standard (non-ticker-aware) path, inputs = [X] where X: (B, L, 1).
    This adapter accepts that single positional tensor and maps it to the
    backbone's keyword argument, returning prediction_outputs: (B, H, 1).
    """

    def __init__(self, backbone: PatchTSTForPrediction) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, 1) → (B, H, 1)."""
        return self.backbone(past_values=x).prediction_outputs


#  Ticker-aware components (used only in pooled mode with num_tickers > 1)

class _TickerAwarePatchTST(nn.Module):
    """Wraps ``PatchTSTForPrediction`` with a lightweight per-ticker input conditioning.

    Architecture
    ------------
    For each sample the module:
      1. Normalises ``past_values`` per-sample over the time axis (RevIN-style,
         no learnable affine), storing ``loc`` and ``scale`` for later.
      2. Projects a learnable ticker embedding to a per-timestep bias vector and
         *adds it to the already-normalised input*.  Conditioning is therefore
         injected in the normalised space, leaving the denormalisation anchor
         (``loc = mean(past_values)``) uncontaminated by the embedding.
      3. Passes the conditioned, normalised tensor to the PatchTST backbone
         (which must be built with ``scaling=None`` so it does not re-normalise).
      4. Denormalises the backbone output with the clean ``loc`` / ``scale``.

    This mirrors the LSTM design, where the ticker embedding is concatenated
    *after* RevIN normalisation, keeping the normalisation anchor clean.

    Input / output shapes
    ---------------------
    forward(past_values, ticker_ids) → denormalised prediction tensor
      past_values : (B, L, 1)   float32  — raw prices
      ticker_ids  : (B,)        long
      returns     : (B, H, 1)   float32  — raw price predictions
    """

    def __init__(
        self,
        backbone: PatchTSTForPrediction,
        num_tickers: int,
        embedding_dim: int,
        context_length: int,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.ticker_emb = nn.Embedding(num_tickers, embedding_dim)
        # Projects the embedding to a per-timestep additive bias (L scalars).
        self.cond_proj = nn.Linear(embedding_dim, context_length, bias=False)
        self.eps = eps

    def forward(
        self,
        past_values: torch.Tensor,
        ticker_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return denormalised prediction tensor (B, H, 1)."""
        if ticker_ids is None:
            ticker_ids = torch.zeros(
                past_values.size(0), dtype=torch.long, device=past_values.device
            )

        # 1. Per-sample normalisation over the time axis — anchor is untouched
        #    by the ticker embedding.  past_values: (B, L, 1)
        loc   = past_values.mean(dim=1, keepdim=True)              # (B, 1, 1)
        scale = past_values.std(dim=1, keepdim=True) + self.eps    # (B, 1, 1)
        normalised = (past_values - loc) / scale                   # (B, L, 1)

        # 2. Ticker conditioning injected in normalised space
        emb  = self.ticker_emb(ticker_ids)           # (B, E)
        bias = self.cond_proj(emb).unsqueeze(-1)     # (B, L, 1)
        conditioned = normalised + bias              # (B, L, 1)

        # 3. Backbone forward — backbone must have scaling=None (set in fit())
        pred_norm = self.backbone(past_values=conditioned).prediction_outputs  # (B, H, 1)

        # 4. Denormalise with the clean context statistics
        return pred_norm * scale + loc               # (B, H, 1)


class _TickerAwarePatchTSTDataset(Dataset):
    """Returns (past_values, ticker_id, future_values) tuples for the custom training loop."""

    def __init__(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)  # (N, L)
        self.T = torch.tensor(T, dtype=torch.long)     # (N,)
        self.Y = torch.tensor(Y, dtype=torch.float32)  # (N, H)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return (
            self.X[idx].unsqueeze(-1),  # (L, 1)
            self.T[idx],                # scalar long tensor
            self.Y[idx].unsqueeze(-1),  # (H, 1)
        )


#  Public model wrapper                                                        
class PatchTSTModel(BaseModel):
    """PatchTST forecaster supporting MIMO and recursive strategies.

    Ticker-aware pooled mode
    ------------------------
    Pass ``num_tickers > 1`` at construction time.  In :meth:`fit` supply
    ``ticker_ids_train`` / ``ticker_ids_val`` (shape ``(N,)``, dtype int64).
    In :meth:`predict` supply ``ticker_ids`` of the same length as X.

    When ``num_tickers > 1`` the ``_TickerAwarePatchTST`` wrapper is trained
    via the custom ``fit_pytorch`` loop (same as the LSTM), using MSELoss,
    AdamW, ReduceLROnPlateau, and the shared EarlyStopping / Checkpoint logic.

    Per-ticker mode (``num_tickers == 1``, default)
    ------------------------------------------------
    Also uses the custom ``fit_pytorch`` loop, driven through ``_PatchTSTAdapter``
    which maps the 2-tensor batch convention to the backbone's ``past_values=``
    keyword interface.  Optimization methodology is identical to the LSTM.
    """

    _VALID_STRATEGIES = {"mimo", "recursive"}

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
        num_tickers: int = 1,
        ticker_embedding_dim: int = 4,
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
        self.num_tickers = num_tickers
        self.ticker_embedding_dim = ticker_embedding_dim

        self._model: PatchTSTForPrediction | None = None
        # Non-None only when num_tickers > 1 and fit() has been called.
        self._wrapper: _TickerAwarePatchTST | None = None

    # Internal: build PatchTST backbone config                                
    def _build_backbone_config(self, train_horizon: int) -> PatchTSTConfig:
        return PatchTSTConfig(
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

        # For recursive strategy, train with step-sized prediction length.
        train_horizon = self.step if self.strategy == "recursive" else self.horizon

        if Y_train.shape[1] < train_horizon:
            raise ValueError(
                f"Y_train must have at least {train_horizon} target steps, got shape {Y_train.shape}."
            )
        if Y_val is not None and Y_val.shape[1] < train_horizon:
            raise ValueError(
                f"Y_val must have at least {train_horizon} target steps, got shape {Y_val.shape}."
            )

        Y_tr = Y_train[:, :train_horizon]
        Y_v = Y_val[:, :train_horizon] if Y_val is not None else None

        if self.num_tickers > 1 and ticker_ids_train is not None:
            # Ticker-aware mode: _TickerAwarePatchTST handles explicit per-sample
            # normalisation, so the backbone must NOT apply its own internal scaling.
            backbone_cfg = self._build_backbone_config(train_horizon)
            backbone_cfg.scaling = None
            backbone = PatchTSTForPrediction(backbone_cfg)
            self._fit_ticker_aware(backbone, X_train, Y_tr, X_val, Y_v,
                                   ticker_ids_train, ticker_ids_val)
        else:
            backbone = PatchTSTForPrediction(self._build_backbone_config(train_horizon))
            self._fit_standard(backbone, X_train, Y_tr, X_val, Y_v)

    def _fit_standard(
        self,
        backbone: PatchTSTForPrediction,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_val: np.ndarray | None,
        Y_val: np.ndarray | None,
    ) -> None:
        """Custom fit_pytorch path for per-ticker / single-ticker mode.

        Uses the same optimizer (AdamW, wd=0), scheduler (ReduceLROnPlateau),
        early stopping, and checkpoint logic as the LSTM and the ticker-aware
        PatchTST path, making the optimization methodology consistent across
        all models.

        The backbone is wrapped in _PatchTSTAdapter so fit_pytorch can call
        model(x) using the 2-tensor batch convention.
        """
        self._wrapper = None
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        adapter = _PatchTSTAdapter(backbone)

        # X needs the channel dim for past_values: (N, L) → (N, L, 1)
        X_tr_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
        Y_tr_t = torch.tensor(Y_train, dtype=torch.float32).unsqueeze(-1)
        train_loader = DataLoader(
            TensorDataset(X_tr_t, Y_tr_t),
            batch_size=self.batch_size,
            shuffle=True,
        )

        val_loader = None
        if X_val is not None and Y_val is not None:
            X_v_t = torch.tensor(X_val, dtype=torch.float32).unsqueeze(-1)
            Y_v_t = torch.tensor(Y_val, dtype=torch.float32).unsqueeze(-1)
            val_loader = DataLoader(
                TensorDataset(X_v_t, Y_v_t),
                batch_size=self.batch_size,
                shuffle=False,
            )

        optimizer = AdamW(adapter.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
        early_stopping = EarlyStopping(patience=self.patience)

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint = Checkpoint(path=Path(tmp_dir) / "best.pt")
            self.history = fit_pytorch(
                model=adapter,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                criterion=criterion,
                scheduler=scheduler,
                early_stopping=early_stopping,
                checkpoint=checkpoint,
                max_epochs=self.max_epochs,
                device=device,
            )

        self._model = adapter.backbone

    def _fit_ticker_aware(
        self,
        backbone: PatchTSTForPrediction,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_val: np.ndarray | None,
        Y_val: np.ndarray | None,
        ticker_ids_train: np.ndarray,
        ticker_ids_val: np.ndarray | None,
    ) -> None:
        """Custom fit_pytorch path for ticker-aware pooled mode.

        Trains a :class:`_TickerAwarePatchTST` wrapper (backbone + embedding +
        conditioning projection) using the same ``fit_pytorch`` loop as the
        LSTM, with nn.MSELoss as the criterion.  The updated engine dispatches
        ``model(past, ticker)`` → prediction tensor, so this integrates cleanly
        with the 3-tensor batch convention.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._wrapper = _TickerAwarePatchTST(
            backbone=backbone,
            num_tickers=self.num_tickers,
            embedding_dim=self.ticker_embedding_dim,
            context_length=self.context_length,
        )

        train_loader = DataLoader(
            _TickerAwarePatchTSTDataset(X_train, ticker_ids_train, Y_train),
            batch_size=self.batch_size,
            shuffle=True,
        )
        val_loader = None
        if X_val is not None and Y_val is not None and ticker_ids_val is not None:
            val_loader = DataLoader(
                _TickerAwarePatchTSTDataset(X_val, ticker_ids_val, Y_val),
                batch_size=self.batch_size,
                shuffle=False,
            )

        optimizer = AdamW(self._wrapper.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
        early_stopping = EarlyStopping(patience=self.patience)

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint = Checkpoint(path=Path(tmp_dir) / "best.pt")
            self.history = fit_pytorch(
                model=self._wrapper,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                criterion=criterion,
                scheduler=scheduler,
                early_stopping=early_stopping,
                checkpoint=checkpoint,
                max_epochs=self.max_epochs,
                device=device,
            )

        # After training, expose the backbone so save() can serialise it via
        # save_pretrained(), and also keep the wrapper for inference.
        self._model = self._wrapper.backbone


    # Prediction                                                               
    def predict(self, X: np.ndarray, ticker_ids: np.ndarray | None = None) -> np.ndarray:
        if self._model is None:
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

        if self._wrapper is not None and ticker_ids is not None:
            return self._predict_wrapper_mimo(X, ticker_ids)

        device = next(self._model.parameters()).device
        self._model.eval()
        results = []
        with torch.no_grad():
            for i in range(0, len(X), self.batch_size):
                batch = torch.tensor(X[i : i + self.batch_size], dtype=torch.float32)
                batch = batch.unsqueeze(-1).to(device)  # (B, L, 1)
                out = self._model(past_values=batch).prediction_outputs  # (B, H, 1)
                results.append(out.squeeze(-1).cpu().numpy())
        return np.concatenate(results, axis=0).astype(np.float32)

    def _predict_wrapper_mimo(self, X: np.ndarray, ticker_ids: np.ndarray) -> np.ndarray:
        """Batched MIMO predict through the _TickerAwarePatchTST wrapper."""
        device = next(self._wrapper.parameters()).device
        X_t = torch.tensor(X, dtype=torch.float32)
        T_t = torch.tensor(ticker_ids, dtype=torch.long)
        self._wrapper.eval()
        results = []
        with torch.no_grad():
            for i in range(0, len(X_t), self.batch_size):
                past = X_t[i : i + self.batch_size].unsqueeze(-1).to(device)   # (B, L, 1)
                tids = T_t[i : i + self.batch_size].to(device)
                preds = self._wrapper(past, tids)                               # (B, H, 1)
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
            Integer ticker IDs of shape (N,).  Constant across iterations.

        Returns
        -------
        np.ndarray
            Predictions of shape (N, total_horizon).
        """
        if self._model is None:
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

        use_wrapper = self._wrapper is not None and ticker_ids is not None
        if use_wrapper:
            device = next(self._wrapper.parameters()).device
            self._wrapper.eval()
        else:
            device = next(self._model.parameters()).device
            self._model.eval()

        N = len(X)
        X_t = torch.tensor(X, dtype=torch.float32)
        T_t = torch.tensor(ticker_ids, dtype=torch.long) if ticker_ids is not None else None
        all_results = []

        with torch.no_grad():
            for i in range(0, N, self.batch_size):
                x_b = X_t[i : i + self.batch_size]
                t_b = T_t[i : i + self.batch_size].to(device) if T_t is not None else None
                batch_results = []
                remaining = total_horizon
                while remaining > 0:
                    block_size = min(step, remaining)
                    past = x_b.unsqueeze(-1).to(device)           # (B, L, 1)
                    if use_wrapper:
                        preds_raw = self._wrapper(past, t_b)      # (B, step, 1)
                    else:
                        preds_raw = self._model(past_values=past).prediction_outputs  # (B, step, 1)
                    preds = preds_raw.squeeze(-1).cpu()           # (B, step)
                    block = preds[:, :block_size]                 # (B, block_size)
                    batch_results.append(block.numpy())
                    x_b = torch.cat([x_b[:, block_size:], block], dim=1)  # (B, L)
                    remaining -= block_size
                all_results.append(np.concatenate(batch_results, axis=1))

        return np.concatenate(all_results, axis=0).astype(np.float32)


    # Persistence                                                              
    def save(self, path: Path) -> None:
        if self._model is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() before save().")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Always serialise the backbone via HF save_pretrained.
        self._model.save_pretrained(path)

        # When a ticker-aware wrapper was trained, persist the extra weights
        # (embedding table + conditioning projection) separately.
        if self._wrapper is not None:
            torch.save(
                {
                    "ticker_emb": self._wrapper.ticker_emb.state_dict(),
                    "cond_proj": self._wrapper.cond_proj.state_dict(),
                },
                path / "ticker_cond.pt",
            )

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
            "num_tickers": self.num_tickers,
            "ticker_embedding_dim": self.ticker_embedding_dim,
        }
        (path / "hparams.json").write_text(json.dumps(hparams, indent=2))

    @classmethod
    def load(cls, path: Path) -> "PatchTSTModel":
        path = Path(path)
        hparams = json.loads((path / "hparams.json").read_text())

        required_keys = {
            "context_length",
            "horizon",
            "patch_length",
            "patch_stride",
            "d_model",
            "num_attention_heads",
            "num_hidden_layers",
            "ffn_dim",
            "dropout",
            "lr",
            "max_epochs",
            "batch_size",
            "patience",
            "random_state",
            "output_dir",
            "strategy",
            "step",
            "use_revin",
        }
        missing = required_keys.difference(hparams)
        if missing:
            raise ValueError(f"Missing required hyperparameters in hparams.json: {sorted(missing)}")

        # Backward compat: checkpoints saved before the ticker-aware update
        # will not have these keys — default to single-ticker behaviour.
        hparams.setdefault("num_tickers", 1)
        hparams.setdefault("ticker_embedding_dim", 4)

        instance = cls(**hparams)
        instance._model = PatchTSTForPrediction.from_pretrained(path)

        # Rebuild the wrapper and restore its conditioning weights if present.
        ticker_cond_path = path / "ticker_cond.pt"
        if ticker_cond_path.exists():
            instance._wrapper = _TickerAwarePatchTST(
                backbone=instance._model,
                num_tickers=instance.num_tickers,
                embedding_dim=instance.ticker_embedding_dim,
                context_length=instance.context_length,
            )
            state = torch.load(ticker_cond_path, map_location="cpu")
            instance._wrapper.ticker_emb.load_state_dict(state["ticker_emb"])
            instance._wrapper.cond_proj.load_state_dict(state["cond_proj"])

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
