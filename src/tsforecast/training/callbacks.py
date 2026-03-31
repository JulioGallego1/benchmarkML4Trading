from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(self, patience: int = 10, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss: float = float("inf")
        self._no_improve: int = 0

    def step(self, val_loss: float) -> bool:
        """Return True if training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self._no_improve = 0
            return False
        self._no_improve += 1
        return self._no_improve >= self.patience


class Checkpoint:
    """Save and restore the best model weights during training."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.best_loss: float = float("inf")

    def update(self, model: nn.Module, val_loss: float) -> bool:
        """Save model weights if val_loss improved. Returns True if saved."""
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), self.path)
            return True
        return False

    def load_best(self, model: nn.Module) -> None:
        """Load the best saved weights back into model."""
        if self.path.exists():
            model.load_state_dict(torch.load(self.path, weights_only=True))
