from __future__ import annotations

import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tsforecast.training.callbacks import EarlyStopping, Checkpoint

logger = logging.getLogger(__name__)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """One training epoch. Returns mean loss over all batches.

    Supports variable-arity batches: every tensor except the last is forwarded
    as positional inputs to the model; the last tensor is the target.

      2-tensor batch  (X, Y)      → model(X),     criterion(pred, Y)
      3-tensor batch  (X, T, Y)   → model(X, T),  criterion(pred, Y)
    """
    model.train()
    total_loss = 0.0
    for batch in loader:
        *inputs, Y_batch = [t.to(device) for t in batch]
        optimizer.zero_grad()
        preds = model(*inputs)
        loss = criterion(preds, Y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """One validation epoch. Returns mean loss over all batches.

    Supports the same variable-arity batch convention as train_epoch.
    """
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            *inputs, Y_batch = [t.to(device) for t in batch]
            preds = model(*inputs)
            loss = criterion(preds, Y_batch)
            total_loss += loss.item()
    return total_loss / len(loader)


def fit_pytorch(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scheduler,
    early_stopping: EarlyStopping,
    checkpoint: Checkpoint,
    max_epochs: int,
    device: torch.device,
    use_gpu: bool = True,
) -> dict:
    """Full training loop for PyTorch models.

    Returns a dict with keys:
        - train_losses: list of float (one per epoch)
        - val_losses: list of float (one per epoch)
        - best_epoch: int (0-indexed epoch with best val loss)
        - stopped_early: bool
    """
    logger.info(f"Training on device: {device}")
    model.to(device)
    train_losses: list[float] = []
    val_losses: list[float] = []
    best_epoch = 0
    stopped_early = False

    for epoch in range(max_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = eval_epoch(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if scheduler is not None:
            scheduler.step(val_loss)

        if checkpoint.update(model, val_loss):
            best_epoch = epoch

        if early_stopping.step(val_loss):
            stopped_early = True
            break

    checkpoint.load_best(model)

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_epoch": best_epoch,
        "stopped_early": stopped_early,
    }
