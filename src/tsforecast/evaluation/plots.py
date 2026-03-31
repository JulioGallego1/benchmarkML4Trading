from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd



_WINDOW_COLORS = [
    "#e6194b",  # red
    "#f58231",  # orange
    "#ffe119",  # yellow
    "#3cb44b",  # green
    "#42d4f4",  # cyan
    "#911eb4",  # purple
    "#f032e6",  # magenta
    "#a9a9a9",  # grey
    "#9A6324",  # brown
    "#000075",  # navy
]

# Private helpers

def _reconstruct_full_series(
    dates_pd: pd.DatetimeIndex,
    y_true: np.ndarray,
) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Build the complete true series spanning all N + H − 1 unique time points.

    ``dates_pd`` has N entries (window start dates, stride = 1).  ``y_true``
    has shape (N, H).  The step-0 column gives us N values at dates[0..N-1].
    The tail (H-1 extra values) comes from the last window's remaining steps;
    their dates are inferred via business-day offset from the last known date.
    This is for visualisation only, so bdate_range approximation is fine.
    """
    N, H = y_true.shape
    main_values = y_true[:, 0]  # shape (N,)

    if H == 1:
        return dates_pd, main_values

    tail_values = y_true[-1, 1:]          # shape (H-1,)
    tail_dates = pd.bdate_range(
        start=dates_pd[-1], periods=H
    )[1:]                                  # H-1 dates after the last known date

    full_values = np.concatenate([main_values, tail_values])
    full_dates = dates_pd.append(pd.DatetimeIndex(tail_dates))
    return full_dates, full_values


def _select_evenly_spaced_windows(
    valid_indices: np.ndarray,
    n_windows: int,
) -> np.ndarray:
    """Pick up to n_windows evenly spaced valid start indices.

    Overlap is allowed. This only spreads starts across the available range.
    """
    n_valid = len(valid_indices)
    if n_valid == 0:
        return np.array([], dtype=int)

    n_select = min(n_windows, n_valid)
    if n_select == 1:
        return np.array([valid_indices[0]], dtype=int)

    positions = np.linspace(0, n_valid - 1, n_select)
    positions = np.round(positions).astype(int)
    selected = valid_indices[positions]

    # Remove accidental duplicates caused by rounding, preserving order
    selected = pd.Index(selected).unique().to_numpy(dtype=int)
    return selected


def _adaptive_n_windows(H: int, N: int) -> int:
    """Choose a readable number of forecast windows based on horizon H.

    Desired behavior for a full test span around 126 trading days:
    - H=10  -> about 10 windows
    - H=21  -> about 7 windows
    - H=42  -> about 5-6 windows
    - H=63  -> about 5 windows

    Also capped by the number of available valid windows N.
    """
    if H <= 10:
        n_target = 10
    elif H <= 21:
        n_target = 7
    elif H <= 42:
        n_target = 6
    else:
        n_target = 5

    return min(n_target, max(1, N))


# Public 

def plot_ticker_forecast(
    dates: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ticker: str = "",
    n_windows: int | None = None,
    save_path: Path | None = None,
) -> plt.Figure:
    """Plot the full test series with evenly spaced forecast windows."""
    N, H = y_true.shape
    dates_pd = pd.DatetimeIndex(pd.to_datetime(dates))

    n_windows_resolved = _adaptive_n_windows(H, N) if n_windows is None else n_windows

    # Full series shown in black across the whole test span
    full_dates, full_values = _reconstruct_full_series(dates_pd, y_true)

    fig, ax = plt.subplots(figsize=(14, 5))

    # Select forecast windows
    selected_idxs = _select_evenly_spaced_windows(np.arange(N, dtype=int), n_windows_resolved)
    n_sel = len(selected_idxs)
    colors = [_WINDOW_COLORS[i % len(_WINDOW_COLORS)] for i in range(n_sel)]

    for color, idx in zip(colors, selected_idxs):
        # Use full_dates, not dates_pd, so each window can span the full horizon
        horizon_dates = full_dates[idx : idx + H]

        # True forecast window (colored solid)
        ax.plot(
            horizon_dates,
            y_true[idx, :H],
            color=color,
            linewidth=1.8,
            alpha=0.95,
            zorder=2,
        )

        # Predicted forecast window (same color, dashed)
        ax.plot(
            horizon_dates,
            y_pred[idx, :H],
            color=color,
            linewidth=2.0,
            linestyle="--",
            dash_capstyle="round",
            zorder=3,
        )

        # Vertical marker at window start
        ax.axvline(
            full_dates[idx],
            color=color,
            linewidth=0.8,
            linestyle=":",
            alpha=0.55,
            zorder=0,
        )

    # Ground truth backbone always black, drawn as background reference
    ax.plot(
        full_dates,
        full_values,
        color="black",
        linewidth=1.6,
        label="Ground truth (full test series)",
        zorder=1,
    )

    ax.set_xlim(full_dates[0], full_dates[-1])

    legend_handles = [
        mlines.Line2D([], [], color="black", linewidth=1.6, label="Ground truth (full test series)"),
        mlines.Line2D([], [], color="dimgray", linewidth=1.8, linestyle="-", label="True forecast window"),
        mlines.Line2D([], [], color="dimgray", linewidth=1.8, linestyle="--", label="Predicted forecast window"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc="upper left")

    ax.annotate(
        "Colors distinguish different forecast windows.",
        xy=(0.01, 0.01),
        xycoords="axes fraction",
        fontsize=7,
        color="gray",
        ha="left",
        va="bottom",
    )

    title = f"Test Forecast Windows (H={H}, {n_sel} windows shown)"
    if ticker:
        title = f"{ticker} — {title}"

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)

    return fig



def plot_ticker_returns(
    dates: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    anchors: np.ndarray,
    ticker: str = "",
    save_path: Path | None = None,
) -> plt.Figure:
    """Plot horizon-end return per window: predicted vs actual.

    For each window ``i``, computes a single return value from the anchor to the
    last step of the horizon (``H - 1``):

        real_return[i]  = (y_true[i, H-1] / anchor[i]) - 1
        pred_return[i]  = (y_pred[i, H-1] / anchor[i]) - 1

    The result is two lines across all test windows — one real, one predicted —
    showing whether the model correctly captured the net move over the full horizon.
    """
    _N, H = y_true.shape
    anchors = np.asarray(anchors, dtype=np.float64)

    real_returns = (y_true[:, H - 1].astype(np.float64) / anchors - 1.0) * 100
    pred_returns = (y_pred[:, H - 1].astype(np.float64) / anchors - 1.0) * 100

    dates_pd = pd.DatetimeIndex(pd.to_datetime(dates))

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(dates_pd, real_returns, color="steelblue", linewidth=1.5, label="Real return (horizon end)")
    ax.plot(dates_pd, pred_returns, color="tomato", linewidth=1.5, linestyle="--", label="Predicted return (horizon end)")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5, zorder=0)

    ax.set_xlim(dates_pd[0], dates_pd[-1])
    ax.legend(fontsize=9)
    ax.annotate(
        f"Return = (price[H-1] / anchor) − 1,  H={H}",
        xy=(0.01, 0.01),
        xycoords="axes fraction",
        fontsize=7,
        color="gray",
        ha="left",
        va="bottom",
    )

    title = f"Horizon-end Return per Window (H={H})"
    if ticker:
        title = f"{ticker} — {title}"

    ax.set_title(title)
    ax.set_xlabel("Window start date")
    ax.set_ylabel("Return from anchor (%)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)

    return fig


def plot_training_curves(
    history: dict,
    save_path: Path | None = None,
) -> plt.Figure:
    """Plot train and val loss curves over epochs. Mark best_epoch if present."""
    train_losses = history.get("train_losses", [])
    val_losses = history.get("val_losses", [])
    best_epoch = history.get("best_epoch", None)

    fig, ax = plt.subplots(figsize=(8, 4))

    epochs = range(1, len(train_losses) + 1)
    if train_losses:
        ax.plot(epochs, train_losses, label="Train loss", linewidth=1.5, color="steelblue")
    if val_losses:
        val_epochs = range(1, len(val_losses) + 1)
        ax.plot(val_epochs, val_losses, label="Val loss", linewidth=1.5, color="tomato")

    if best_epoch is not None:
        ax.axvline(
            x=best_epoch + 1,
            color="green",
            linestyle=":",
            linewidth=1.5,
            label=f"Best epoch ({best_epoch + 1})",
        )

    ax.set_title("Training and Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)

    return fig

