from __future__ import annotations

import random
import time


def now_ts() -> str:
    """Return a high-resolution timestamp string for use in run IDs.

    Uses millisecond precision plus a 2-digit random suffix to make
    concurrent runs on the same machine practically collision-free.

    Format: ``{milliseconds_since_epoch}{2-digit-random}``
    Example: ``"17411234567891"`` (14 characters)
    """
    ms = int(time.time() * 1000)
    rnd = random.randint(0, 99)
    return f"{ms}{rnd:02d}"


def make_run_id(
    model: str,
    regime: str,
    L: int,
    H: int,
    strategy: str = "mimo",
    step: int = 16,
    training_mode: str = "per_ticker",
    extra_tags: list[str] | None = None,
) -> str:
    """Construct a unique run identifier.

    Format (MIMO):      ``{MODEL}_mimo_{regime}_L{L}_H{H}_{training_mode}[_tag...]_{timestamp}``
    Format (recursive): ``{MODEL}_rec_step{step}_{regime}_L{L}_H{H}_{training_mode}[_tag...]_{timestamp}``

    Parameters
    ----------
    model:
        Model name (e.g. ``'RF'``, ``'LSTM'``). Will be uppercased.
    regime:
        Market regime string (e.g. ``'bear'``), used as-is.
    L:
        Context length in trading days.
    H:
        Forecast horizon in trading days.
    strategy:
        Forecasting strategy: ``'mimo'`` or ``'recursive'``.
    step:
        Block size for recursive strategy (ignored when ``strategy='mimo'``).
    training_mode:
        Training mode string (e.g. ``'per_ticker'``).
    extra_tags:
        Optional list of short strings appended between the training mode and the
        timestamp (e.g. ``['revin']`` or ``['norevin']``).

    Returns
    -------
    str
        Run identifier string.
    """
    if strategy == "recursive":
        strat_tag = f"rec_step{step}"
    else:
        strat_tag = "mimo"
    tag_suffix = ("_" + "_".join(extra_tags)) if extra_tags else ""
    return f"{model.upper()}_{strat_tag}_{regime}_L{L}_H{H}_{training_mode}{tag_suffix}_{now_ts()}"
