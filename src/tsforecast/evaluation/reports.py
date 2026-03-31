from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import pandas as pd

# Run ID format: {MODEL}_{strategy}_{regime}_L{L}_H{H}_{timestamp}
# Strategy can be "mimo" or "rec_step{N}"
_RUN_ID_PATTERN = re.compile(
    r"^(?P<model>[^_]+)_(?P<strategy>mimo|rec_step\d+)_(?P<regime>.+)_L(?P<L>\d+)_H(?P<H>\d+)"
)

logger = logging.getLogger(__name__)


def _parse_run_id(run_id: str) -> dict:
    """Extract model, strategy, regime, L, H from a run_id string.

    Returns a dict with keys model, strategy, regime, L, H, or empty dict if no match.
    """
    match = _RUN_ID_PATTERN.match(run_id)
    if not match:
        return {}
    return {
        "model": match.group("model"),
        "strategy": match.group("strategy"),
        "regime": match.group("regime"),
        "L": int(match.group("L")),
        "H": int(match.group("H")),
    }


def build_summary_table(runs_dir: str | Path = "runs") -> pd.DataFrame:
    """Scan all runs/*/metrics.json files and build a summary DataFrame.

    Each row corresponds to one run. Columns come from the metrics dict plus
    metadata parsed from the run_id directory name: model, regime, L, H.

    Run ID format: {MODEL}_MIMO_{regime}_L{L}_H{H}_{timestamp}

    Returns empty DataFrame if no runs found.
    """
    runs_dir = Path(runs_dir)
    records = []

    for metrics_path in sorted(runs_dir.glob("*/metrics.json")):
        run_id = metrics_path.parent.name
        # Skip hidden directories or non-run entries
        if run_id.startswith("."):
            continue
        try:
            with metrics_path.open() as f:
                metrics = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(f"Could not read {metrics_path}: {exc}. Skipping.")
            continue

        metadata = _parse_run_id(run_id)
        row = {"run_id": run_id}
        row.update(metadata)
        row.update(metrics)
        records.append(row)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # Place metadata columns first
    meta_cols = [c for c in ["run_id", "model", "strategy", "regime", "L", "H"] if c in df.columns]
    metric_cols = [c for c in df.columns if c not in meta_cols]
    df = df[meta_cols + metric_cols]

    # Sort by model, strategy, regime, L, H when those columns exist
    sort_cols = [c for c in ["model", "strategy", "regime", "L", "H"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    return df


def print_summary(runs_dir: str | Path = "runs") -> None:
    """Print a formatted summary table to stdout."""
    df = build_summary_table(runs_dir)
    if df.empty:
        print("No runs found.")
        return
    print(df.to_string(index=False))
