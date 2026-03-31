from __future__ import annotations

import os
from typing import Optional


def get_runs_dir(base_dir: str, sub_dir: Optional[str] = None) -> str:
    """Get the path to the runs directory, optionally creating a subfolder.

    Parameters
    ----------
    base_dir : str
        Root directory of the project (e.g. where ``runs/`` should live).
    sub_dir : str, optional
        Name of a subdirectory to create under ``runs``. If ``None``,
        returns the path to the top-level runs directory.

    Returns
    -------
    str
        Absolute path to the requested directory. The directory is created
        if it does not already exist.
    """
    runs_root = os.path.join(base_dir, "runs")
    os.makedirs(runs_root, exist_ok=True)
    if sub_dir:
        path = os.path.join(runs_root, sub_dir)
        os.makedirs(path, exist_ok=True)
        return path
    return runs_root