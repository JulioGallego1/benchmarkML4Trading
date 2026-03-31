from __future__ import annotations

import random

import numpy as np


def set_seed(seed: int = 2024) -> None:
    """Fix Python, NumPy, and PyTorch seeds for reproducibility."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
