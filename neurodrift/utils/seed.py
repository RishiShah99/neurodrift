"""Deterministic seeding across random, numpy, and torch."""

from __future__ import annotations

import os
import random


def seed_everything(seed: int) -> int:
    """Seed Python, NumPy, and PyTorch (if present). Returns the seed used."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    return seed
