"""Device selection and reproducibility (chapter 2)."""

import os
import random

import numpy as np
import torch


def resolve_device(preference: str = "auto") -> torch.device:
    """Pick the best available device unless one is forced."""
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """Seed every random generator used in the book."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
