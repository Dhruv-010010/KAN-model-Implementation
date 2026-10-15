"""Reproducibility helpers."""

# %% [utils]
import json
import os
import random
import numpy as np
import torch


def set_seed(seed):
    """Seed Python, NumPy and PyTorch (CPU and CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    """Use CUDA when present, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
