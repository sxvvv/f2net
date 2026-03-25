# utils/misc.py
# General-purpose utilities.

import os
import random
import numpy as np
import torch


def seed_everything(seed=42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_id=None):
    """Return a ``torch.device`` (GPU if available, else CPU)."""
    if torch.cuda.is_available():
        if device_id is not None:
            return torch.device(f"cuda:{device_id}")
        return torch.device("cuda:0")
    return torch.device("cpu")


def makedirs(path):
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)