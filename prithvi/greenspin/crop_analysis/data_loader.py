"""
data_loader.py

Loads the raw chip, binary field mask, and geo-metadata.
"""
import json

import numpy as np

from config import chip_path, mask_path, meta_path


def load_chip() -> np.ndarray:
    """Return the raw multi-band chip array."""          # satellite image chip with multiple bands
    return np.load(chip_path())


def load_mask() -> np.ndarray:
    """Return the binary field mask."""                  # isolates the field pixels from the background
    return np.load(mask_path()).astype(np.float32)


def load_meta() -> dict:
    """Return the geo-metadata dict from JSON."""        # geometadata
    with open(meta_path()) as f:
        return json.load(f)