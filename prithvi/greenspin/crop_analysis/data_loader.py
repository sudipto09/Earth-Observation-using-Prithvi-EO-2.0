"""
Loads the raw chip, binary field mask, and geo-metadata from disk.
"""
import json
import numpy as np
from config import chip_path, mask_path, meta_path


def load_chip() -> np.ndarray:                                                                       # satellite image chip with multiple bands
    """Return the raw multi-band chip array """
    return np.load(chip_path())


def load_mask() -> np.ndarray:                                                                          #isolates the field pixels from the background
    """Return the binary field mask"""
    return np.load(mask_path()).astype(np.float32)


def load_meta() -> dict:                                                                                # needed for geolocation later
    """Return the geo-metadata dict from JSON."""
    with open(meta_path()) as f:
        return json.load(f)