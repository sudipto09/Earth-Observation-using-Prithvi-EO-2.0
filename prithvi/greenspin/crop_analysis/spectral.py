"""
spectral.py

build_rgb() and build_nir_false() create display composites for the dashboard.
"""
import numpy as np


#helper for display normalization

def norm(arr: np.ndarray) -> np.ndarray:
    """Stretch a 2 D band to [0, 1] using 2nd / 98th percentile clipping."""
    lo, hi = np.percentile(arr, 2), np.percentile(arr, 98)
    return np.clip((arr - lo) / (hi - lo + 1e-9), 0, 1)


#spectral composites

def make_rgb(chip: np.ndarray) -> np.ndarray:
    """True-colour composite (B3-B2-B1 -> R-G-B), shape (H, W, 3)."""
    return np.stack([norm(chip[2]), norm(chip[1]), norm(chip[0])], axis=-1)


def make_nir_false(chip: np.ndarray) -> np.ndarray:
    """NIR false-colour composite (B4-B3-B2 -> R-G-B), shape (H, W, 3)."""
    return np.stack([norm(chip[3]), norm(chip[2]), norm(chip[1])], axis=-1)


#spectral indices

def compute_ndvi(chip: np.ndarray) -> np.ndarray:
    """NDVI = (NIR - Red) / (NIR + Red).  Returns raw float array."""
    B4, B8 = chip[2], chip[3]
    return (B8 - B4) / (B8 + B4 + 1e-6)


def compute_indices(chip: np.ndarray) -> dict[str, np.ndarray]:
    """
    Compute four spectral indices from the chip bands.

    Band order: B2, B3, B4, B8, B11, B12 

    Returns a dict with keys: ndvi, ndwi, savi, ndre.
    """
    B2, B3, B4, B8, B11, B12 = [chip[i] for i in range(6)]

    return {
        'ndvi': (B8  - B4) / (B8  + B4  + 1e-6),          # vegetation greenness
        'ndwi': (B3  - B8) / (B3  + B8  + 1e-6),          # water content
        'savi': 1.5 * (B8 - B4) / (B8 + B4 + 0.5),        # soil-adjusted veg
        'ndre': (B8  - B4) / (B8  + B4  + 1e-6),          # red edge
    }


#derived chip with 10 channels: original 6 + 4 indices

def build_enriched_chip(chip: np.ndarray) -> np.ndarray:
    """
    Concatenate the raw 6-band chip with the 4 derived indices.

    Returns an array of shape (10, H, W).
    """
    indices = compute_indices(chip)
    extra   = np.stack(list(indices.values()), axis=0)  
    return np.concatenate([chip, extra], axis=0)        