"""
cloud_mask.py

Generates per-pixel binary cloud/shadow masks for Sentinel-2 chips.
Combines Sentinel-2 SCL class filtering with a spectral heuristic fallback.
Main entry point: make_combined_cloud_mask(chip, scl). Returns float32 mask, 1=cloud.

"""
from __future__ import annotations
import numpy as np

_SCL_BAD_CLASSES = frozenset([0, 1, 3, 8, 9, 10, 11])
_SCL_DARK_AREA_CLASS = 2


def make_scl_mask(scl: np.ndarray, include_dark_area: bool = True) -> np.ndarray:
    bad = np.zeros(scl.shape, dtype=bool)
    for cls in _SCL_BAD_CLASSES:
        bad |= (scl == cls)
    if include_dark_area:
        bad |= (scl == _SCL_DARK_AREA_CLASS)
    return bad.astype(np.float32)


def make_spectral_mask(chip: np.ndarray) -> np.ndarray:
    B2  = chip[0].astype(np.float32)
    B4  = chip[2].astype(np.float32)
    B8  = chip[3].astype(np.float32)
    B11 = chip[4].astype(np.float32)
    scale = 10_000.0 if chip.max() > 10.0 else 1.0
    ndvi = (B8 - B4) / (B8 + B4 + 1e-6)
    vegetation = ndvi > 0.3
    cloud = (
        (B2  > 0.20 * scale) &
        (B4  > 0.20 * scale) &
        (B11 > 0.10 * scale) &
        (~vegetation)
    )
    shadow = (
        (B8  < 0.15 * scale) &
        (B11 < 0.10 * scale) &
        (ndvi < 0.2)
    )
    return (cloud | shadow).astype(np.float32)


def make_combined_cloud_mask(
    chip: np.ndarray,
    scl:  np.ndarray | None,
    include_dark_area: bool = True,
    use_spectral_supplement: bool = True,
) -> np.ndarray:
    if scl is None:
        return make_spectral_mask(chip)
    scl_mask = make_scl_mask(scl, include_dark_area=include_dark_area)
    if not use_spectral_supplement:
        return scl_mask
    spectral_on_clear = make_spectral_mask(chip) * (1.0 - scl_mask)
    return np.clip(scl_mask + spectral_on_clear, 0.0, 1.0).astype(np.float32)


def scl_coverage_report(scl: np.ndarray) -> dict:
    labels = {
        0: 'NO_DATA', 1: 'SATURATED', 2: 'DARK_AREA', 3: 'CLOUD_SHADOW',
        4: 'VEGETATION', 5: 'NOT_VEGETATED', 6: 'WATER', 7: 'UNCLASSIFIED',
        8: 'CLOUD_MED', 9: 'CLOUD_HIGH', 10: 'THIN_CIRRUS', 11: 'SNOW_ICE',
    }
    return {labels.get(c, str(c)): int((scl == c).sum()) for c in range(12)}


def make_robust_cloud_shadow_mask(chip: np.ndarray) -> np.ndarray:
    """Backward-compatible alias."""
    return make_spectral_mask(chip)