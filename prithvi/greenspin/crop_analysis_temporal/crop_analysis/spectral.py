"""
spectral.py

"""
import numpy as np


#normalization

def norm(arr: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(arr, 2), np.percentile(arr, 98)
    return np.clip((arr - lo) / (hi - lo + 1e-9), 0, 1)


#spectral composites

def make_rgb(chip: np.ndarray) -> np.ndarray:
    return np.stack([norm(chip[2]), norm(chip[1]), norm(chip[0])], axis=-1)


def make_nir_false(chip: np.ndarray) -> np.ndarray:
    return np.stack([norm(chip[3]), norm(chip[2]), norm(chip[1])], axis=-1)


#spectral indices

def compute_ndvi(chip: np.ndarray) -> np.ndarray:
    B4, B8 = chip[2], chip[3]
    return (B8 - B4) / (B8 + B4 + 1e-6)


def compute_indices(chip: np.ndarray) -> dict[str, np.ndarray]:
    """
    Band order: B2(blue), B3(green), B4(red), B8(NIR), B11(SWIR1), B12(SWIR2).
    """
    B2, B3, B4, B8, B11, B12 = [chip[i] for i in range(6)]
    return {
        'ndvi': (B8  - B4) / (B8  + B4  + 1e-6),# vegetation greenness
        'ndwi': (B3  - B8) / (B3  + B8  + 1e-6),  # open-water content
        'savi': 1.5 * (B8 - B4) / (B8 + B4 + 0.5), # soil-adjusted vegetation
        'ndre': (B8  - B3) / (B8  + B3  + 1e-6), # red-edge proxy (B8–B3)
    }


#enriched chip

def build_enriched_chip(chip: np.ndarray) -> np.ndarray:
    
    indices = compute_indices(chip)
    extra= np.stack(list(indices.values()), axis=0)
    return np.concatenate([chip, extra], axis=0)


#cloud and shadow masking

def make_cloud_shadow_mask(chip: np.ndarray) -> np.ndarray:
    
    B2 = chip[0].astype(np.float32)  # Blue
    B4  =chip[2].astype(np.float32)# Red
    B8 = chip[3].astype(np.float32)  # NIR
    B11= chip[4].astype(np.float32)   # SWIR1

    scale= 10_000.0 if chip.max() > 10.0 else 1.0

    cloud_thresh  = 0.20 * scale
    shadow_nir = 0.12 * scale
    shadow_swir  = 0.08 * scale

    ndvi = (B8 - B4) / (B8 + B4 + 1e-6)
    vegetation = ndvi > 0.25

    # fixed: added B11 SWIR confirmation and vegetation guard to cloud check
    cloud = (
        (B2  > cloud_thresh) &
        (B4  > cloud_thresh) &
        (B11 > 0.1 * scale) &   # SWIR confirmation avoids bright-veg false positives
        (~vegetation)
    )
    shadow= (B8 < shadow_nir) & (B11 < shadow_swir) & (~vegetation)

    return (cloud | shadow).astype(np.float32)


#temporal composite and cloud handling

def make_temporal_composite(
    chip_temporal: np.ndarray,   
    cloud_masks:   np.ndarray,   
) -> np.ndarray:
    
    T, C, H, W = chip_temporal.shape
    valid = (cloud_masks == 0)          # (T, H, W)  bool
    composite = np.zeros((C, H, W), dtype=np.float32)

    for c in range(C):
        band = chip_temporal[:, c, :, :].astype(np.float32)   # (T, H, W)
        masked  =np.where(valid, band, np.nan)
        median = np.nanmedian(masked, axis=0)        # NaN where all cloudy
        fallback = np.median(band, axis=0)                 # always defined
        composite[c] = np.where(np.isnan(median), fallback, median)

    return composite


def make_per_date_cloud_masks(chip_temporal: np.ndarray) -> np.ndarray:
    
    return np.stack(
        [make_cloud_shadow_mask(chip_temporal[t]) for t in range(chip_temporal.shape[0])],
        axis=0,
    )