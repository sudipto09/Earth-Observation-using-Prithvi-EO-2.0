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

   
    cloud = (
        (B2  > cloud_thresh) &
        (B4  > cloud_thresh) &
        (B11 > 0.1 * scale) &   
        (~vegetation)
    )
    shadow= (B8 < shadow_nir) & (B11 < shadow_swir) & (~vegetation)

    return (cloud | shadow).astype(np.float32)


#temporal composite and cloud handling

def make_temporal_composite(
    chip_temporal: np.ndarray,
    cloud_masks: np.ndarray,
    top_n_greenest: int = 5,
) -> np.ndarray:
    
    T, C, H, W = chip_temporal.shape
    valid= (cloud_masks == 0).astype(np.float32)   

    # Per-date NDVI
    B4= chip_temporal[:, 2, :, :].astype(np.float32)
    B8 = chip_temporal[:, 3, :, :].astype(np.float32)
    ndvi_t = (B8 - B4) / (B8 + B4 + 1e-6)           

    
    ndvi_valid = np.where(valid, ndvi_t, -np.inf)    

    
    k = min(top_n_greenest, T)
    sorted_idx = np.argsort(ndvi_valid, axis=0)       
    top_idx    = sorted_idx[-k:, :, :]                

    composite = np.zeros((C, H, W), dtype=np.float32)
    for c in range(C):
        band = chip_temporal[:, c, :, :].astype(np.float32)   
        gathered= np.take_along_axis(band, top_idx, axis=0)  
        valid_gathered = np.take_along_axis(valid, top_idx, axis=0)

        gathered_masked = np.where(valid_gathered, gathered, np.nan)
        
        with np.errstate(all='ignore'):
            px_median = np.nanmedian(gathered_masked, axis=0)      

        
        with np.errstate(all='ignore'):
            fallback = np.nanmedian(np.where(valid, band, np.nan), axis=0)
        all_nan_fallback = np.median(band, axis=0)
        fallback = np.where(np.isnan(fallback), all_nan_fallback, fallback)

        composite[c] = np.where(np.isnan(px_median), fallback, px_median)

    return composite


def make_per_date_cloud_masks(chip_temporal: np.ndarray) -> np.ndarray:
    
    return np.stack(
        [make_cloud_shadow_mask(chip_temporal[t]) for t in range(chip_temporal.shape[0])],
        axis=0,
    )