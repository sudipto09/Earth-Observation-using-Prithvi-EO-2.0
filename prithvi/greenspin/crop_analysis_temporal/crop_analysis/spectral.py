"""
spectral.py

Spectral index computation (NDVI, NDWI, SAVI, NDRE), cloud-free temporal
compositing (median of top-N greenest observations per pixel), display image
generation, best-date selection, and per-pixel temporal NDVI statistics (7 features).


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




#display date selection

def select_best_clear_date(
    chip_temporal: np.ndarray,
    cloud_masks:   np.ndarray,
    mask_224:      np.ndarray,
    used_dates:    list[str],
    min_valid_pct: float = 0.70,
) -> tuple[int, str, float]:
    
    T = chip_temporal.shape[0]
    field_flat = mask_224.ravel() == 1
    n_field = int(field_flat.sum())

    scores: list[float] = []
    for t in range(T):
        valid_field = (cloud_masks[t].ravel() == 0) & field_flat
        valid_pct = valid_field.sum() / max(n_field, 1)

        if valid_pct >= min_valid_pct and valid_field.sum() > 0:
            B4 = chip_temporal[t, 2].ravel()[valid_field].astype(np.float32)
            B8 = chip_temporal[t, 3].ravel()[valid_field].astype(np.float32)
            ndvi_val = float(np.mean((B8 - B4) / (B8 + B4 + 1e-6)))
            
            scores.append(valid_pct * max(ndvi_val, 0.01))
        else:
            scores.append(0.0)

    if max(scores) > 0:
        best_t = int(np.argmax(scores))
    else:
        # pick date has most clean field pixels
        valid_counts = [
            int(((cloud_masks[t].ravel() == 0) & field_flat).sum())
            for t in range(T)
        ]
        best_t= int(np.argmax(valid_counts))

    best_valid_pct = float(
        ((cloud_masks[best_t].ravel() == 0) & field_flat).sum() / max(n_field, 1)
    )
    return best_t, used_dates[best_t], best_valid_pct


#temporal ndvi statistics

def extract_temporal_ndvi_stats(
    chip_temporal: np.ndarray,
    cloud_masks: np.ndarray,
    mask_224: np.ndarray,
) -> np.ndarray:
    
    T, C, H, W = chip_temporal.shape
    valid = (cloud_masks == 0).astype(np.float32)    

    B4 = chip_temporal[:, 2, :, :].astype(np.float32)
    B8 = chip_temporal[:, 3, :, :].astype(np.float32)
    ndvi_ts = (B8 - B4) / (B8 + B4 + 1e-6)              
    ndvi_masked = np.where(valid, ndvi_ts, np.nan)

    with np.errstate(all='ignore'):
        mean_ndvi= np.nanmean(ndvi_masked, axis=0)            
        peak_ndvi= np.nanmax(ndvi_masked,  axis=0)
        min_ndvi = np.nanmin(ndvi_masked,  axis=0)
        std_ndvi = np.nanstd(ndvi_masked,  axis=0)

    #0 = start of stack, 1 = end
    peak_idx  = np.nanargmax(
        np.where(valid.astype(bool), ndvi_ts, -np.inf), axis=0
    )
    peak_timing = peak_idx.astype(np.float32) / max(T - 1, 1)  

    #compare first-half vs second-half mean NDVI
    half = max(T // 2, 1)
    with np.errstate(all='ignore'):
        early = np.nanmean(ndvi_masked[:half], axis=0)
        late= np.nanmean(ndvi_masked[half:], axis=0)

    greenup_rate = np.where(np.isnan(early) | np.isnan(late), 0.0, late  - early)
    senescence_rate = np.where(np.isnan(early) | np.isnan(late), 0.0, early - late)

    stats_raw = np.stack(
        [mean_ndvi, peak_ndvi, min_ndvi, std_ndvi,
         peak_timing, greenup_rate, senescence_rate],
        axis=0,
    )   

    
    out  = np.zeros((7, H, W), dtype=np.float32)
    field_px = mask_224 == 1
    for s in range(7):
        plane = stats_raw[s]
        fmed  = float(np.nanmedian(plane[field_px])) if field_px.any() else 0.0
        out[s] =np.where(np.isnan(plane), fmed, plane)

    return out.reshape(7, H * W).T    