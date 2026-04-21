"""
cloud_mask.py

"""
import numpy as np

def make_robust_cloud_shadow_mask(chip: np.ndarray) -> np.ndarray:
    B2 = chip[0]  # Blue
    B4= chip[2] # Red
    B8= chip[3]   # NIR
    B11 =chip[4] # SWIR1

    scale = 10000.0 if chip.max() > 10.0 else 1.0

    # NDVI
    ndvi = (B8 - B4) / (B8 + B4 + 1e-6)

    # Vegetation (protect this!)
    vegetation = ndvi > 0.3

    #cloud detection 
    cloud = (
        (B2 > 0.2 * scale) &
        (B4 > 0.2 * scale) &
        (B11 > 0.1 * scale)
    )

    #shadow detection
    shadow = (
        (B8 < 0.15 * scale) &
        (B11 < 0.1 * scale) &
        (ndvi < 0.2)
    )

    mask = (cloud | shadow)

    
    mask= mask & (~vegetation)

    return mask.astype(np.float32)