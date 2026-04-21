"""
cloud_mask.py

Absolute threshold-based cloud and shadow masking.
Replaces the percentile-based approach to correctly flag fully cloudy scenes.
"""
import numpy as np

def make_robust_cloud_shadow_mask(chip: np.ndarray) -> np.ndarray:
    """
    """
    B2 = chip[0]  # Blue
    B4= chip[2]  # Red
    B8 = chip[3]  # NIR
    B11= chip[4]  # SWIR1

    
    scale = 10000.0 if chip.max() > 10.0 else 1.0

    # NDVI
    ndvi = (B8 - B4) / (B8 + B4 + 1e-6)

    # Vegetation (protect this!)
    vegetation = ndvi > 0.3

    # Cloud: bright in Blue + Red + SWIR confirmation, never flag vegetation
    cloud = (
        (B2  > 0.15 * scale) &
        (B4  > 0.15 * scale) &
        (B11 > 0.10 * scale) &   # SWIR confirmation avoids bright-veg false positives
        (~vegetation)
    )

    # Shadow: dark in NIR + SWIR, low NDVI
    shadow = (
        (B8  < 0.12 * scale) &
        (B11 < 0.08 * scale) &
        (ndvi < 0.2)
    )

    mask = (cloud | shadow)

    
    mask = mask & (~vegetation)

    return mask.astype(np.float32)