"""
data_loader.py
"""

import json
import os
import numpy as np
from config import chip_path, mask_path, meta_path, DATES


def load_temporal_chips() -> np.ndarray:
    
    chips = []
    for d in DATES:
        path = chip_path(d)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing temporal chip: {path}\n"
                "Please generate chips in QGIS before running the pipeline."
            )
        chips.append(np.load(path))
    return np.stack(chips, axis=0)


def load_mask() -> np.ndarray:
    
    from config import OUTPUT_PATH, FIELD_ID
    tried = []
    for d in DATES:
        path = os.path.join(OUTPUT_PATH, f'prithvi_mask_FID{FIELD_ID}_{d}.npy')
        tried.append(path)
        if os.path.exists(path):
            m = np.load(path).astype(np.float32)
            if m.sum() > 0:
                print(f'  Mask loaded from: {os.path.basename(path)}  '
                      f'({int(m.sum())} field pixels)')
                return m
            else:
                print(f'  Warning: mask at {os.path.basename(path)} is all-zero, trying next date.')

    raise FileNotFoundError(
        "No valid (non-zero) field mask found for any date.\n"
        f"Tried:\n" + "\n".join(f"  {p}" for p in tried) + "\n"
        "Regenerate masks from QGIS."
    )


def load_meta() -> dict:
    
    for d in reversed(DATES):
        path = meta_path(d)
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    raise FileNotFoundError("No metadata JSON found for any date.")