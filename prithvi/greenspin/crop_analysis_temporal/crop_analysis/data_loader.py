"""
data_loader.py
"""
import json
import os
import numpy as np
from config import chip_path, meta_path, scl_path, DATES, get_available_dates


def load_temporal_chips() -> tuple[np.ndarray, list[str]]:
   
    used_dates = get_available_dates()
    if not used_dates:
        raise FileNotFoundError(
            f"No chip files found in {chip_path('*')}\n"
            "Run the QGIS extraction script first."
        )

    skipped = [d for d in DATES if d not in used_dates]
    if skipped:
        print(f"  {len(used_dates)}/{len(DATES)} dates found. "
              f"Missing: {skipped}")
    else:
        print(f"All {len(used_dates)} dates found.")

    chips = [np.load(chip_path(d)) for d in used_dates]
    return np.stack(chips, axis=0), used_dates


def load_scl_chips(used_dates: list[str]) -> list[np.ndarray | None]:
   
    scl_chips: list[np.ndarray | None] = []
    n_found = 0
    for d in used_dates:
        path = scl_path(d)
        if os.path.exists(path):
            arr = np.load(path)
            if arr.ndim == 3 and arr.shape[0] == 1:
                arr = arr[0]
            scl_chips.append(arr.astype(np.uint8))
            n_found += 1
        else:
            scl_chips.append(None)

    n_total= len(used_dates)
    if n_found == 0:
        print(
            f" No SCL files found for any of the {n_total} dates.\n"
            f"        Falling back to spectral-only cloud masking."
        )
    elif n_found < n_total:
        missing = [d for d, s in zip(used_dates, scl_chips) if s is None]
        print(f"  {n_found}/{n_total} SCL files loaded. "
              f"Spectral fallback for {len(missing)} dates: {missing}")
    else:
        print(f"   All {n_found}/{n_total} SCL files loaded.")

    return scl_chips


def load_mask() -> np.ndarray:
    from config import OUTPUT_PATH, FIELD_FOLDER, FIELD_ID
    tried = []
    for d in get_available_dates():
        path = os.path.join(FIELD_FOLDER, f'prithvi_mask_FID{FIELD_ID}_{d}.npy')
        tried.append(path)
        if os.path.exists(path):
            m = np.load(path).astype(np.float32)
            if m.sum() > 0:
                print(f'({int(m.sum())} field pixels)')
                return m
            else:
                print(f'  Mask {os.path.basename(path)} is all-zero, trying next date.')

    raise FileNotFoundError(
        "No valid (non-zero) field mask found for any date.\n"
        "Tried:\n" + "\n".join(f"  {p}" for p in tried) + "\n"
        "Regenerate masks from QGIS."
    )


def load_meta() -> dict:
    for d in reversed(get_available_dates()):
        path = meta_path(d)
        if os.path.exists(path):
            with open(path) as f:
        
                return json.load(f)
    raise FileNotFoundError("No metadata JSON found for any date.")