"""
config.py
"""
import os

# field / dates
FIELD_ID = 587

DATES = ['2024-04-23', '2024-05-13', '2024-06-17', '2024-07-14', '2024-08-16', '2024-09-15']
OUTPUT_PATH = r'C:\Users\Sudipto\internship\EO\prithvi\greenspin\multi_crop_output'


MEANINGFUL_NDVI_SPREAD = 0.05

MIN_VALID_PATCHES = 10   #

def chip_path(date_str: str) -> str:
    return os.path.join(OUTPUT_PATH, f'prithvi_input_FID{FIELD_ID}_{date_str}.npy')


def meta_path(date_str: str) -> str:
    return os.path.join(OUTPUT_PATH, f'prithvi_meta_FID{FIELD_ID}_{date_str}.json')


def mask_path(date_str: str | None = None) -> str:
    
    d = date_str if date_str is not None else DATES[0]
    return os.path.join(OUTPUT_PATH, f'prithvi_mask_FID{FIELD_ID}_{d}.npy')


def dashboard_path() -> str:
    return os.path.join(OUTPUT_PATH, f'prithvi_dashboard_v2_FID{FIELD_ID}_Temporal.png')


def tif_path() -> str:
    return os.path.join(OUTPUT_PATH, f'cluster_map_v2_FID{FIELD_ID}_Temporal.tif')


# model and data parameters
TEMPORAL_REPEATS = len(DATES)      # dynamically set
PATCH_GRID   = 14
CHIP_SIZE   = 224

# adaptive clustering via BIC
MIN_CLUSTERS = 2                   # lower bound
MAX_CLUSTERS = 8                   # upper bound

# patch-level masking
PATCH_MASK_THRESHOLD = 0.1         # fraction of patch pixels that must be valid field
INFRA_NDVI_THRESH = -0.05       # mean NDVI threshold for infra-pixel removal

RANDOM_SEED = 42
PCA_COMPONENT = 10