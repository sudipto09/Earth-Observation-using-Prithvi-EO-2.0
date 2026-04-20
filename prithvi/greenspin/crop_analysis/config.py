"""
config.py
"""
import os

# field / date
FIELD_ID  = 2539
DATE     = '2024-07-14'
OUTPUT_PATH = r'C:\Users\Sudipto\internship\EO\prithvi\greenspin\multi_crop_output'


def chip_path() -> str:
    return os.path.join(OUTPUT_PATH, f'prithvi_input_FID{FIELD_ID}_{DATE}.npy')

def meta_path() -> str:
    return os.path.join(OUTPUT_PATH, f'prithvi_meta_FID{FIELD_ID}_{DATE}.json')

def mask_path() -> str:
    return os.path.join(OUTPUT_PATH, f'prithvi_mask_FID{FIELD_ID}_{DATE}.npy')

def dashboard_path() -> str:
    return os.path.join(OUTPUT_PATH, f'prithvi_dashboard_v2_FID{FIELD_ID}_{DATE}.png')

def tif_path() -> str:
    return os.path.join(OUTPUT_PATH, f'cluster_map_v2_FID{FIELD_ID}_{DATE}.tif')


# model and data parameters
TEMPORAL_REPEATS = 4
PATCH_GRID    = 14
CHIP_SIZE    = 224

# adaptive clustering via BIC 
MIN_CLUSTERS = 2        # lower bound  
MAX_CLUSTERS = 8        # upper bound 

# patch-level masking 
PATCH_MASK_THRESHOLD = 0.25    # fraction of patch pixels that must be valid field
INFRA_NDVI_THRESH    = -0.05   # mean ndvi threshold 


RANDOM_SEED   = 42
PCA_COMPONENT = 10