"""
config.py
"""
import os
import glob


FIELD_ID    = 587
OUTPUT_PATH = r'C:\Users\Sudipto\internship\EO\prithvi\greenspin\multi_crop_output'
FIELD_FOLDER = os.path.join(OUTPUT_PATH, f'FID_{FIELD_ID}')


DATES = [
    '2024-04-08', '2024-04-23', '2024-04-30','2024-05-10', '2024-05-13', '2024-05-20','2024-05-23', '2024-05-28', '2024-05-30',
    '2024-06-04', '2024-06-07', '2024-06-09','2024-06-12', '2024-06-17', '2024-06-19','2024-06-22', '2024-06-24', '2024-06-27',
    '2024-06-29', '2024-07-09', '2024-07-14','2024-07-17', '2024-07-19', '2024-07-22','2024-07-29', '2024-08-11', '2024-08-13',
    '2024-08-16', '2024-08-21', '2024-08-23','2024-08-26', '2024-08-28', '2024-08-31','2024-09-02', '2024-09-05', '2024-09-07','2024-09-15',]


def get_available_dates() -> list[str]:
    
    pattern = os.path.join(FIELD_FOLDER, f'prithvi_input_FID{FIELD_ID}_*.npy')
    found = []
    for p in glob.glob(pattern):
        date_str = os.path.basename(p).replace(
            f'prithvi_input_FID{FIELD_ID}_', '').replace('.npy', '')
        if date_str in DATES:         
            found.append(date_str)
    return sorted(found)




def chip_path(date_str: str) -> str:
    return os.path.join(FIELD_FOLDER, f'prithvi_input_FID{FIELD_ID}_{date_str}.npy')


def scl_path(date_str: str) -> str:
    
    return os.path.join(FIELD_FOLDER, f'prithvi_scl_FID{FIELD_ID}_{date_str}.npy')


def meta_path(date_str: str) -> str:
    return os.path.join(FIELD_FOLDER, f'prithvi_meta_FID{FIELD_ID}_{date_str}.json')


def mask_path(date_str: str | None = None) -> str:
    d = date_str if date_str is not None else DATES[0]
    return os.path.join(FIELD_FOLDER, f'prithvi_mask_FID{FIELD_ID}_{d}.npy')


def dashboard_path() -> str:
    return os.path.join(FIELD_FOLDER, f'prithvi_dashboard_v2_Temporal.png')


def tif_path() -> str:
    return os.path.join(FIELD_FOLDER, f'prithvi_cluster_map_v2_Temporal.tif')



TEMPORAL_REPEATS = len(DATES)      
PATCH_GRID  = 14
CHIP_SIZE     = 224


MIN_CLUSTERS = 1
MAX_CLUSTERS = 8


PATCH_MASK_THRESHOLD = 0.05
INFRA_NDVI_THRESH  = -0.05


SCL_INCLUDE_DARK_AREA       = True
SCL_USE_SPECTRAL_SUPPLEMENT = True


MEANINGFUL_NDVI_SPREAD = 0.05
MIN_VALID_PATCHES      = 10
RANDOM_SEED            = 42
PCA_COMPONENT          = 10

TEMPORAL_STAT_WEIGHT   = 2.5   
DISPLAY_MIN_VALID_PCT  = 0.70  
EMB_STAT_PCA_DIMS      = 16