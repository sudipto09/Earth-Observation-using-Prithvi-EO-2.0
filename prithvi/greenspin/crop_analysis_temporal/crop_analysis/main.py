"""
main.py 
"""
import os
import numpy as np
import torch
import config
from clustering import run_clustering
from data_loader import load_temporal_chips, load_mask, load_meta
from encoder import (
    build_input_tensor,
    extract_patch_tokens,
    make_feature_map, 
    mask_patch_embeddings,
    upsample_embeddings,
)
from export import save_geotiff
from modelfactory import load_pipeline
from spectral import (
    build_enriched_chip,
    
    compute_ndvi,
    make_nir_false,
    make_rgb,
    make_temporal_composite,
)
from visualization import build_dashboard
from cloud_mask import make_robust_cloud_shadow_mask


def main() -> None:
    os.makedirs(config.OUTPUT_PATH, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device.type.upper()}')

    model, decoder = load_pipeline(device)

    #load data
    chip_temporal =load_temporal_chips()          # (T, C, H, W)

    mask_224 = load_mask()
    mask_224 =(mask_224 > 0.5).astype(np.float32)

    # guard
    if mask_224.sum() == 0:
        raise ValueError(
            f"Field mask is entirely zero.\n"
            f"Check the .npy file at: {config.mask_path()}\n"
            "Regenerate the mask from QGIS for the correct field ID and date."
        )

    meta = load_meta()

    #temporal cloud handling
    cloud_masks = np.stack(
        [make_robust_cloud_shadow_mask(chip_temporal[t]) for t in range(chip_temporal.shape[0])],
        axis=0,
    )

    
    chip = make_temporal_composite(chip_temporal, cloud_masks)  # (C, H, W)

    
    all_cloudy   = (cloud_masks.sum(axis=0) == len(config.DATES)).astype(np.float32)
    mask_224_clean = np.where(all_cloudy == 1, 0.0, mask_224).astype(np.float32)

    n_orig= int(mask_224.sum())
    n_clean= int(mask_224_clean.sum())
    print(f'  Field pixels  original: {n_orig}  |  after temporal cloud filter: {n_clean}')

    if n_clean < 100:
        raise ValueError(
            f"Only {n_clean} valid field pixels after cloud filtering across all "
            f"{len(config.DATES)} dates. Check input chips or cloud-mask thresholds."
        )

    # spectral processing on composite chip
    rgb      = make_rgb(chip)
    nir_false  = make_nir_false(chip)
    ndvi    = compute_ndvi(chip)
    ndvi_display = np.clip(ndvi, 0, 1)
    chip_enriched = build_enriched_chip(chip)      # (10, H, W)   

    #Prithvi encoder
    input_tensor = build_input_tensor(chip_temporal, device)
    embeddings= extract_patch_tokens(model, input_tensor)

    embeddings  = mask_patch_embeddings(embeddings, mask_224_clean, chip=chip)
    feature_map = make_feature_map(embeddings, mask_224=mask_224_clean, mode='l2')  

    n_field_patches = int(np.sum(~np.isnan(feature_map)))   # count non-NaN patches
    if n_field_patches < config.MIN_VALID_PATCHES:
        print(
            f"  WARNING: Only {n_field_patches} Prithvi patches cover "
            f"Field {config.FIELD_ID}."
        )

    emb_pixels = upsample_embeddings(embeddings)

    mask_flat = mask_224_clean.ravel() == 1
    emb_pixels[~mask_flat] = 0.0

    #bic clustering
    result = run_clustering(
        emb_pixels = emb_pixels,
        chip_enriched = chip_enriched,
        mask_224 = mask_224_clean,
        ndvi    = ndvi,
    )
    print(f'  BIC-optimal zones: {result.optimal_n}  '
          f'|  BIC scores: {[f"{s:.0f}" for s in result.bic_scores]}')

    
    result.temporal_dates = config.DATES
    result.n_dates        = len(config.DATES)

    #dashboard
    build_dashboard(
        rgb    = rgb,
        nir_false   = nir_false,
        ndvi_display = ndvi_display,
        ndvi    = ndvi,
        feature_map  = feature_map,
        chip     = chip,
        chip_enriched= chip_enriched,
        mask_224  = mask_224,
        mask_clean = mask_224_clean,
        emb_pixels  = emb_pixels,
        result   = result,
        device   = device,
        meta   = meta,
        save_path  = config.dashboard_path(),
    )

    #export geotiff
    save_geotiff(result, meta, config.tif_path())
    print('Processing complete.')


if __name__ == '__main__':
    main()