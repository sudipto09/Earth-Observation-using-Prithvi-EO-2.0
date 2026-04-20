"""
main.py

Loads data, runs each stage, saves outputs.
"""
import os

import numpy as np
import torch
import config
from clustering import run_clustering
from data_loader import load_chip, load_mask, load_meta
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
    make_cloud_shadow_mask,
    make_nir_false,
    make_rgb,
)
from visualization import build_dashboard


def main() -> None:
    os.makedirs(config.OUTPUT_PATH, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, decoder = load_pipeline(device)
    # load raw data
    chip     = load_chip()        # (C, H, W)
    mask_224 = load_mask()        # (H, W) float32
    meta     = load_meta()
    
    # spectral preprocessing and indices
    rgb      = make_rgb(chip)
    nir_false   = make_nir_false(chip)
    ndvi     = compute_ndvi(chip)
    ndvi_display  = np.clip(ndvi, 0, 1)
    chip_enriched = build_enriched_chip(chip)   # (10, H, W)

    # cloud / shadow filtering
    cloud_shadow   = make_cloud_shadow_mask(chip)                                       # (H, W) 
    mask_224_clean = np.where(cloud_shadow == 1, 0.0, mask_224).astype(np.float32)  # clear field pixels only
    print(f"  Field pixels original : {int(mask_224.sum())}  "
          f"|  after cloud/shadow filter: {int(mask_224_clean.sum())}")

    # Prithvi encoder
    input_tensor = build_input_tensor(chip, device)
    embeddings   = extract_patch_tokens(model, input_tensor)   # (196, D)

    # mask and upsample
    embeddings   = mask_patch_embeddings(embeddings, mask_224_clean, chip=chip)

    feature_map  = make_feature_map(embeddings, mask_224=mask_224_clean, mode='l2')
    emb_pixels   = upsample_embeddings(embeddings)              # (H*W, D)

    mask_flat = mask_224_clean.ravel() == 1
    emb_pixels[~mask_flat] = 0

    # adaptive clustering 
    result = run_clustering(
        emb_pixels = emb_pixels,
        chip_enriched = chip_enriched,
        mask_224   = mask_224_clean,
        ndvi        = ndvi,
    )
    print(f"  BIC-optimal zones: {result.optimal_n}  "
          f"|  BIC scores: {[f'{s:.0f}' for s in result.bic_scores]}")

    # dashboard
    build_dashboard(
        rgb     = rgb,
        nir_false  = nir_false,
        ndvi_display = ndvi_display,
        ndvi    = ndvi,
        feature_map = feature_map,
        chip    = chip,
        chip_enriched= chip_enriched,
        mask_224 = mask_224,
        mask_clean  = mask_224_clean,
        emb_pixels   = emb_pixels,
        result    = result,
        device   = device,
        save_path  = config.dashboard_path(),
    )

    # GeoTIFF export
    save_geotiff(result, meta, config.tif_path())
    print("Processing complete.")


if __name__ == '__main__':
    main()