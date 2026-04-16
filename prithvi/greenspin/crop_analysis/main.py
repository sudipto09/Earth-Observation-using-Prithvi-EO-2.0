"""
loads data, runs each stage, saves outputs.

"""

import os
import torch
import numpy as np
from modelfactory import load_pipeline
from sklearn.decomposition import PCA
import config
from data_loader import load_chip, load_mask, load_meta
from spectral import make_rgb, make_nir_false, compute_ndvi, build_enriched_chip, make_cloud_shadow_mask
from encoder import build_input_tensor, extract_patch_tokens, mask_patch_embeddings, upsample_embeddings, make_feature_map
from clustering import run_clustering
from visualization import build_dashboard
from export import save_geotiff


def main() -> None:
    os.makedirs(config.OUTPUT_PATH, exist_ok=True)

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, decoder = load_pipeline(device)

    #load raw data
    chip   = load_chip()       # (C, H, W)
    mask_224 = load_mask()       # (H, W) float32
    meta  = load_meta()

    #spectral preprocessing and indices
    rgb       = make_rgb(chip)
    nir_false  = make_nir_false(chip)
    ndvi       = compute_ndvi(chip)
    ndvi_display  = np.clip(ndvi, 0, 1)
    chip_enriched = build_enriched_chip(chip)   # (10, H, W)

    #cloud / shadow filtering 
    cloud_shadow   = make_cloud_shadow_mask(chip)                                       # (H, W) float32
    mask_224_clean = np.where(cloud_shadow == 1, 0.0, mask_224).astype(np.float32)  # clear field pixels only
    print(f"  Field pixels original: {int(mask_224.sum())}  |  after cloud/shadow filter: {int(mask_224_clean.sum())}")

    #prithvi encoder
    input_tensor = build_input_tensor(chip, device)
    embeddings   = extract_patch_tokens(model, input_tensor)   # (196, 1024)
    embeddings   = mask_patch_embeddings(embeddings, mask_224_clean) # zero background + cloud/shadow patches
    

    pca_temp = PCA(n_components=1)
    pca_vals = pca_temp.fit_transform(embeddings)
    pca_vals = np.abs(pca_vals).reshape(-1)

    feature_map = make_feature_map(embeddings, mask_224 = mask_224_clean, mode='l2')
    emb_pixels  = upsample_embeddings(embeddings)             # (H*W, 1024)
    mask_flat = mask_224_clean.ravel() == 1
    emb_pixels[~mask_flat] = 0
    #clustering
    result = run_clustering(
        emb_pixels  = emb_pixels,
        chip_enriched = chip_enriched,
        mask_224   = mask_224_clean,
        ndvi     = ndvi,
    )

    #dashboard
    build_dashboard(
        rgb    = rgb,
        nir_false   = nir_false,
        ndvi_display = ndvi_display,
        ndvi      = ndvi,
        feature_map  = feature_map,
        chip     = chip,
        chip_enriched= chip_enriched,
        mask_224= mask_224,
        mask_clean  = mask_224_clean,
        emb_pixels = emb_pixels,
        result  = result,
        device =   device,
        save_path  = config.dashboard_path(),
    )

    #geotiff export
    save_geotiff(result, meta, config.tif_path())

    print("Processing complete.")


if __name__ == '__main__':
    main()