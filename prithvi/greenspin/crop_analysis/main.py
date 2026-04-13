"""
loads data, runs each stage, saves outputs.

"""

import os
import torch
import numpy as np
from modelfactory import load_pipeline

import config
from data_loader import load_chip, load_mask, load_meta
from spectral import make_rgb, make_nir_false, compute_ndvi, build_enriched_chip
from encoder import build_input_tensor, extract_patch_tokens, upsample_embeddings, make_feature_map
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

    #prithvi encoder
    input_tensor = build_input_tensor(chip, device)
    embeddings   = extract_patch_tokens(model, input_tensor)   # (196, 1024)
    feature_map  = make_feature_map(embeddings, mask_224=mask_224)                # (14, 14)
    emb_pixels  = upsample_embeddings(embeddings)             # (H*W, 1024)

    #clustering
    result = run_clustering(
        emb_pixels  = emb_pixels,
        chip_enriched = chip_enriched,
        mask_224   = mask_224,
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