"""
main.py
"""
import os
import numpy as np
import torch
import config
from clustering import run_clustering
from data_loader import load_temporal_chips, load_mask, load_meta, load_scl_chips
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
from cloud_mask import make_combined_cloud_mask, scl_coverage_report


def main() -> None:
    os.makedirs(config.OUTPUT_PATH, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device.type.upper()}')

    model, _ = load_pipeline(device)

    # load data
    
    chip_temporal, used_dates = load_temporal_chips()   
    scl_chips = load_scl_chips(used_dates)              

    #
    config.TEMPORAL_REPEATS = len(used_dates)

    mask_224 = load_mask()
    mask_224 = (mask_224 > 0.5).astype(np.float32)

    if mask_224.sum() == 0:
        raise ValueError(
            f"Field mask is entirely zero.\n"
            f"Check: {config.mask_path()}\n"
            "Regenerate the mask from QGIS."
        )

    meta = load_meta()

    #cloud masking scl and spectral
    print(f"\n  Cloud masking for {len(used_dates)} dates:")
    cloud_masks_list = []
    for t, (d, scl) in enumerate(zip(used_dates, scl_chips)):
        chip_t = chip_temporal[t]

        if scl is not None:
            report = scl_coverage_report(scl)
            bad_px= int(sum(v for k, v in report.items()
                              if k not in ('VEGETATION', 'NOT_VEGETATED',
                                           'WATER', 'UNCLASSIFIED')))
            src = 'SCL+spectral'
        else:
            bad_px = -1
            src = 'spectral-only'

        cm = make_combined_cloud_mask(
            chip = chip_t,
            scl = scl,
            include_dark_area = config.SCL_INCLUDE_DARK_AREA,
            use_spectral_supplement = config.SCL_USE_SPECTRAL_SUPPLEMENT,
        )
        cloud_masks_list.append(cm)

        cloud_pct = float(cm.mean()) * 100
        if scl is not None:
            print(f"    {d}  [{src}]  SCL bad px: {bad_px}/{int(scl.size)}  "
                  f"-> mask: {cloud_pct:.1f}% cloudy")
        else:
            print(f"    {d}  [{src}]  mask: {cloud_pct:.1f}% cloudy")

    cloud_masks = np.stack(cloud_masks_list, axis=0) 

    #cloud-masked NDVI per date
    print("\n  Per-date field NDVI (clean pixels only):")
    field_flat = mask_224.ravel() == 1
    for t, d in enumerate(used_dates):
        valid_field = (cloud_masks[t].ravel() == 0) & field_flat
        if valid_field.sum() > 0:
            B4_t = chip_temporal[t, 2].ravel()[valid_field]
            B8_t   = chip_temporal[t, 3].ravel()[valid_field]
            ndvi_t= float(np.mean((B8_t - B4_t) / (B8_t + B4_t + 1e-6)))
            valid_pct = valid_field.sum() / max(field_flat.sum(), 1) * 100
            
            print(f"    {d}  NDVI={ndvi_t:+.3f}  valid={valid_pct:.0f}% ")
        else:
            print(f"    {d}  ALL CLOUDY - excluded ")

    # Composite chip using top-N greenest pixels per location
    chip = make_temporal_composite(
        chip_temporal, cloud_masks, top_n_greenest=5)  

    # Remove field pixels that are cloudy in every single date
    all_cloudy= (cloud_masks.sum(axis=0) == len(used_dates)).astype(np.float32)
    mask_224_clean = np.where(all_cloudy == 1, 0.0, mask_224).astype(np.float32)

    n_orig= int(mask_224.sum())
    n_clean = int(mask_224_clean.sum())
    # print(f'\n  Field pixels  original: {n_orig}  |  after cloud filter: {n_clean}')

    if n_clean < 100:
        raise ValueError(
            f"Only {n_clean} valid field pixels after cloud filtering.\n"
            
        )

    #spectral indices and enriched chip for dashboard and clustering
    rgb = make_rgb(chip)
    nir_false  = make_nir_false(chip)
    ndvi   = compute_ndvi(chip)
    ndvi_display= np.clip(ndvi, 0, 1)
    chip_enriched = build_enriched_chip(chip)   

    #composite NDVI over field
    field_ndvi = float(np.mean(ndvi[mask_224_clean == 1])) if n_clean > 0 else 0.0
    print(f'  Composite NDVI over field: {field_ndvi:.3f}')

    #prithvi encoder
    input_tensor= build_input_tensor(chip_temporal, device)
    embeddings= extract_patch_tokens(model, input_tensor)

    embeddings= mask_patch_embeddings(embeddings, mask_224_clean, chip=chip)
    feature_map = make_feature_map(embeddings, mask_224=mask_224_clean, mode='l2')

    n_field_patches=int(np.sum(~np.isnan(feature_map)))
    if n_field_patches < config.MIN_VALID_PATCHES:
        print(f"  Only {n_field_patches} Prithvi patches cover "
              f"Field {config.FIELD_ID}.")

    emb_pixels= upsample_embeddings(embeddings)
    mask_flat= mask_224_clean.ravel() == 1
    emb_pixels[~mask_flat] = 0.0

    #bic clustering 
    result = run_clustering(
        emb_pixels  = emb_pixels,
        chip_enriched    = chip_enriched,
        mask_224  = mask_224_clean,
        ndvi   = ndvi,
        n_field_patches  = n_field_patches,
    )
    print(f'\n  BIC-optimal zones: {result.optimal_n}  '
          f'|  BIC scores: {[f"{s:.0f}" for s in result.bic_scores]}')

    result.temporal_dates = used_dates
    result.n_dates   = len(used_dates)

    #dashboard
    build_dashboard(
        rgb    = rgb,
        nir_false   = nir_false,
        ndvi_display = ndvi_display,
        ndvi    = ndvi,
        feature_map = feature_map,
        chip   = chip,
        chip_enriched= chip_enriched,
        mask_224  = mask_224,
        mask_clean = mask_224_clean,
        emb_pixels  = emb_pixels,
        result   = result,
        device  = device,
        meta  = meta,
        save_path  = config.dashboard_path(),
    )

    #geoTIFF export
    save_geotiff(result, meta, config.tif_path())
    print('Processing complete.')

 
if __name__ == '__main__':
    main()