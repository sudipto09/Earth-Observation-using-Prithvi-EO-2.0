"""
main.py

Single-field entry point for development and debugging. Runs the complete
pipeline for config.FIELD_ID with full console output. For batch processing
of many fields use screening.py / batch_pipeline.py instead.

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
    average_patch_tokens,
    extract_temporal_emb_stats,
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
    select_best_clear_date,
    extract_temporal_ndvi_stats,
)
from visualization import build_dashboard
from cloud_mask import make_combined_cloud_mask, scl_coverage_report
from per_date_clustering import run_per_date_clustering


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

    # Select best single clear date 
    display_t, display_date_str, display_valid_pct = select_best_clear_date(
        chip_temporal = chip_temporal,
        cloud_masks  = cloud_masks,
        mask_224  = mask_224,
        used_dates  = used_dates,
        min_valid_pct  = config.DISPLAY_MIN_VALID_PCT,
    )
    print(f'\n  Display date: {display_date_str}  '
          f'(field coverage: {display_valid_pct * 100:.1f}% clean)')

    # use the best single clear date
    chip_display  = chip_temporal[display_t]
    rgb  = make_rgb(chip_display)
    nir_false  = make_nir_false(chip_display)
    ndvi_display_single = np.clip(compute_ndvi(chip_display), 0, 1)

    # Temporal composite chip
    chip_composite = make_temporal_composite(
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

    # enriched composite chip 
    chip_enriched = build_enriched_chip(chip_composite)

    # composite NDVI 
    ndvi_composite = compute_ndvi(chip_composite)
    field_ndvi_comp = float(np.mean(ndvi_composite[mask_224_clean == 1])) if n_clean > 0 else 0.0
    print(f'  Composite NDVI over field: {field_ndvi_comp:.3f}')

    # temporal NDVI statistics per pixel (7 features: mean, peak, min, std, timing, rates)
    print('  Extracting temporal NDVI statistics...')
    temporal_ndvi_stats = extract_temporal_ndvi_stats(
        chip_temporal = chip_temporal,
        cloud_masks = cloud_masks,
        mask_224  = mask_224_clean,
    )   

    #prithvi encoder 
    input_tensor= build_input_tensor(chip_temporal, device)
    patch_tokens_temporal= extract_patch_tokens(model, input_tensor, n_dates=len(used_dates))

    # averaged tokens 
    emb_avg = average_patch_tokens(patch_tokens_temporal)              
    emb_avg = mask_patch_embeddings(emb_avg, mask_224_clean, chip=chip_composite)
    feature_map = make_feature_map(emb_avg, mask_224=mask_224_clean, mode='l2')

    n_field_patches=int(np.sum(~np.isnan(feature_map)))
    if n_field_patches < config.MIN_VALID_PATCHES:
        print(f"  Only {n_field_patches} Prithvi patches cover "
              f"Field {config.FIELD_ID}.")

    # per-date temporal embedding statistics 
    print('  Extracting temporal embedding statistics...')
    emb_temporal_stats = extract_temporal_emb_stats(
        patch_tokens_temporal,
        n_pca_dims = config.EMB_STAT_PCA_DIMS,
    )   
    
    emb_pixels   = upsample_embeddings(emb_avg)          
    emb_temporal_pixels = upsample_embeddings(emb_temporal_stats) 

    mask_flat = mask_224_clean.ravel() == 1
    emb_pixels[~mask_flat]   = 0.0
    emb_temporal_pixels[~mask_flat] = 0.0
    temporal_ndvi_stats[~mask_flat] = 0.0

    # fused temporal feature matrix 
    temporal_stats = np.concatenate(
        [emb_temporal_pixels, temporal_ndvi_stats], axis=1
    )   

    #bic clustering
    result = run_clustering(
        emb_pixels  = emb_pixels,
        chip_enriched  = chip_enriched,
        mask_224    = mask_224_clean,
        ndvi      = ndvi_composite,
        n_field_patches = n_field_patches,
        temporal_stats  = temporal_stats,
    )
    print(f'\n  BIC-optimal phenotypes: {result.optimal_n}  '
          f'|  BIC scores: {[f"{s:.0f}" for s in result.bic_scores]}')
    print(f'  Silhouette: {result.silhouette:.3f}  '
          f'|  Davies-Bouldin: {result.db_score:.3f}')

    result.temporal_dates = used_dates
    result.n_dates   = len(used_dates)

    #per-date clustering for temporal vs snapshot comparison
    print('\nRunning per-date single-date clustering...')
    run_per_date_clustering(
        chip_temporal  = chip_temporal,
        cloud_masks = cloud_masks,
        field_mask   = mask_224_clean,
        used_dates= used_dates,
        temporal_result  = result,
        output_dir= os.path.join(config.FIELD_FOLDER, 'per_date_clustering'),
        min_valid_pixels = 200,
    )

    #dashboard
    build_dashboard(
        rgb      = rgb,
        nir_false  = nir_false,
        ndvi_display  = ndvi_display_single,
        ndvi   = ndvi_composite,
        feature_map   = feature_map,
        chip     = chip_composite,
        chip_enriched = chip_enriched,
        mask_224  = mask_224,
        mask_clean  = mask_224_clean,
        emb_pixels  = emb_pixels,
        result  = result,
        device = device,
        meta    = meta,
        display_date  = display_date_str,
        chip_temporal = chip_temporal,
        cloud_masks  = cloud_masks,
        save_path  = config.dashboard_path(),
    )

    #geoTIFF export
    save_geotiff(result, meta, config.tif_path())
    print('Processing complete.')

 
if __name__ == '__main__':
    main()