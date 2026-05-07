"""
batch_pipeline.py

Two-stage batch orchestrator. Stage 1 (screen_field): encodes and clusters all
fields, saves classification labels. Stage 2 (run_full_pipeline): generates
dashboards, GeoTIFFs, and per-date maps for segmented fields only. GPU memory
is freed between fields. Entry point: run_batch(field_ids, run_per_date).

"""
import os
import gc
import csv
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
from cloud_mask import make_combined_cloud_mask
from per_date_clustering import run_per_date_clustering
from field_classifier import classify_field, is_segmented, FieldClassification
from reporting.extractors import extract_per_phenotype_trajectories_with_std
import pickle



def _set_field_context(field_id: int) -> None:
    
    config.FIELD_ID = field_id
    config.FIELD_FOLDER = os.path.join(
        config.OUTPUT_PATH, f'FID_{field_id}'
    )


def _save_temporal_trajectory_csv(
    field_folder: str,
    field_id: int,
    trajectories_mean: list[list[float]],
    trajectories_std: list[list[float]],
    used_dates: list[str],
) -> None:
    
    csv_path = os.path.join(
        field_folder, f'prithvi_temporal_trajectories_FID{field_id}.csv'
    )
    try:
        with open(csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['phenotype', 'metric'] + list(used_dates))
            # Write mean trajectories
            for i, traj in enumerate(trajectories_mean):
                row = [f'P{i}', 'mean'] + [
                    ('' if v is None or (isinstance(v, float) and np.isnan(v))
                     else f'{float(v):.4f}')
                    for v in traj
                ]
                w.writerow(row)
            
            for i, traj in enumerate(trajectories_std):
                row = [f'P{i}', 'std'] + [
                    ('' if v is None or (isinstance(v, float) and np.isnan(v))
                     else f'{float(v):.4f}')
                    for v in traj
                ]
                w.writerow(row)
    except Exception as e:
        print(f'  FID {field_id}: failed to save temporal trajectory CSV  {e}')


#screening

def screen_field(
    field_id: int,
    model,
    device: torch.device,
) -> dict | None:
    
   
    _set_field_context(field_id)

    if not os.path.isdir(config.FIELD_FOLDER):
        print(f'  FID {field_id}: folder not found  ({config.FIELD_FOLDER}). Skipping.')
        return None

    # load data 
    try:
        chip_temporal, used_dates = load_temporal_chips()
    except FileNotFoundError as e:
        print(f'  FID {field_id}: {e}')
        return None

    scl_chips = load_scl_chips(used_dates)
    config.TEMPORAL_REPEATS = len(used_dates)

    mask_224 = load_mask()
    mask_224 = (mask_224 > 0.5).astype(np.float32)
    if mask_224.sum() == 0:
        print(f'  FID {field_id}: empty field mask. Skipping.')
        return None

    meta = load_meta()

    # cloud masks 
    cloud_masks_list = []
    for t, scl in enumerate(scl_chips):
        cm = make_combined_cloud_mask(
            chip = chip_temporal[t],
            scl  = scl,
            include_dark_area = config.SCL_INCLUDE_DARK_AREA,
            use_spectral_supplement = config.SCL_USE_SPECTRAL_SUPPLEMENT,
        )
        cloud_masks_list.append(cm)
    cloud_masks = np.stack(cloud_masks_list, axis=0)

    # display + composite 
    display_t, display_date_str, display_valid_pct = select_best_clear_date(
        chip_temporal = chip_temporal,
        cloud_masks = cloud_masks,
        mask_224 = mask_224,
        used_dates = used_dates,
        min_valid_pct = config.DISPLAY_MIN_VALID_PCT,
    )

    chip_display = chip_temporal[display_t]
    rgb = make_rgb(chip_display)
    nir_false = make_nir_false(chip_display)
    ndvi_display_single = np.clip(compute_ndvi(chip_display), 0, 1)

    chip_composite = make_temporal_composite(
        chip_temporal, cloud_masks, top_n_greenest=5)

    all_cloudy = (cloud_masks.sum(axis=0) == len(used_dates)).astype(np.float32)
    mask_224_clean = np.where(all_cloudy == 1, 0.0, mask_224).astype(np.float32)

    n_clean = int(mask_224_clean.sum())
    if n_clean < 100:
        print(f'  FID {field_id}: only {n_clean} clean field pixels. Skipping.')
        return None

    chip_enriched = build_enriched_chip(chip_composite)
    ndvi_composite = compute_ndvi(chip_composite)

    temporal_ndvi_stats = extract_temporal_ndvi_stats(
        chip_temporal = chip_temporal,
        cloud_masks  = cloud_masks,
        mask_224 = mask_224_clean,
    )

    # encoder 
    input_tensor = build_input_tensor(chip_temporal, device)
    patch_tokens_temporal = extract_patch_tokens(
        model, input_tensor, n_dates=len(used_dates))

    emb_avg = average_patch_tokens(patch_tokens_temporal)
    emb_avg = mask_patch_embeddings(emb_avg, mask_224_clean, chip=chip_composite)
    feature_map = make_feature_map(emb_avg, mask_224=mask_224_clean, mode='l2')

    n_field_patches = int(np.sum(~np.isnan(feature_map)))

    emb_temporal_stats = extract_temporal_emb_stats(
        patch_tokens_temporal,
        n_pca_dims = config.EMB_STAT_PCA_DIMS,
    )

    emb_pixels= upsample_embeddings(emb_avg)
    emb_temporal_pixels = upsample_embeddings(emb_temporal_stats)

    mask_flat = mask_224_clean.ravel() == 1
    emb_pixels[~mask_flat]  = 0.0
    emb_temporal_pixels[~mask_flat] = 0.0    
    temporal_ndvi_stats[~mask_flat] = 0.0

    temporal_stats = np.concatenate(
        [emb_temporal_pixels, temporal_ndvi_stats], axis=1
    )

    # clustering 
    result = run_clustering(
        emb_pixels= emb_pixels,
        chip_enriched = chip_enriched,
        mask_224  = mask_224_clean,
        ndvi  = ndvi_composite,
        n_field_patches = n_field_patches,
        temporal_stats  = temporal_stats,
    )
    result.temporal_dates = used_dates
    result.n_dates = len(used_dates)

    # classification 
    classification = classify_field(field_id, result)

    print(f'  FID {field_id}: {classification.label.upper():16s}  '
          f'n_eff={classification.n_phenotypes_effective}  '
          f'NDVI_diff={classification.ndvi_diff:.3f}  '
          f'sil={classification.silhouette:.2f}  '
          f'frag={classification.fragmentation:.1f}')

    
    return {
        'field_id': field_id,
        'classification': classification,
        'result': result,
        'meta': meta,
        'rgb' : rgb,
        'nir_false' : nir_false,
        'ndvi_display_single': ndvi_display_single,
        'ndvi_composite' : ndvi_composite,
        'feature_map' : feature_map,
        'chip_composite' : chip_composite,
        'chip_enriched'  : chip_enriched,
        'mask_224' : mask_224,
        'mask_224_clean' : mask_224_clean,
        'emb_pixels'  : emb_pixels,
        'chip_temporal' : chip_temporal,
        'cloud_masks' : cloud_masks,
        'used_dates' : used_dates,
        'display_date_str' : display_date_str,
    }


# run full pipeline only on segmented fields  

def run_full_pipeline(
    bundle: dict,
    device: torch.device,
    run_per_date: bool = False,
) -> None:
    
  
    field_id = bundle['field_id']
    _set_field_context(field_id)

    result = bundle['result']

    if run_per_date:
        try:
            run_per_date_clustering(
                chip_temporal = bundle['chip_temporal'],
                cloud_masks  = bundle['cloud_masks'],
                field_mask = bundle['mask_224_clean'],
                used_dates  = bundle['used_dates'],
                temporal_result  = result,
                output_dir = os.path.join(config.FIELD_FOLDER, 'per_date_clustering'),
                min_valid_pixels = 200,
            )
        except Exception as e:
            print(f'  FID {field_id}: per_date_clustering failed: {e}')

    # dashboard 
    try:
        build_dashboard(
            rgb  = bundle['rgb'],
            nir_false = bundle['nir_false'],
            ndvi_display = bundle['ndvi_display_single'],
            ndvi  = bundle['ndvi_composite'],
            feature_map  = bundle['feature_map'],
            chip = bundle['chip_composite'],
            chip_enriched = bundle['chip_enriched'],
            mask_224  = bundle['mask_224'],
            mask_clean   = bundle['mask_224_clean'],
            emb_pixels  = bundle['emb_pixels'],
            result   = result,
            device = device,
            meta  = bundle['meta'],
            display_date  = bundle['display_date_str'],
            chip_temporal = bundle['chip_temporal'],
            cloud_masks  = bundle['cloud_masks'],
            save_path  = config.dashboard_path(),
        )
    except Exception as e:
        print(f'  FID {field_id}: dashboard failed: {e}')

    # GeoTIFF 
    try:
        save_geotiff(result, bundle['meta'], config.tif_path())
    except Exception as e:
        print(f'  FID {field_id}: GeoTIFF export failed: {e}')


# entry point 

def run_batch(
    field_ids: list[int],
    run_per_date: bool = False,
    summary_csv_path: str | None = None,
    summary_plot_path: str | None = None,
) -> pd.DataFrame:
    
   
    os.makedirs(config.OUTPUT_PATH, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device.type.upper()}')
    print(f'Batch size: {len(field_ids)} field(s)')
    

    # load Prithvi once 
    model, _ = load_pipeline(device)

    classifications:list[FieldClassification] = []
    cluster_maps: dict[int, np.ndarray] = {}
    nir_false_imgs:dict[int, np.ndarray] = {}
    masks:  dict[int, np.ndarray] = {}
    ndvi_trajectories: dict[int, tuple] = {}
    segmented_bundles: list[dict] = []

    # screening 
    
    print('SCREENING')
    
    for fid in field_ids:
        print(f'\nField {fid}')
        try:
            bundle = screen_field(field_id=fid, model=model, device=device)
        except Exception as e:
            print(f'  FID {fid}: ERROR during screening  {e}')
            traceback.print_exc()
            bundle = None

        if bundle is None:
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
            continue

        classification = bundle['classification']
        result = bundle['result']
        classifications.append(classification)
        cluster_maps[fid] = result.pixel_cluster_map.copy()
        nir_false_imgs[fid] = bundle['nir_false'].copy()
        masks[fid] = bundle['mask_224_clean'].copy()

        
        ndvi_traj_mean, ndvi_traj_std = extract_per_phenotype_trajectories_with_std(
            chip_temporal = bundle['chip_temporal'],
            cloud_masks = bundle['cloud_masks'],
            field_mask = bundle['mask_224_clean'],
            cluster_map = result.pixel_cluster_map,
            n_phenotypes= result.optimal_n,
        )
        ndvi_trajectories[fid] = (ndvi_traj_mean, ndvi_traj_std, bundle['used_dates'], result.optimal_n)

        
        _save_temporal_trajectory_csv(
            field_folder = config.FIELD_FOLDER,
            field_id     = fid,
            trajectories_mean = ndvi_traj_mean,
            trajectories_std = ndvi_traj_std,
            used_dates   = bundle['used_dates'],
        )

        if is_segmented(classification):
            segmented_bundles.append(bundle)
        else:
            del bundle
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()

    
    print(f'{len(classifications)} screened, '
          f'{len(segmented_bundles)} flagged as segmented.')
    

    #full pipeline for segmented fields only
    if segmented_bundles:
        
        print(f'FULL PIPELINE on {len(segmented_bundles)} segmented field(s)')
        
        for bundle in segmented_bundles:
            fid = bundle['field_id']
            print(f'\nFID {fid} : '
                  f'{bundle["classification"].label} -> running full pipeline ')
            try:
                run_full_pipeline(
                    bundle = bundle,
                    device = device,
                    run_per_date = run_per_date,
                )
            except Exception as e:
                print(f'  FID {fid}: ERROR during full pipeline  {e}')
                traceback.print_exc()

            if device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
    else:
        print('\nNo segmented fields detected')

    if not classifications:
        print('\nNo fields processed successfully.')
        return pd.DataFrame()

    # build summary table
    rows = []
    for c in classifications:
        rows.append({
            'field_id'  : c.field_id,
            'label' : c.label,
            'segmented'  : is_segmented(c),
            'full_pipeline_ran' : is_segmented(c),
            'n_phenotypes_effective': c.n_phenotypes_effective,
            'n_phenotypes_raw' : c.n_phenotypes_raw,
            'ndvi_diff'  : round(c.ndvi_diff, 4),
            'silhouette' : round(c.silhouette, 3) if c.silhouette == c.silhouette else None,
            'db_score' : round(c.db_score, 3),
            'dominant_pct'  : round(c.dominant_pct, 1),
            'fragmentation' : round(c.fragmentation, 2),
            'avg_confidence' : round(c.avg_confidence, 3),
            'n_field_pixels' : c.n_field_pixels,
            'verdict' : c.verdict,
        })
    df = pd.DataFrame(rows).sort_values(
        by=['segmented', 'ndvi_diff'], ascending=[False, False]
    ).reset_index(drop=True)

    #  CSV
    if summary_csv_path is None:
        summary_csv_path = os.path.join(config.OUTPUT_PATH, 'batch_summary.csv')
    df.to_csv(summary_csv_path, index=False)
    print(f'\nSummary CSV saved: {summary_csv_path}')

    
    if summary_plot_path is None:
        summary_plot_path = os.path.join(config.OUTPUT_PATH, 'batch_summary_final.png')

    
    _thumbnail_data: dict[int, dict] = {}
    for _fid in cluster_maps:
        _traj_data = ndvi_trajectories.get(_fid)
        _traj_mean, _traj_std, _traj_dates = (_traj_data[0], _traj_data[1], _traj_data[2]) if _traj_data else ([], [], [])
        _thumbnail_data[_fid] = {
            'rgb':  nir_false_imgs.get(_fid, np.zeros((224, 224, 3), dtype=np.float32)),
            'cluster_map': cluster_maps[_fid],
            'mask': masks.get(_fid),
            'ndvi_trajectory':  _traj_mean,
            'ndvi_trajectory_std': _traj_std,
            'trajectory_dates': _traj_dates,
        }

    from reporting import build_batch_report
    build_batch_report(
        classifications = classifications,
        thumbnail_data  = _thumbnail_data,
        save_path = summary_plot_path,
        region_name = 'batch run',
        season   = str(config.DATES[0][:4]) if config.DATES else 'unknown',
        model_name   = 'Prithvi-EO Temporal',
        n_dates  = len(config.DATES),
        n_thumbnails  = 6,
        max_table_rows= 12,
        n_total_processed  = len(classifications),
    )

    #  summary
    
    print('BATCH SUMMARY')
    
    print(df[['field_id', 'label', 'n_phenotypes_effective',
             'ndvi_diff', 'silhouette', 'fragmentation',
             'full_pipeline_ran']].to_string(index=False))
    
    seg_count = int(df['segmented'].sum())
    print(f'\n{seg_count} / {len(df)} fields are segmented '
          f'(multi-crop or intra-crop). Full pipeline  run on segemented fields only.')

    

    with open(os.path.join(config.OUTPUT_PATH, 'classifications.pkl'), 'wb') as f:
        pickle.dump(classifications, f)
        
    return df


# script entry 

if __name__ == '__main__':
    
    field_ids = [
        0, 107, 115, 116, 118, 120, 123, 127, 134, 135, 138, 144, 149, 150, 153, 
        170, 1734, 178, 18, 184, 19, 190, 192, 193, 198, 2064, 207, 210, 22, 
        224, 226, 230, 233, 238, 243, 244, 25, 259, 266, 269, 276, 284, 289, 29, 
        290, 30, 306, 307, 309, 31, 310, 312, 316, 317, 321, 322, 325, 327, 33, 
        337, 340, 341, 343, 344, 347, 348, 349, 35, 357, 36, 39, 4, 48, 5, 
        50, 52, 56, 57, 6, 64, 7, 71, 72, 73, 74, 75, 81, 88, 89, 90, 92, 
        95, 97, 98, 99, 29
    ]  

    run_batch(
        field_ids = field_ids,
        run_per_date = True,    
    )