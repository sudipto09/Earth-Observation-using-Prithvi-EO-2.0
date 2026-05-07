"""
start.py

Post-processing aggregator. Reads existing FID_* output folders, loads
classifications.pkl (or rebuilds from _metrics.json / TIF), and regenerates
batch_summary.csv and batch_summary_final.png without re-running any clustering.


"""
from __future__ import annotations

import os
import glob
import json
import pickle
import csv
import traceback
import numpy as np
import rasterio
from dataclasses import dataclass


OUTPUT_PATH = r'C:\Users\Sudipto\internship\EO\prithvi\greenspin\multi_crop_output'



@dataclass
class FieldClassification:
    field_id:  int
    label:str
    n_phenotypes_effective: int
    n_phenotypes_raw: int
    ndvi_diff: float
    silhouette:float
    db_score:float
    dominant_pct: float
    fragmentation:float
    avg_confidence:float
    n_field_pixels: int
    verdict: str



def _make_nir_false(chip: np.ndarray) -> np.ndarray:
    
    def _norm(arr: np.ndarray) -> np.ndarray:
        lo, hi = np.percentile(arr, 2), np.percentile(arr, 98)
        return np.clip((arr - lo) / (hi - lo + 1e-9), 0.0, 1.0)
    return np.stack([_norm(chip[3]), _norm(chip[2]), _norm(chip[1])], axis=-1)


def _load_nir_false(field_folder: str, field_id: int) -> np.ndarray | None:
    pattern = os.path.join(field_folder, f'prithvi_input_FID{field_id}_*.npy')
    for p in sorted(glob.glob(pattern)):
        try:
            chip = np.load(p, mmap_mode='r')
            if chip.ndim == 3 and chip.shape[0] >= 4:
                return _make_nir_false(np.array(chip))
        except Exception:
            continue
    return None


def _load_mask(field_folder: str, field_id: int) -> np.ndarray | None:
    pattern = os.path.join(field_folder, f'prithvi_mask_FID{field_id}_*.npy')
    for p in sorted(glob.glob(pattern)):
        try:
            m = np.load(p).astype(np.float32)
            if m.sum() > 0:
                return m
        except Exception:
            continue
    return None


def _load_ndvi_trajectories(
    per_date_csv: str,
    n_temporal_pheno: int,
) -> tuple[list[list[float]], list[str]] | None:
    
    if not os.path.exists(per_date_csv):
        return None
    try:
        dates: list[str] = []
        per_date_vals: list[list[float]] = []

        with open(per_date_csv, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                dates.append(row['date'])
                raw  = row.get('cluster_ndvi_means', '')
                if raw is None:
                    raw = ''
                raw = str(raw).strip()
                vals = [float(v) for v in raw.split(';') if v.strip()] if raw else []
                per_date_vals.append(vals)

        if not dates:
            return None

        n_pheno = max(n_temporal_pheno, 1)
        trajectories: list[list[float]] = [
            [row[i] if i < len(row) else float('nan') for row in per_date_vals]
            for i in range(n_pheno)
        ]
        return trajectories, dates

    except Exception:
        traceback.print_exc()
        return None


def _clf_from_pkl(c) -> FieldClassification:
   
    return FieldClassification(
        field_id  = int(c.field_id),
        label    = c.label,
        n_phenotypes_effective = c.n_phenotypes_effective,
        n_phenotypes_raw = c.n_phenotypes_raw,
        ndvi_diff    = float(c.ndvi_diff),
        silhouette   = float(c.silhouette),
        db_score  = float(c.db_score),
        dominant_pct   = float(c.dominant_pct),
        fragmentation  = float(c.fragmentation),
        avg_confidence  = float(c.avg_confidence),
        n_field_pixels  = int(c.n_field_pixels),
        verdict  = c.verdict,
    )


def _derive_label(ndvi_diff: float, silhouette: float, n_pheno: int) -> str:
    if n_pheno <= 1:
        return 'homogeneous'
    if ndvi_diff >= 0.15:
        return 'multi-crop'
    if ndvi_diff >= 0.05 and silhouette >= 0.10:
        return 'intra-crop'
    return 'weakly-variable'



pkl_path = os.path.join(OUTPUT_PATH, 'classifications.pkl')
pkl_all: list = []
pkl_by_fid: dict[int, object] = {}

if os.path.exists(pkl_path):
    with open(pkl_path, 'rb') as _f:
        pkl_all = pickle.load(_f)
    pkl_by_fid = {c.field_id: c for c in pkl_all}
    print(f"Loaded {len(pkl_all)} classifications from pkl "
          f"({sum(1 for c in pkl_all if c.label in ('multi-crop','intra-crop'))} segmented)")
else:
    print("classifications.pkl not found, will build from TIF folders only")



clf_by_fid: dict[int, FieldClassification] = {}

# Seed from pkl 
for c in pkl_all:
    clf_by_fid[c.field_id] = _clf_from_pkl(c)


field_folders = sorted(glob.glob(os.path.join(OUTPUT_PATH, 'FID_*')))
print(f"Found {len(field_folders)} FID_* folders\n")

thumbnail_data: dict[int, dict] = {}

for field_folder in field_folders:
    field_id = int(os.path.basename(field_folder).replace('FID_', ''))

    cluster_tif  = os.path.join(field_folder, 'prithvi_cluster_map_v2_Temporal.tif')
    metrics_json = cluster_tif.replace('.tif', '_metrics.json')

    if not os.path.exists(cluster_tif):
       
        continue

    try:
        with rasterio.open(cluster_tif) as src:
            cluster_map    = src.read(1).astype(int)
            confidence_map = (src.read(2).astype(np.float32)
                              if src.count >= 2 else None)

        valid_pixels = cluster_map[cluster_map >= 0]
        if valid_pixels.size == 0:
            print(f"  FID {field_id}: no valid pixels, skipping thumbnail")
            continue

        
        if field_id not in clf_by_fid:
            if os.path.exists(metrics_json):
                with open(metrics_json) as f:
                    m = json.load(f)
                n_pheno = int(m['optimal_n'])
                ndvi_diff  = float(m['ndvi_diff'])
                avg_conf = float(m['avg_confidence'])
                silhouette = float(m.get('silhouette', 0.0))
                db_score = float(m.get('db_score',   0.0))
                verdict = m.get('verdict', '')
                unique, counts = np.unique(valid_pixels, return_counts=True)
                dominant_pct = float(counts.max()) / float(counts.sum()) * 100.0
                clf_by_fid[field_id] = FieldClassification(
                    field_id = field_id,
                    label  = _derive_label(ndvi_diff, silhouette, n_pheno),
                    n_phenotypes_effective = n_pheno,
                    n_phenotypes_raw = n_pheno,
                    ndvi_diff   = ndvi_diff,
                    silhouette = silhouette,
                    db_score   = db_score,
                    dominant_pct  = dominant_pct,
                    fragmentation  = 1.0,
                    avg_confidence = avg_conf,
                    n_field_pixels   = int((cluster_map >= 0).sum()),
                    verdict  = verdict,
                )
            else:
                n_pheno  = int(valid_pixels.max() + 1)
                avg_conf = 0.85
                if confidence_map is not None:
                    vc = confidence_map[(cluster_map >= 0) & ~np.isnan(confidence_map)]
                    avg_conf = float(np.mean(vc)) if vc.size > 0 else 0.85
                unique, counts = np.unique(valid_pixels, return_counts=True)
                clf_by_fid[field_id] = FieldClassification(
                    field_id = field_id,
                    label  = 'homogeneous' if n_pheno == 1 else 'intra-crop',
                    n_phenotypes_effective = n_pheno,
                    n_phenotypes_raw = n_pheno,
                    ndvi_diff   = 0.0,
                    silhouette = 0.0,
                    db_score   = 0.0,
                    dominant_pct  = float(counts.max()) / float(counts.sum()) * 100.0,
                    fragmentation  = 1.0,
                    avg_confidence = avg_conf,
                    n_field_pixels   = int((cluster_map >= 0).sum()),
                    verdict  = 'TIF only',
                )

        
        clf = clf_by_fid[field_id]
        nir_arr = _load_nir_false(field_folder, field_id)
        mask_arr = _load_mask(field_folder, field_id)
        if nir_arr is None:
            nir_arr = np.full((224, 224, 3), 0.15, dtype=np.float32)

        per_date_csv = os.path.join(
            field_folder, 'per_date_clustering', 'per_date_summary.csv')
        traj_result = _load_ndvi_trajectories(
            per_date_csv,
            n_temporal_pheno=clf.n_phenotypes_effective,
        )

        if traj_result is not None:
            trajectories, traj_dates = traj_result
        else:
            trajectories = [[float('nan')] for _ in range(clf.n_phenotypes_effective)]
            traj_dates = []

        thumbnail_data[field_id] = {
            'rgb':nir_arr,
            'cluster_map': cluster_map,
            'mask':  mask_arr,
            'ndvi_trajectory':  trajectories,
            'trajectory_dates': traj_dates,
        }

        src_tag = ('pkl' if field_id in pkl_by_fid
                   else 'json' if os.path.exists(metrics_json) else 'tif')
        print(f"  FID {field_id:5d} : {clf.label:16s}  "
              f"n={clf.n_phenotypes_effective}  "
              f"NDVI_diff={clf.ndvi_diff:.3f}  "
            )

    except Exception as e:
        print(f"  FID {field_id}: ERROR - {e}")
        traceback.print_exc()


classifications = sorted(clf_by_fid.values(), key=lambda c: c.field_id)


if pkl_all:
    n_total_processed = len(pkl_all)
else:
    n_total_processed = sum(
        1 for ff in field_folders
        if glob.glob(os.path.join(ff, 'prithvi_input_FID*.npy'))
    )

from collections import Counter
label_counts = Counter(c.label for c in classifications)
print(f"\nTotal processed : {n_total_processed}")
print(f"In classifications list : {len(classifications)}")
#print(f"Label breakdown  : {dict(label_counts)}")
print(f"Thumbnails available  : {len(thumbnail_data)}\n")


if not pkl_all:
    pkl_out = os.path.join(OUTPUT_PATH, 'classifications.pkl')
    with open(pkl_out, 'wb') as _f:
        pickle.dump(classifications, _f)
    print(f"Saved classifications.pkl  ({len(classifications)} entries)")
else:
    print("classifications.pkl already exists, not overwriting")


import pandas as pd
from field_classifier import is_segmented as _is_segmented

csv_rows = []
for c in classifications:
    csv_rows.append({
        'field_id': c.field_id,
        'label': c.label,
        'segmented': _is_segmented(c),
        'n_phenotypes_effective': c.n_phenotypes_effective,
        'n_phenotypes_raw': c.n_phenotypes_raw,
        'ndvi_diff':  round(c.ndvi_diff,      4),
        'silhouette':  round(c.silhouette, 3) if c.silhouette == c.silhouette else None,
        'db_score':round(c.db_score,       3),
        'dominant_pct': round(c.dominant_pct,   1),
        'fragmentation':  round(c.fragmentation,  2),
        'avg_confidence':  round(c.avg_confidence, 3),
        'n_field_pixels':c.n_field_pixels,
        'verdict':c.verdict,
    })

df = (pd.DataFrame(csv_rows)
        .sort_values(['segmented', 'ndvi_diff'], ascending=[False, False])
        .reset_index(drop=True))

csv_out = os.path.join(OUTPUT_PATH, 'batch_summary.csv')
df.to_csv(csv_out, index=False)
print(f"Saved batch_summary.csv    ({len(df)} rows)")


from reporting import build_batch_report

build_batch_report(
    classifications= classifications,
    thumbnail_data  = thumbnail_data,
    save_path  = os.path.join(OUTPUT_PATH, 'batch_summary_final.png'),
    region_name  = 'Würzburg',
    season   = '2024',
    model_name  = 'Prithvi-EO Temporal',
    n_dates  = 16,
    n_thumbnails  = 6,
    max_table_rows  = 12,
    n_total_processed= n_total_processed,
)