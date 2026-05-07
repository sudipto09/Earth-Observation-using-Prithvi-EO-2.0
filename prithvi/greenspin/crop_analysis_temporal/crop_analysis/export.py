"""
export.py

Writes the final cluster map as a 2-band GeoTIFF (phenotype labels + confidence)
with correct CRS and geotransform from the metadata JSON. Also writes a
_metrics.json sidecar with all cluster statistics and the verdict string.

"""
import json
import os
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin
from config import CHIP_SIZE
from clustering import ClusterResult


def save_geotiff(result: ClusterResult, meta: dict, save_path: str) -> None: 

   
    if 'chip_gt' in meta:
        cgt = meta['chip_gt']
        px_w = abs(cgt[1])
        px_h = abs(cgt[5])
        chip_origin_x = cgt[0]
        chip_origin_y = cgt[3]
    else:
        gt = meta['source_gt']
        px_w = gt[1]
        px_h = -gt[5]  

        chip_origin_x = gt[0] + meta['x1_px'] * px_w
        chip_origin_y = gt[3] + meta['y1_px'] * gt[5]

    geo_transform = from_origin(chip_origin_x, chip_origin_y, px_w, px_h)

    #Save GeoTIFF
    with rasterio.open(
        save_path,
        'w',
        driver='GTiff',
        height=CHIP_SIZE,
        width=CHIP_SIZE,
        count=2,
        dtype='float32',
        crs=CRS.from_wkt(meta['crs_wkt']),
        transform=geo_transform,
    ) as dst:
        dst.write(result.pixel_cluster_map.astype('float32'), 1)
        dst.write(result.confidence_map.astype('float32'), 2)

    print(f"GeoTIFF saved: {save_path}")

    #Save metrics JSON 
    json_path = save_path.replace('.tif', '_metrics.json')

    metrics = {
        'optimal_n': int(result.optimal_n),
        'silhouette': float(result.silhouette),
        'db_score': float(result.db_score),
        'ndvi_diff': float(result.ndvi_diff),
        'avg_confidence': float(result.avg_confidence),
        'cluster_counts': [int(c) for c in result.cluster_counts],
        'cluster_pct': [float(p) for p in result.cluster_pct],
        'cluster_ndvi_avg': [float(v) for v in result.cluster_ndvi_avg],
        'cluster_ndvi_std': [float(v) for v in result.cluster_ndvi_std],
        'cluster_conf_avg': [float(v) for v in result.cluster_conf_avg],
        'verdict': result.verdict,
        'crop_names': result.crop_names,
        'temporal_dates': result.temporal_dates,
        'n_dates': int(result.n_dates),
    }

    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics saved: {json_path}")