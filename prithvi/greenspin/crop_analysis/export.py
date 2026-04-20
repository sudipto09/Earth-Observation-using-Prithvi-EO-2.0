"""
export.py

Writes the cluster map and confidence raster as a two-band GeoTIFF.
"""
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin
from config import CHIP_SIZE
from clustering import ClusterResult


def save_geotiff(result: ClusterResult, meta: dict, save_path: str) -> None:
    """
    Write a two-band GeoTIFF:
        Band 1 - integer cluster labels
        Band 2 - per-pixel GMM confidence

   
    """
    gt  = meta['source_gt']
    x1  = meta['x1_px']
    y1  = meta['y1_px']
    px_w =  gt[1]
    px_h = -gt[5]

    chip_origin_x = gt[0] + x1 * px_w
    chip_origin_y = gt[3] + y1 * gt[5]

    geo_transform = from_origin(chip_origin_x, chip_origin_y, px_w, px_h)

    with rasterio.open(
        save_path, 'w',
        driver   = 'GTiff',
        height  = CHIP_SIZE,
        width  = CHIP_SIZE,
        count  = 2,
        dtype   = 'float32',
        crs     = CRS.from_wkt(meta['crs_wkt']),
        transform = geo_transform,
    ) as dst:
        dst.write(result.pixel_cluster_map.astype('float32'), 1)
        dst.write(result.confidence_map.astype('float32'),    2)

    print(f"GeoTIFF saved:  {save_path}")