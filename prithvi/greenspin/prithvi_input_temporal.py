from qgis.core import QgsProject, QgsCoordinateTransform, QgsCoordinateReferenceSystem
import numpy as np
from osgeo import gdal, ogr, osr
import json
import os



DATE_FOLDERS = [
    "2024-04-08", "2024-04-23", "2024-04-30",
    "2024-05-10", "2024-05-13", "2024-05-20",
    "2024-05-23", "2024-05-28", "2024-05-30",
    "2024-06-04", "2024-06-07", "2024-06-09",
    "2024-06-12", "2024-06-17", "2024-06-19",
    "2024-06-22", "2024-06-24", "2024-06-27",
    "2024-06-29", "2024-07-09", "2024-07-14",
    "2024-07-17", "2024-07-19", "2024-07-22",
    "2024-07-29", "2024-08-11", "2024-08-13",
    "2024-08-16", "2024-08-21", "2024-08-23",
    "2024-08-26", "2024-08-28", "2024-08-31",
    "2024-09-02", "2024-09-05", "2024-09-07",
    "2024-09-15",
]

PATCH_ID    = '32UPA_0_4'
BASE_PATH   = r'Y:\14_Zuckerrübe\Sentinel_2\wue'
OUTPUT_PATH = r'C:\Users\Sudipto\internship\EO\prithvi\greenspin\multi_crop_output'

PRITHVI_BANDS = [1, 2, 3, 4, 5, 6]  # B2,B3,B4,B8,B11,B12
CHIP_SIZE     = 224          # Prithvi required input size


BBOX_PAD_FRACTION = 0.5


MIN_TIGHT_PX = 32


SCL_FILENAME       = 'scl.tif'
SCL_NEEDS_UPSAMPLE = True   



def _find_scl_path(date_str):
    p = os.path.join(BASE_PATH, date_str, PATCH_ID, SCL_FILENAME)
    return p if os.path.exists(p) else None


def _field_bbox_geo(geom_wkt, srs_wkt):
    
    ogr_geom = ogr.CreateGeometryFromWkt(geom_wkt)
    env = ogr_geom.GetEnvelope()   # (xmin, xmax, ymin, ymax)
    return env[0], env[2], env[1], env[3]


def _compute_tight_window(xmin, ymin, xmax, ymax, pad_frac, min_px, gt, ds):
   
    bw = xmax - xmin
    bh = ymax - ymin
    pad_x = max(bw * pad_frac, min_px * abs(gt[1]))
    pad_y = max(bh * pad_frac, min_px * abs(gt[5]))

    wx_min = xmin - pad_x
    wy_max = ymax + pad_y
    wx_max = xmax + pad_x
    wy_min = ymin - pad_y

    # Convert to pixel coords
    px_min = int((wx_min - gt[0]) / gt[1])
    py_min = int((wy_max - gt[3]) / gt[5])   
    px_max = int((wx_max - gt[0]) / gt[1])
    py_max = int((wy_min - gt[3]) / gt[5])

    
    x1 = max(0, min(px_min, px_max))
    y1 = max(0, min(py_min, py_max))
    x2 = min(ds.RasterXSize, max(px_min, px_max))
    y2 = min(ds.RasterYSize, max(py_min, py_max))

    win_w = max(x2 - x1, 1)
    win_h = max(y2 - y1, 1)
    return x1, y1, win_w, win_h


def _warp_band_to_chip(ds, band_idx, x1, y1, win_w, win_h, chip_size,resample=gdal.GRA_Bilinear):
    
    gt = ds.GetGeoTransform()
    src_xmin = gt[0] + x1 * gt[1]
    src_ymax = gt[3] + y1 * gt[5]
    src_xmax = src_xmin + win_w * gt[1]
    src_ymin = src_ymax + win_h * gt[5]

    
    mem_drv = gdal.GetDriverByName('MEM')
    src_ds  = mem_drv.Create('', win_w, win_h, 1, gdal.GDT_Float32)
    win_gt  = (src_xmin, gt[1], 0, src_ymax, 0, gt[5])
    src_ds.SetGeoTransform(win_gt)
    src_ds.SetProjection(ds.GetProjection())
    arr = ds.GetRasterBand(band_idx).ReadAsArray(x1, y1, win_w, win_h)
    src_ds.GetRasterBand(1).WriteArray(arr.astype(np.float32))

    
    dst_ds = mem_drv.Create('', chip_size, chip_size, 1, gdal.GDT_Float32)
    dst_gt = (src_xmin, (src_xmax - src_xmin) / chip_size, 0,src_ymax, 0, (src_ymin - src_ymax) / chip_size)
    dst_ds.SetGeoTransform(dst_gt)
    dst_ds.SetProjection(ds.GetProjection())
    gdal.ReprojectImage(src_ds, dst_ds, None, None, resample)
    result = dst_ds.GetRasterBand(1).ReadAsArray()
    src_ds = None; dst_ds = None
    return result.astype(np.float32)


def _rasterize_mask(geom_wkt, chip_gt, projection, chip_size):
    
    driver  = gdal.GetDriverByName('MEM')
    mask_ds = driver.Create('', chip_size, chip_size, 1, gdal.GDT_Byte)
    mask_ds.SetGeoTransform(chip_gt)
    mask_ds.SetProjection(projection)

    ogr_ds = ogr.GetDriverByName('Memory').CreateDataSource('')
    srs= osr.SpatialReference(); srs.ImportFromWkt(projection)
    lyr  = ogr_ds.CreateLayer('field', srs=srs)
    feat = ogr.Feature(lyr.GetLayerDefn())
    feat.SetGeometry(ogr.CreateGeometryFromWkt(geom_wkt))
    lyr.CreateFeature(feat)
    gdal.RasterizeLayer(mask_ds, [1], lyr, burn_values=[1])

    mask    = mask_ds.GetRasterBand(1).ReadAsArray()
    mask_ds = None; ogr_ds = None
    return mask


def _warp_scl_to_chip(scl_path, x1_geo, y1_geo, x2_geo, y2_geo, chip_size, projection):
    
    scl_ds = gdal.Open(scl_path)
    if scl_ds is None:
        return None
    warp_opts = gdal.WarpOptions(
        format = 'MEM',
        outputBounds = (x1_geo, y2_geo, x2_geo, y1_geo),
        width   = chip_size,
        height   = chip_size,
        resampleAlg = gdal.GRA_NearestNeighbour,
        outputType= gdal.GDT_Byte,
    )
    warped = gdal.Warp('', scl_ds, options=warp_opts)
    result = warped.GetRasterBand(1).ReadAsArray().astype(np.uint8)
    scl_ds = None; warped = None
    return result


def _scl_summary(scl):
    labels = {0:'NO_DATA',1:'SAT',2:'DARK',3:'SHADOW',4:'VEG',5:'NOT_VEG',6:'WATER',7:'UNCLASS',8:'CLOUD_M',
                9:'CLOUD_H',10:'CIRRUS',11:'SNOW'}
    parts = [f"{labels.get(int(c),str(c))}={int((scl==c).sum())}"
             for c in np.unique(scl)]
    return '  '.join(parts)


#main

os.makedirs(OUTPUT_PATH, exist_ok=True)

LAYER_NAME = 'sb_fields_wue_2024'
layer = QgsProject.instance().mapLayersByName(LAYER_NAME)[0]
selected = layer.selectedFeatures()

if not selected:
    print("No field selected.")
else:
    
    for field in selected:
        
        field_folder = os.path.join(OUTPUT_PATH, f"FID_{field.id()}")
        os.makedirs(field_folder, exist_ok=True)

        raster_crs = QgsCoordinateReferenceSystem("EPSG:32632")
        to_raster  = QgsCoordinateTransform(layer.crs(), raster_crs,QgsProject.instance())

        # Transform field geometry to raster CRS
        geom = field.geometry()
        geom.transform(to_raster)
        geom_wkt = geom.asWkt()

        #geographic bbox of the field
        xmin_geo, ymin_geo, xmax_geo, ymax_geo = _field_bbox_geo(
            geom_wkt, raster_crs.toWkt())

        bw = xmax_geo - xmin_geo
        bh = ymax_geo - ymin_geo
        print(f"\nField {field.id()}")

        n_ok = 0; n_skip = 0; n_scl_ok = 0; n_scl_miss = 0

        for current_date in DATE_FOLDERS:
            img_path = os.path.join(BASE_PATH, current_date, PATCH_ID, 'bands.tif')
            ds = gdal.Open(img_path)
            if ds is None:
                print(f"SKIP {current_date}: cannot open {img_path}")
                n_skip += 1
                continue

            try:
                gt = ds.GetGeoTransform()

                # Compute tight window in source pixel coords
                x1, y1, win_w, win_h = _compute_tight_window(
                    xmin_geo, ymin_geo, xmax_geo, ymax_geo,
                    BBOX_PAD_FRACTION, MIN_TIGHT_PX, gt, ds)

                # Geographic extent of the tight window (for SCL warp)
                win_xmin = gt[0] + x1 * gt[1]
                win_ymax = gt[3] + y1 * gt[5]
                win_xmax = win_xmin + win_w * gt[1]
                win_ymin = win_ymax + win_h * gt[5]

                # Native pixel size and resampled pixel size
                native_px  = abs(gt[1])
                resamp_px  = (win_w * native_px) / CHIP_SIZE
                field_patches = int((bw / resamp_px / 16) * (bh / resamp_px / 16))

                print(f"\n{current_date}")
                #print(f"   Tight window: {win_w}x{win_h}px  "
                 #     f"native={native_px:.0f}m  resampled={resamp_px:.1f}m/px")
                #print(f"   Field covers ~{field_patches} Prithvi patches "
                 #     f"(was 4 before)")

                #Spectral chip
                bands = [_warp_band_to_chip(ds, b, x1, y1, win_w, win_h,
                                            CHIP_SIZE, gdal.GRA_Bilinear)
                         for b in PRITHVI_BANDS]
                chip = np.stack(bands, axis=0)   # (6, 224, 224)

                chip_file = os.path.join(
                    field_folder,
                    f"prithvi_input_FID{field.id()}_{current_date}.npy")
                np.save(chip_file, chip)
                print(f"   chip  {chip.shape}  -> {os.path.basename(chip_file)}")

                #Chip geotransform
                chip_gt = (win_xmin,
                           (win_xmax - win_xmin) / CHIP_SIZE,
                           0,
                           win_ymax,
                           0,
                           (win_ymin - win_ymax) / CHIP_SIZE)

                #Field mask
                mask = _rasterize_mask(geom_wkt, chip_gt,
                                       ds.GetProjection(), CHIP_SIZE)
                mask_file = os.path.join(
                    field_folder,
                    f"prithvi_mask_FID{field.id()}_{current_date}.npy")
                np.save(mask_file, mask)
                print(f"   mask  field_px={int(mask.sum())}  "
                      f"-> {os.path.basename(mask_file)}")

                #SCL chip
                scl_src = _find_scl_path(current_date)
                if scl_src:
                    scl_chip = _warp_scl_to_chip(
                        scl_src, win_xmin, win_ymax, win_xmax, win_ymin,
                        CHIP_SIZE, ds.GetProjection())
                    if scl_chip is not None:
                        scl_file = os.path.join(
                            field_folder,
                            f"prithvi_scl_FID{field.id()}_{current_date}.npy")
                        np.save(scl_file, scl_chip)
                        print(f"   SCL   {_scl_summary(scl_chip)}")
                        n_scl_ok += 1
                    else:
                        print(f"   SCL warp failed")
                        n_scl_miss += 1
                else:
                    print(f"   SCL   not found (spectral fallback)")
                    n_scl_miss += 1

                # metadata
                meta = {
                    "field_id"  : int(field.id()),
                    "date"    : current_date,
                    "x1_px" :     int(x1),
                    "y1_px"    : int(y1),
                    "win_w_px"   : int(win_w),
                    "win_h_px"  : int(win_h),
                    "chip_size" : CHIP_SIZE,
                    "source_gt" : list(gt),
                    "chip_gt"  : list(chip_gt),   
                    "crs_wkt" : ds.GetProjection(),
                    "resamp_px_m"  : float(resamp_px),
                    "field_patches": int(field_patches),
                    "scl_saved" : scl_src is not None,
                }
                meta_file = os.path.join(
                    field_folder,
                    f"prithvi_meta_FID{field.id()}_{current_date}.json")
                with open(meta_file, 'w') as f:
                    json.dump(meta, f, indent=2)
                print(f"   meta  -> {os.path.basename(meta_file)}")
                n_ok += 1

            except Exception as e:
                import traceback
                print(f"   ERROR {current_date}: {e}")
                traceback.print_exc()
                n_skip += 1
            finally:
                ds = None

        print(f"""

  Dates processed : {n_ok}
  Dates skipped   : {n_skip}
  SCL saved       : {n_scl_ok}
  SCL missing     : {n_scl_miss}
""")