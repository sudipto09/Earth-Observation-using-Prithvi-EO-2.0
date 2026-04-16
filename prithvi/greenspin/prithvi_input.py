from qgis.core import QgsProject, QgsCoordinateTransform, QgsCoordinateReferenceSystem
import numpy as np
from osgeo import gdal, ogr, osr
import json
import os
#config
DATE_FOLDER  = '2024-07-14'
PATCH_ID = '32UPA_0_4'
BASE_PATH = r'Y:\14_Zuckerrübe\Sentinel_2\wue'
OUTPUT_PATH= r'C:\Users\Sudipto\internship\EO\prithvi\greenspin\multi_crop_output'
# B2 (Blue), B3 (Green), B4 (Red), B8 (NIR), B11 (SWIR1), B12 (SWIR2)
PRITHVI_BANDS = [1, 2, 3, 4, 5, 6]
CHIP_SIZE = 224   #prithvi's required input size in pixels
HALF = CHIP_SIZE // 2

os.makedirs(OUTPUT_PATH, exist_ok=True)
#load field
LAYER_NAME = 'sb_fields_wue_2024'
layer    = QgsProject.instance().mapLayersByName(LAYER_NAME)[0]
selected = layer.selectedFeatures()
if not selected:
    print("No field selected.")
else:
    field = selected[0]
    #field centroid into the raster CRS
    raster_crs  = QgsCoordinateReferenceSystem("EPSG:32632")
    to_raster   = QgsCoordinateTransform(layer.crs(), raster_crs, QgsProject.instance())
    centroid     = field.geometry().centroid().asPoint()
    centroid_utm = to_raster.transform(centroid)
    #utm coords to pixel indices
    img_path = os.path.join(BASE_PATH, DATE_FOLDER, PATCH_ID, 'bands.tif')
    ds = gdal.Open(img_path)
    if ds is None:
        print(f"Could not open raster at:\n  {img_path}")
    else:
        gt = ds.GetGeoTransform()
        # gt[0], gt[3] = top left corner
        # gt[1]= pixel width
        # gt[5]= pixel height
        cx = int((centroid_utm.x() - gt[0]) / gt[1])
        cy = int((centroid_utm.y() - gt[3]) / gt[5])
        #pin the top left corner of the chip
        x1 = max(0, min(cx - HALF, ds.RasterXSize - CHIP_SIZE))
        y1 = max(0, min(cy - HALF, ds.RasterYSize - CHIP_SIZE))
        print(f"Field {field.id()} centroid -> pixel ({cx}, {cy})")
        print(f"Chip top-left clamped to pixel ({x1}, {y1})")
        print(f"Extracting {CHIP_SIZE}×{CHIP_SIZE} chip from {DATE_FOLDER}...")
        #read bands and stack
        try:
            bands = [
                ds.GetRasterBand(b).ReadAsArray(x1, y1, CHIP_SIZE, CHIP_SIZE)
                for b in PRITHVI_BANDS
            ]
            chip = np.stack(bands, axis=0).astype(np.float32)
            
            #save chip
            filename  = f"prithvi_input_FID{field.id()}_{DATE_FOLDER}.npy"
            save_path = os.path.join(OUTPUT_PATH, filename)
            np.save(save_path, chip)
            print(f"Saved {chip.shape} array → {save_path}")

            #field polygon into  224 x 224 boolean mask
            geom = field.geometry()
            geom.transform(to_raster)
            wkt = geom.asWkt()

            driver  = gdal.GetDriverByName('MEM')
            mask_ds = driver.Create('', CHIP_SIZE, CHIP_SIZE, 1, gdal.GDT_Byte)

            #geotransform aligned to chip 
            chip_gt = (
                gt[0] + x1 * gt[1],# chip origin X
                gt[1],# pixel width
                0,
                gt[3] + y1 * gt[5],   # chip origin Y
                0,
                gt[5] # pixel height (-)
            )
            mask_ds.SetGeoTransform(chip_gt)
            mask_ds.SetProjection(ds.GetProjection())

            #rasterize field polygon into mask
            ogr_ds = ogr.GetDriverByName('Memory').CreateDataSource('')
            srs    = osr.SpatialReference(); srs.ImportFromWkt(ds.GetProjection())
            lyr    = ogr_ds.CreateLayer('field', srs=srs)
            feat   = ogr.Feature(lyr.GetLayerDefn())
            feat.SetGeometry(ogr.CreateGeometryFromWkt(wkt))
            lyr.CreateFeature(feat)
            gdal.RasterizeLayer(mask_ds, [1], lyr, burn_values=[1])

            mask     = mask_ds.GetRasterBand(1).ReadAsArray()
            mask_ds  = None
            ogr_ds   = None

            mask_path = os.path.join(OUTPUT_PATH, f"prithvi_mask_FID{field.id()}_{DATE_FOLDER}.npy")
            np.save(mask_path, mask)
            print(f"Saved field mask {mask.shape} at {mask_path}")

            #save georef metadata 
            meta = {
                "field_id"  : int(field.id()),
                "date"      : DATE_FOLDER,
                "x1_px"     : int(x1),#chip top left column
                "y1_px"     : int(y1),#chip top left row 
                "chip_size" : CHIP_SIZE,
                "source_gt" : list(gt),
                "crs_wkt"   : ds.GetProjection()
            }
            meta_path = os.path.join(
                OUTPUT_PATH,
                f"prithvi_meta_FID{field.id()}_{DATE_FOLDER}.json"
            )
            with open(meta_path, 'w') as f:
                json.dump(meta, f, indent=2)
            print(f"Saved metadata :{meta_path}")
        except Exception as e:
            print(f"Extraction failed: {e}")
        finally:
            ds = None