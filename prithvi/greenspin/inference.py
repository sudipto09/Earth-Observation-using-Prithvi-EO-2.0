import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS
from modelfactory import load_pipeline
import json
import os

#config
FIELD_ID= 920
DATE = '2024-05-13'
OUTPUT_PATH= r'C:\Users\Sudipto\internship\EO\prithvi\greenspin\multi_crop_output'

CHIP_PATH= os.path.join(OUTPUT_PATH, f'prithvi_input_FID{FIELD_ID}_{DATE}.npy')
META_PATH= os.path.join(OUTPUT_PATH, f'prithvi_meta_FID{FIELD_ID}_{DATE}.json')
MASK_PATH= os.path.join(OUTPUT_PATH, f'prithvi_mask_FID{FIELD_ID}_{DATE}.npy')

TEMPORAL_REPEATS = 4
PATCH_GRID= 14
CHIP_SIZE= 224       
N_CLUSTERS= 2
RANDOM_SEED= 42 

os.makedirs(OUTPUT_PATH, exist_ok=True) 

#device,model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")
model, decoder = load_pipeline(device)


#shape: (6, 224, 224) — bands b2 b3 b4 b8 b11 b12
chip = np.load(CHIP_PATH)
print(f"Chip loaded: {chip.shape}")

# Load field mask (224×224)
mask_224 = np.load(MASK_PATH).astype(np.float32)   # shape (224, 224), values 0/1
print(f"Mask unique values: {np.unique(mask_224)}|sum: {mask_224.sum()}")

#stretch a band to 0–1 using 2nd/98th percentile
def norm(arr):
    lo, hi = np.percentile(arr, 2), np.percentile(arr, 98)
    return np.clip((arr - lo) / (hi - lo + 1e-9), 0, 1)

#display images
rgb= np.stack([norm(chip[2]), norm(chip[1]), norm(chip[0])], axis=-1)
nir_false = np.stack([norm(chip[3]), norm(chip[2]), norm(chip[1])], axis=-1)
ndvi= (chip[3] - chip[2]) / (chip[3] + chip[2] + 1e-6)
ndvi_display = np.clip(ndvi, 0, 1)

#run prithvi encoder 
input_tensor = (
    torch.from_numpy(chip).float()
    .unsqueeze(0).unsqueeze(2)
    .repeat(1, 1, TEMPORAL_REPEATS, 1, 1)   # (1, 6, 4, 224, 224)
    .to(device)
) 

print("Running encoder...")
with torch.no_grad():
    last_block= model.forward_features(input_tensor)[-1]
    patch_tokens = last_block[:, 1:, :]
    patch_tokens = patch_tokens.reshape(1, TEMPORAL_REPEATS, 196, 1024).mean(dim=1)

embeddings = patch_tokens.squeeze(0).cpu().numpy()   # (196, 1024)
feature_map= np.linalg.norm(embeddings, axis=1).reshape(PATCH_GRID, PATCH_GRID)

#pixel-level clustering inside field mask
print("Clustering field pixels...")
pixels= chip.reshape(6, -1).T                  
field_pixels = pixels[mask_224.ravel() == 1]          # only field pixels
print(f"Field pixels: {len(field_pixels)}")

scaler = StandardScaler()
field_pixels_scaled = scaler.fit_transform(field_pixels)

kmeans= KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_SEED, n_init='auto')
pixel_labels = kmeans.fit_predict(field_pixels_scaled)

# full 224×224 label map (2 : outside field)
pixel_cluster_map= np.full(224 * 224, 2, dtype=int)
pixel_cluster_map[mask_224.ravel() == 1] = pixel_labels
pixel_cluster_map = pixel_cluster_map.reshape(224, 224)

#NDVI per field pixel
ndvi_flat= ndvi.ravel()
field_ndvi= ndvi_flat[mask_224.ravel() == 1]

#per-cluster stats
cluster_counts= [int(np.sum(pixel_labels == i)) for i in range(N_CLUSTERS)]
cluster_ndvi_avg= [float(np.mean(field_ndvi[pixel_labels == i])) for i in range(N_CLUSTERS)]
cluster_pct = [c / len(pixel_labels) * 100 for c in cluster_counts]

#greener cluster as crop a
greener_idx = int(np.argmax(cluster_ndvi_avg))
crop_names = ['', '']
crop_names[greener_idx]   = 'Crop A  (higher NDVI)'
crop_names[1-greener_idx] = 'Crop B  (lower NDVI)'

#dashboard
fig = plt.figure(figsize=(16, 12), facecolor='#0f0f0f')
fig.suptitle(
    f'Prithvi field analysis  |  Field {FIELD_ID}  |  {DATE}',
    fontsize=15, color='white', fontweight='bold', y=0.98
)

gs = gridspec.GridSpec(3, 3, figure=fig,
                       hspace=0.38, wspace=0.25,
                       height_ratios=[1, 1, 0.38])

ax = [
    fig.add_subplot(gs[0, 0]),
    fig.add_subplot(gs[0, 1]),
    fig.add_subplot(gs[0, 2]),
    fig.add_subplot(gs[1, 0]),
    fig.add_subplot(gs[1, 1]),
    fig.add_subplot(gs[1, 2]),
    fig.add_subplot(gs[2, :]),
]

LABEL_COLOR = '#cccccc'
TITLE_COLOR = 'white'

def style(a, title, xlabel='', ylabel=''):
    a.set_title(title, color=TITLE_COLOR, fontsize=11, pad=6)
    a.set_facecolor('#1a1a1a')
    a.tick_params(colors=LABEL_COLOR, labelsize=8)
    for spine in a.spines.values():
        spine.set_edgecolor('#444444')
    if xlabel: a.set_xlabel(xlabel, color=LABEL_COLOR, fontsize=8)
    if ylabel: a.set_ylabel(ylabel, color=LABEL_COLOR, fontsize=8)

#1. true colour
ax[0].imshow(rgb)
ax[0].set_xticks([]); ax[0].set_yticks([])
style(ax[0], 'True colour  (RGB)')
ax[0].text(4, 218, 'Bands 3-2-1', color=LABEL_COLOR, fontsize=7, va='bottom')

#2. false colour NIR
ax[1].imshow(nir_false)
ax[1].set_xticks([]); ax[1].set_yticks([])
style(ax[1], 'False colour NIR  (crops glow red)')
ax[1].text(4, 218, 'Bands 4-3-2', color=LABEL_COLOR, fontsize=7, va='bottom')

#3. NDVI heatmap
im2 = ax[2].imshow(ndvi_display, cmap='YlGn', vmin=0, vmax=1)
ax[2].set_xticks([]); ax[2].set_yticks([])
style(ax[2], 'NDVI  (greenness per pixel)')
cb2 = fig.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)
cb2.ax.yaxis.set_tick_params(color=LABEL_COLOR, labelsize=7)
cb2.outline.set_edgecolor('#444444')
plt.setp(cb2.ax.yaxis.get_ticklabels(), color=LABEL_COLOR)

#4. feature intensity
im3 = ax[3].imshow(feature_map, cmap='viridis', interpolation='nearest')
for x in range(PATCH_GRID + 1):
    ax[3].axvline(x - 0.5, color='white', linewidth=0.3, alpha=0.4)
    ax[3].axhline(x - 0.5, color='white', linewidth=0.3, alpha=0.4)
style(ax[3], 'Encoder feature intensity  (14×14)',
      xlabel='Patch column', ylabel='Patch row')
cb3 = fig.colorbar(im3, ax=ax[3], fraction=0.046, pad=0.04)
cb3.set_label('L2 norm', color=LABEL_COLOR, fontsize=7)
cb3.ax.yaxis.set_tick_params(color=LABEL_COLOR, labelsize=7)
cb3.outline.set_edgecolor('#444444')
plt.setp(cb3.ax.yaxis.get_ticklabels(), color=LABEL_COLOR)


#5. cluster map
cmap_three = ListedColormap(['#2ecc71', '#e74c3c', '#1a1a1a'])
ax[4].imshow(pixel_cluster_map, cmap=cmap_three, interpolation='nearest', vmin=0, vmax=2)
style(ax[4], 'Crop zone map  (pixel-level K-Means)')
ax[4].set_xticks([]); ax[4].set_yticks([])

# zoom into field bounding box with padding
field_rows, field_cols = np.where(mask_224 == 1)
pad = 20
ax[4].set_xlim(field_cols.min() - pad, field_cols.max() + pad)
ax[4].set_ylim(field_rows.max() + pad, field_rows.min() - pad)   # inverted y for image coords

ax[4].legend(
    handles=[Patch(facecolor='#2ecc71', label=crop_names[0]),
             Patch(facecolor='#e74c3c', label=crop_names[1]),
             Patch(facecolor='#1a1a1a', label='Outside field')],
    loc='lower right', fontsize=7,
    facecolor='#1a1a1a', edgecolor='#444444', labelcolor=LABEL_COLOR
)

#6. cluster vs ndvi
scatter_colors = ['#2ecc71' if l == 0 else '#e74c3c' for l in pixel_labels]
jitter = np.random.default_rng(42).uniform(-0.08, 0.08, size=len(pixel_labels))
ax[5].scatter(pixel_labels + jitter, field_ndvi,
              c=scatter_colors, alpha=0.7, s=18, edgecolors='none')
ax[5].set_xticks([0, 1])
ax[5].set_xticklabels(['Cluster 0', 'Cluster 1'], color=LABEL_COLOR, fontsize=8)
ax[5].set_xlim(-0.5, 1.5); ax[5].set_ylim(0, 1)
for i, color in enumerate(['#2ecc71', '#e74c3c']):
    mean_val = cluster_ndvi_avg[i]
    ax[5].axhline(mean_val, xmin=i*0.5+0.05, xmax=i*0.5+0.45,
                  color=color, linewidth=2, linestyle='--')
    ax[5].text(i + 0.05, mean_val + 0.02, f'mean={mean_val:.2f}',
               color=color, fontsize=7)
style(ax[5], 'Do clusters match NDVI?',
      xlabel='Cluster', ylabel='Mean patch NDVI')

#6. summary 
ax[6].set_facecolor('#111111')
ax[6].set_xlim(0, 1); ax[6].set_ylim(0, 1)
ax[6].set_xticks([]); ax[6].set_yticks([])
for spine in ax[6].spines.values():
    spine.set_edgecolor('#333333')
left = 0.05
for i in range(N_CLUSTERS):
    width = (cluster_pct[i] / 100) * 0.9
    ax[6].barh(0.65, width, left=left, height=0.28,
               color=['#2ecc71', '#e74c3c'][i], alpha=0.85)
    ax[6].text(left + width/2, 0.65, f'{cluster_pct[i]:.0f}%',
               ha='center', va='center', color='white',
               fontsize=9, fontweight='bold')
    left += width
ndvi_diff = abs(cluster_ndvi_avg[0] - cluster_ndvi_avg[1])
verdict = (
    'Strong spectral separation : likely two distinct crop types.'
    if ndvi_diff > 0.08 else
    'Weak spectral separation : could be stress zones rather than different crops.'
)
ax[6].text(0.5, 0.22,
    f'{crop_names[0]}: {cluster_counts[0]} pixels ({cluster_pct[0]:.0f}%), '
    f'mean NDVI = {cluster_ndvi_avg[0]:.3f}    |    '
    f'{crop_names[1]}: {cluster_counts[1]} pixels ({cluster_pct[1]:.0f}%), '
    f'mean NDVI = {cluster_ndvi_avg[1]:.3f}    |    '
    f'NDVI gap = {ndvi_diff:.3f}    |    {verdict}',
    ha='center', va='center', color=LABEL_COLOR, fontsize=8,
    transform=ax[6].transAxes
)
ax[6].set_title('Field summary', color=TITLE_COLOR, fontsize=10, pad=4)

dashboard_path = os.path.join(OUTPUT_PATH, f'prithvi_dashboard_FID{FIELD_ID}_{DATE}.png')
plt.savefig(dashboard_path, dpi=150, bbox_inches='tight', facecolor='#0f0f0f')
plt.show()
print(f"Dashboard saved: {dashboard_path}")

#geottiff output
print("Saving GeoTIFF...")

with open(META_PATH) as f:
    meta = json.load(f)

gt = meta['source_gt']   
x1 = meta['x1_px']       
y1 = meta['y1_px']       
px_w =  gt[1]              
px_h = -gt[5]              

#top-left corner of the chip in geospatial coordinates
chip_origin_x = gt[0] + x1 * px_w
chip_origin_y = gt[3] + y1 * gt[5]   #gt[5] is -, adding to go downwards

#pixel resolution (10 m)
geo_transform = from_origin(chip_origin_x, chip_origin_y, px_w, px_h)

tif_path = os.path.join(OUTPUT_PATH, f'cluster_map_FID{FIELD_ID}_{DATE}.tif')

with rasterio.open(
    tif_path, 'w',
    driver = 'GTiff',
    height  = CHIP_SIZE,
    width = CHIP_SIZE,
    count = 1,
    dtype = 'uint8',
    crs = CRS.from_wkt(meta['crs_wkt']),
    transform = geo_transform,
) as dst:
    dst.write(pixel_cluster_map.astype('uint8'), 1)

print(f"GeoTIFF saved: {tif_path}")