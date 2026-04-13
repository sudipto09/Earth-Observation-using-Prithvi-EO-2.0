"""
visualization.py
----------------
Contains functions to build a comprehensive dashboard visualising the Prithvi input data, encoder feature map, PCA space, GMM clustering results, and summary statistics.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from clustering import ClusterResult
from config import FIELD_ID, DATE, PATCH_GRID, PCA_COMPONENT, CHIP_SIZE
from spectral import norm

# colour scheme
COL_A      = '#2ecc71'
COL_B      = '#e74c3c'
BG_DARK    = '#0f0f0f'
BG_PANEL   = '#1a1a1a'
BG_SUMMARY = '#111111'
LABEL_COL  = '#ffffff'
TITLE_COL  = 'white'


#helper functions for consistent styling across panels

def _draw_field_boundary(ax, mask_224: np.ndarray,
                          color: str = '#FFD700', linewidth: float = 1.8):
    """Overlay the field boundary on any image panel."""
    ax.contour(mask_224, levels=[0.5], colors=[color],
               linewidths=linewidth, linestyles='solid')

def _style(ax, title, xlabel='', ylabel=''):
    ax.set_title(title, color=TITLE_COL, fontsize=10, pad=6)
    ax.set_facecolor(BG_PANEL)
    ax.tick_params(
    colors=LABEL_COL,
    labelsize=8,        # slightly bigger
    width=1.2,          # thicker ticks
)
    for spine in ax.spines.values():
        spine.set_edgecolor('#888888')
    if xlabel:
        ax.set_xlabel(xlabel, color=LABEL_COL, fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, color=LABEL_COL, fontsize=8)


def _colorbar(fig, im, ax):
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.yaxis.set_tick_params(color=LABEL_COL, labelsize=7)
    cb.outline.set_edgecolor('#444444')
    cb.ax.yaxis.set_tick_params(color='white', labelsize=8)
    for label in cb.ax.get_yticklabels():
        label.set_color('white')
    return cb


#panel functions for each subplot

def _panel_true_colour(ax, rgb, mask_224):         
    ax.imshow(rgb)
    ax.set_xticks([]); ax.set_yticks([])
    _style(ax, 'True colour (RGB)')
    ax.text(4, 218, 'Bands 3-2-1', color=LABEL_COL, fontsize=7, va='bottom')
    _draw_field_boundary(ax, mask_224)            

def _panel_nir_false(ax, nir_false, mask_224):      
    ax.imshow(nir_false)
    ax.set_xticks([]); ax.set_yticks([])
    _style(ax, 'False colour NIR (crops glow red)')
    ax.text(4, 218, 'Bands 4-3-2', color=LABEL_COL, fontsize=7, va='bottom')
    _draw_field_boundary(ax, mask_224)              


def _panel_ndvi(fig, ax, ndvi_display, mask_224):  
    im = ax.imshow(ndvi_display, cmap='YlGn', vmin=0, vmax=1)
    ax.set_xticks([]); ax.set_yticks([])
    _style(ax, 'NDVI (greenness per pixel)')
    _colorbar(fig, im, ax)
    _draw_field_boundary(ax, mask_224)            


def _panel_feature_map(fig, ax, feature_map):
    im = ax.imshow(feature_map, cmap='viridis', interpolation='nearest')
    im.cmap.set_bad(color='#1a1a1a')
    for x in range(PATCH_GRID + 1):
        ax.axvline(x - 0.5, color='white', linewidth=0.3, alpha=0.4)
        ax.axhline(x - 0.5, color='white', linewidth=0.3, alpha=0.4)

    #zoom to field patches
    valid_rows, valid_cols = np.where(~np.isnan(feature_map))
    if valid_rows.size > 0:
        pad = 2
        ax.set_xlim(valid_cols.min() - pad - 0.5, valid_cols.max() + pad + 0.5)
        ax.set_ylim(valid_rows.max() + pad + 0.5, valid_rows.min() - pad - 0.5)
    

    _style(ax, 'Encoder feature intensity (14 x 14)',
           xlabel='Patch column', ylabel='Patch row')
    cb = _colorbar(fig, im, ax)
    cb.set_label('L2 norm', color=LABEL_COL, fontsize=7)


def _panel_pca_variance(ax, pca_model):
    n_components = len(pca_model.explained_variance_ratio_)
    components = np.arange(1, n_components + 1)
    var_ratio  = pca_model.explained_variance_ratio_ * 100
    cumvar = np.cumsum(var_ratio)
    ax.bar(components, var_ratio, color='#3498db', alpha=0.8, label='Per component')
    ax.plot(components, cumvar, color='#e74c3c', marker='o',
            markersize=4, linewidth=1.5, label='Cumulative')
    ax.axhline(80, color='#f39c12', linewidth=1, linestyle='--', alpha=0.6)
    ax.text(PCA_COMPONENT, 81, '80%', color='#f39c12', fontsize=7, ha='right')
    ax.set_xticks(components)
    total = pca_model.explained_variance_ratio_.cumsum()[-1] * 100
    _style(ax, f'PCA variance',
           xlabel='Principal component', ylabel='Variance (%)')
    ax.legend(fontsize=7, facecolor=BG_PANEL, edgecolor='#444444', labelcolor=LABEL_COL)


def _panel_pca_scatter(ax, result: ClusterResult):
    gi = result.greener_idx
    colours = [COL_A if l == gi else COL_B for l in result.pixel_labels]
    ax.scatter(result.field_pca[:, 0], result.field_pca[:, 1],
               c=colours, alpha=0.7, s=20, edgecolors='none')
    _style(ax, 'PCA space (PC1 vs PC2)', xlabel='PC 1', ylabel='PC 2')
    ax.legend(
        handles=[Patch(facecolor=COL_A, label=result.crop_names[gi]),
                 Patch(facecolor=COL_B, label=result.crop_names[1-gi])],
        fontsize=7, facecolor=BG_PANEL, edgecolor='#444444', labelcolor=LABEL_COL,
    )
    ax.grid(True, color='#444444', linestyle='--', linewidth=0.5, alpha=0.5)


def _panel_crop_map(ax, result: ClusterResult, mask_224):
    cmap = ListedColormap([COL_A, COL_B, BG_PANEL])
    ax.imshow(result.pixel_cluster_map, cmap=cmap, interpolation='nearest', vmin=0, vmax=2)
    _style(ax, 'Crop zone map (GMM, Prithvi + spectral)')
    ax.set_xticks([]); ax.set_yticks([])
    rows, cols = np.where(mask_224 == 1)
    pad = 20
    ax.set_xlim(cols.min() - pad, cols.max() + pad)
    ax.set_ylim(rows.max() + pad, rows.min() - pad)
    gi = result.greener_idx
    ax.legend(
        handles=[Patch(facecolor=COL_A, label=result.crop_names[gi]),
                 Patch(facecolor=COL_B, label=result.crop_names[1-gi]),
                 Patch(facecolor=BG_PANEL, label='Outside field')],
        loc='lower right', fontsize=7,
        facecolor=BG_PANEL, edgecolor='#444444', labelcolor=LABEL_COL,
    )


def _panel_confidence(fig, ax, result: ClusterResult, mask_224):
    conf_display = np.where(mask_224 == 1, result.confidence_map, np.nan)
    im = ax.imshow(conf_display, cmap='RdYlGn', vmin=0.5, vmax=1.0, interpolation='nearest')
    _style(ax, 'GMM assignment confidence (0.5=uncertain, 1.0=certain)')
    ax.set_xticks([]); ax.set_yticks([])
    rows, cols = np.where(mask_224 == 1)
    pad = 20
    ax.set_xlim(cols.min() - pad, cols.max() + pad)
    ax.set_ylim(rows.max() + pad, rows.min() - pad)
    _colorbar(fig, im, ax)


def _panel_ndvi_scatter(ax, result: ClusterResult, ndvi, mask_224):
    gi     = result.greener_idx
    field_ndvi  = ndvi.ravel()[mask_224.ravel() == 1]
    colours  = [COL_A if l == gi else COL_B for l in result.pixel_labels]
    jitter  = np.random.default_rng(42).uniform(-0.08, 0.08, size=len(result.pixel_labels))
    ax.scatter(result.pixel_labels + jitter, field_ndvi,
               c=colours, alpha=0.7, s=18, edgecolors='none')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Cluster 0', 'Cluster 1'], color=LABEL_COL, fontsize=8)
    ax.set_xlim(-0.5, 1.5); ax.set_ylim(0, 1)
    for i, color in enumerate([COL_A, COL_B]):
        mean_val = result.cluster_ndvi_avg[i]
        ax.axhline(mean_val, xmin=i*0.5+0.05, xmax=i*0.5+0.45,
                   color=color, linewidth=2, linestyle='--')
        ax.text(i + 0.05, mean_val + 0.02,
                f'mean={mean_val:.2f}  conf={result.cluster_conf_avg[i]:.2f}',
                color=color, fontsize=7)
    _style(ax, 'Do clusters match NDVI?', xlabel='Cluster', ylabel='Pixel NDVI')


def _panel_summary(ax, result: ClusterResult):
    ax.set_facecolor(BG_SUMMARY)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor('#333333')
    ax.set_title('Summary', color=TITLE_COL, fontsize=10, pad=6)

    left = 0.05
    for i, color in enumerate([COL_A, COL_B]):
        width = (result.cluster_pct[i] / 100) * 0.9
        ax.barh(0.70, width, left=left, height=0.20, color=color, alpha=0.85)
        ax.text(left + width/2, 0.70, f'{result.cluster_pct[i]:.0f}%',
                ha='center', va='center', color='white', fontsize=9, fontweight='bold')
        left += width

    gi = result.greener_idx
    ax.text(0.5, 0.38,
        f'{result.crop_names[gi]}: {result.cluster_counts[gi]} px '
        f'({result.cluster_pct[gi]:.0f}%) | Mean NDVI: {result.cluster_ndvi_avg[gi]:.3f}\n'
        f'{result.crop_names[1-gi]}: {result.cluster_counts[1-gi]} px '
        f'({result.cluster_pct[1-gi]:.0f}%) | Mean NDVI: {result.cluster_ndvi_avg[1-gi]:.3f}',
        ha='center', va='center', color=LABEL_COL, fontsize=9,
    )
    ax.text(0.5, 0.12,
        f'NDVI Gap: {result.ndvi_diff:.3f}   |   '
        f'Avg Confidence: {result.avg_confidence:.2f}   |   {result.verdict}',
        ha='center', va='center', color=TITLE_COL, fontsize=9, fontweight='bold',
    )


def _panel_pipeline(ax, chip, chip_enriched, mask_224, emb_pixels, device):
    ax.set_facecolor(BG_SUMMARY)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor('#333333')
    ax.set_title('Pipeline', color=TITLE_COL, fontsize=10, pad=6)

    from config import chip_path
    lines = [
        f"Compute Device: {device.type.upper()}",
        f"Raw Tensor: {chip.shape} -> Enriched: {chip_enriched.shape}",
        f"Mask Target Valid Px: {int(mask_224.sum())}",
        f"Prithvi Embeddings (upsampled): {emb_pixels.shape}",
        f"Combined Feat Matrix: ({CHIP_SIZE*CHIP_SIZE}, {emb_pixels.shape[1]+10}) -> PCA: {PCA_COMPONENT}d",
        f"File Name: {chip_path().split('/')[-1].split(chr(92))[-1]}",
    ]
    for i, line in enumerate(lines):
        ax.text(0.04, 0.82 - i * 0.14, line, color=LABEL_COL, fontsize=8)


#main dashboard function

def build_dashboard(
    rgb, nir_false, ndvi_display, ndvi,
    feature_map,
    chip, chip_enriched,
    mask_224,
    emb_pixels,
    result: ClusterResult,
    device,
    save_path: str,
):
    """Compose all panels and save the dashboard PNG."""
    fig = plt.figure(figsize=(18, 15), facecolor=BG_DARK)
    fig.suptitle(
        f'Prithvi field analysis | Field {FIELD_ID} | {DATE} | GMM + PCA + Prithvi embeddings',
        fontsize=14, color='white', fontweight='bold', y=0.96,
    )

    gs = gridspec.GridSpec(
        4, 3, figure=fig,
        hspace=0.42, wspace=0.28,
        height_ratios=[1, 1, 1, 0.45],
    )

    panels = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[1, 2]),
        fig.add_subplot(gs[2, 0]),
        fig.add_subplot(gs[2, 1]),
        fig.add_subplot(gs[2, 2]),
        fig.add_subplot(gs[3, :2]),
        fig.add_subplot(gs[3, 2]),
    ]

    _panel_true_colour(panels[0], rgb, mask_224)       
    _panel_nir_false(panels[1], nir_false, mask_224)   
    _panel_ndvi(fig, panels[2], ndvi_display, mask_224) 
    _panel_feature_map(fig, panels[3], feature_map)
    _panel_pca_variance(panels[4], result.pca_model)
    _panel_pca_scatter(panels[5], result)
    _panel_crop_map(panels[6], result, mask_224)
    _panel_confidence(fig, panels[7], result, mask_224)
    _panel_ndvi_scatter(panels[8], result, ndvi, mask_224)
    _panel_summary(panels[9], result)
    _panel_pipeline(panels[10], chip, chip_enriched, mask_224, emb_pixels, device)

    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=BG_DARK)
    plt.close(fig)
    print(f"Dashboard saved: {save_path}")