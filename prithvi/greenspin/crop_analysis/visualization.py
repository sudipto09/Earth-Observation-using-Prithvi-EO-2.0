"""
visualization.py

Builds a dashboard visualising the Prithvi crop-analysis pipeline.

"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
from clustering import ClusterResult
from config import FIELD_ID, DATE, PATCH_GRID, PCA_COMPONENT, CHIP_SIZE
from encoder import make_patch_mask
from spectral import norm




CLUSTER_PALETTE = [
    '#27ae60',  
    '#e74c3c',   
    '#3498db',   
    '#f1c40f',   
    '#9b59b6',   
    '#1abc9c',   
    '#e67e22',    
    '#ec407a',   
]

BG_DARK = '#0d0d0d'
BG_PANEL  = '#141414'
BG_SUMMARY= '#0d0d0d'
LABEL_COL= '#e0e0e0'
TITLE_COL= '#ffffff'
GRID_COL = '#252525'
BORDER_COL= '#2a2a2a'
GOLD   = '#f39c12'
BOUNDARY = '#FFD700'


def _zone_color(idx: int) -> str:
    return CLUSTER_PALETTE[idx % len(CLUSTER_PALETTE)]


def _draw_field_boundary(ax, mask_224: np.ndarray,
                         color: str = BOUNDARY, linewidth: float = 1.6):
    ax.contour(mask_224, levels=[0.5], colors=[color],
               linewidths=linewidth, linestyles='solid')


def _style(ax, title: str, xlabel: str = '', ylabel: str = '', fontsize: int = 9):
    ax.set_title(title, color=TITLE_COL, fontsize=fontsize, pad=5, fontweight='semibold')
    ax.set_facecolor(BG_PANEL)
    ax.tick_params(colors=LABEL_COL, labelsize=7.5, width=1.0, length=3)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER_COL)
    if xlabel:
        ax.set_xlabel(xlabel, color=LABEL_COL, fontsize=7.5)
    if ylabel:
        ax.set_ylabel(ylabel, color=LABEL_COL, fontsize=7.5)


def _colorbar(fig, im, ax, label: str = ''):
    cb = fig.colorbar(im, ax=ax, fraction=0.042, pad=0.03)
    cb.ax.yaxis.set_tick_params(color=LABEL_COL, labelsize=7)
    cb.outline.set_edgecolor(BORDER_COL)
    for lbl in cb.ax.get_yticklabels():
        lbl.set_color(LABEL_COL)
    if label:
        cb.set_label(label, color=LABEL_COL, fontsize=7)
    return cb


# row 0: RGB, NIR false, NDVI

def _panel_true_colour(ax, rgb, mask_224):
    ax.imshow(rgb)
    ax.set_xticks([]); ax.set_yticks([])
    _style(ax, 'True colour  (R·G·B = B4·B3·B2)')
    _draw_field_boundary(ax, mask_224)


def _panel_nir_false(ax, nir_false, mask_224):
    ax.imshow(nir_false)
    ax.set_xticks([]); ax.set_yticks([])
    _style(ax, 'NIR false colour  (R·G·B = B8·B4·B3)')
    _draw_field_boundary(ax, mask_224)


def _panel_ndvi(fig, ax, ndvi_display, mask_224):
    im = ax.imshow(ndvi_display, cmap='YlGn', vmin=0, vmax=1)
    ax.set_xticks([]); ax.set_yticks([])
    _style(ax, 'NDVI  (clipped 0 – 1)')
    _colorbar(fig, im, ax, label='NDVI')
    _draw_field_boundary(ax, mask_224)


# row 1: feature map, BIC curve, PCA scatter

def _panel_feature_map(fig, ax, feature_map, mask_clean):
    patch_mask = make_patch_mask(mask_clean)
    valid_rows, valid_cols = np.where(patch_mask)

    fm = np.where(patch_mask, feature_map, np.nan).copy()
    if not np.all(np.isnan(fm)):
        lo, hi = np.nanmin(fm), np.nanmax(fm)
        fm = (fm - lo) / (hi - lo + 1e-9)

    masked = np.ma.masked_invalid(fm)
    im = ax.imshow(masked, cmap='magma', interpolation='nearest', vmin=0, vmax=1)
    im.cmap.set_bad(color=BG_PANEL)

    if valid_rows.size > 0:
        pad = 1
        r0 = max(valid_rows.min() - pad, 0); r1 = min(valid_rows.max() + pad, PATCH_GRID - 1)
        c0 = max(valid_cols.min() - pad, 0); c1 = min(valid_cols.max() + pad, PATCH_GRID - 1)
        ax.set_xlim(c0 - 0.5, c1 + 0.5)
        ax.set_ylim(r1 + 0.5, r0 - 0.5)
        for x in range(c0, c1 + 2):
            ax.axvline(x - 0.5, color='white', linewidth=0.25, alpha=0.35)
        for y in range(r0, r1 + 2):
            ax.axhline(y - 0.5, color='white', linewidth=0.25, alpha=0.35)

    for r, c in zip(valid_rows, valid_cols):
        ax.add_patch(plt.Rectangle(
            (c - 0.5, r - 0.5), 1, 1,
            linewidth=1.2, edgecolor=BOUNDARY, facecolor='none',
        ))

    # field patch count 
    ax.text(0.02, 0.98, f'{patch_mask.sum()} / 196 field patches',
            transform=ax.transAxes, color=BOUNDARY,
            fontsize=7, va='top', fontstyle='italic')

    # inset context map
    axins = ax.inset_axes([0.68, 0.01, 0.30, 0.30])
    ctx = np.zeros((PATCH_GRID, PATCH_GRID))
    ctx[patch_mask] = 1.0
    axins.imshow(ctx, cmap='YlGn', vmin=0, vmax=1.2, interpolation='nearest')
    axins.set_xticks([]); axins.set_yticks([])
    axins.set_title('Field in grid', color=LABEL_COL, fontsize=5, pad=2)
    for sp in axins.spines.values():
        sp.set_edgecolor(BOUNDARY)
    axins.set_facecolor(BG_PANEL)

    cb = _colorbar(fig, im, ax, label='L2 norm (norm.)')
    _style(ax, 'Encoder feature intensity  (14 × 14 patch grid)',
           xlabel='Patch col', ylabel='Patch row')


def _panel_bic_curve(ax, result: ClusterResult):
    """BIC vs cluster count"""
    if not result.bic_n_range:
        ax.text(0.5, 0.5,
                'Insufficient field\npixels for BIC',
                ha='center', va='center', color=LABEL_COL,
                fontsize=9, transform=ax.transAxes)
        _style(ax, 'BIC — cluster selection')
        return

    n_range = result.bic_n_range
    bic   = result.bic_scores
    opt_n  = result.optimal_n
    opt_idx = n_range.index(opt_n)
    span  = (max(bic) - min(bic)) or 1.0

    # main BIC line
    ax.plot(n_range, bic, color='#5dade2', linewidth=2.0,
            marker='o', markersize=4.5,
            markerfacecolor='#5dade2', markeredgecolor='none', zorder=3)

    # optimal point 
    ax.scatter([opt_n], [bic[opt_idx]], s=110, color=GOLD,
               zorder=5, edgecolors='white', linewidths=0.7)
    ax.axvline(opt_n, color=GOLD, linewidth=1.0, linestyle='--', alpha=0.55)

    # annotation
    offset_x = 0.35 if opt_idx < len(n_range) // 2 else -0.35
    ax.annotate(
        f'Optimal N = {opt_n}',
        xy=(opt_n, bic[opt_idx]),
        xytext=(opt_n + offset_x, bic[opt_idx] + span * 0.18),
        color=GOLD, fontsize=7.5,
        arrowprops=dict(arrowstyle='->', color=GOLD, lw=0.9),
        ha='left' if offset_x > 0 else 'right',
    )

    ax.set_xticks(n_range)
    ax.grid(True, color=GRID_COL, linestyle='--', linewidth=0.6, alpha=0.9)
    _style(ax, 'BIC : Adaptive cluster count',
           xlabel='Number of clusters  N', ylabel='BIC  (lower = better fit)')


def _panel_pca_scatter(ax, result: ClusterResult):
    """PCA feature space coloured by N zones."""
    colours = [_zone_color(l) for l in result.pixel_labels]
    two_d   = result.field_pca.shape[1] >= 2

    if two_d:
        ax.scatter(result.field_pca[:, 0], result.field_pca[:, 1],
                   c=colours, alpha=0.55, s=12, edgecolors='none', rasterized=True)
        title, xlabel, ylabel = 'PCA feature space  (PC1 vs PC2)', 'PC 1', 'PC 2'
    else:
        ax.scatter(result.field_pca[:, 0],
                   np.zeros(len(result.pixel_labels)) +
                   np.random.default_rng(0).uniform(-0.3, 0.3, len(result.pixel_labels)),
                   c=colours, alpha=0.55, s=12, edgecolors='none', rasterized=True)
        title, xlabel, ylabel = 'PCA feature space'

    handles = [Patch(facecolor=_zone_color(i), label=result.crop_names[i], edgecolor='none')
               for i in range(result.optimal_n)]
    ax.legend(handles=handles, fontsize=6.5, facecolor=BG_PANEL,
              edgecolor=BORDER_COL, labelcolor=LABEL_COL,
              loc='best', framealpha=0.85, handlelength=1.2)
    ax.grid(True, color=GRID_COL, linestyle='--', linewidth=0.5, alpha=0.8)
    _style(ax, title, xlabel=xlabel, ylabel=ylabel)


# row 2: crop map, confidence, NDVI stats, summary

def _panel_crop_map(ax, result: ClusterResult, mask_224):
    """Spatial crop-zone map for N zones"""
    n     = result.optimal_n
    clrs  = [_zone_color(i) for i in range(n)] + [BG_PANEL]
    cmap  = ListedColormap(clrs)
    bnorm = BoundaryNorm(np.arange(-0.5, n + 1.5, 1), len(clrs))

    ax.imshow(result.pixel_cluster_map, cmap=cmap, norm=bnorm, interpolation='nearest')
    _style(ax, f'Crop zone map  :  {n} zones  (GMM | Prithvi | spectral)')
    ax.set_xticks([]); ax.set_yticks([])

    rows, cols = np.where(mask_224 == 1)
    if rows.size > 0:
        pad = 22
        ax.set_xlim(cols.min() - pad, cols.max() + pad)
        ax.set_ylim(rows.max() + pad, rows.min() - pad)

    handles = [Patch(facecolor=_zone_color(i), label=result.crop_names[i], edgecolor='none')
               for i in range(n)]
    handles.append(Patch(facecolor=BG_PANEL, label='Outside field',
                         edgecolor=BORDER_COL, linewidth=0.5))
    ax.legend(
    handles=handles,
    loc='center left',
    bbox_to_anchor=(1.02, 0.5),   # push outside right
    fontsize=6.5,
    facecolor=BG_PANEL,
    edgecolor=BORDER_COL,
    labelcolor=LABEL_COL,
    framealpha=0.90,
    borderaxespad=0.0
)
    _draw_field_boundary(ax, mask_224)


def _panel_confidence(fig, ax, result: ClusterResult, mask_224, mask_clean):
    conf_display = np.where(mask_clean == 1, result.confidence_map, np.nan)
    im = ax.imshow(conf_display, cmap='RdYlGn', vmin=0.5, vmax=1.0, interpolation='nearest')
    _style(ax, 'GMM assignment confidence')
    ax.set_xticks([]); ax.set_yticks([])

    rows, cols= np.where(mask_224 == 1)
    if rows.size > 0:
        pad = 22
        ax.set_xlim(cols.min() - pad, cols.max() + pad)
        ax.set_ylim(rows.max() + pad, rows.min() - pad)
    _colorbar(fig, im, ax, label='Confidence (0.5 – 1.0)')


def _panel_ndvi_clusters(ax, result: ClusterResult):
    n     = result.optimal_n
    order = list(np.argsort(result.cluster_ndvi_avg)[::-1])   # high NDVI first

    for rank, cidx in enumerate(order):
        color  = _zone_color(cidx)
        mean_v = result.cluster_ndvi_avg[cidx]
        std_v  = result.cluster_ndvi_std[cidx]
        y_pos  = n - 1 - rank                      # invert 

        
        ax.barh(y_pos, 1.0, height=0.58, left=0,
                color='#1e1e1e', alpha=1.0, zorder=1)
        
        ax.barh(y_pos, mean_v, height=0.58, left=0,
                color=color, alpha=0.88, zorder=2)
        
        # ax.errorbar(mean_v, y_pos, xerr=std_v, fmt='none',
        #             color='white', capsize=3.5, linewidth=1.4, zorder=3)
        
        label_x = min(mean_v + std_v + 0.03, 0.97)
        ax.text(label_x, y_pos, f'{mean_v:.3f}',
                va='center', color='white', fontsize=7.5, zorder=4)
        
        ax.text(0.012, y_pos, result.crop_names[cidx],
                va='center', color='white', fontsize=7.0,
                fontweight='bold', zorder=4,
                clip_on=True)

    ax.set_yticks([])
    ax.set_xlim(0, 1.08)
    ax.set_ylim(-0.55, n - 0.45)
    ax.grid(True, axis='x', color=GRID_COL, linestyle='--',
            linewidth=0.6, alpha=0.9)
    _style(ax, 'Mean NDVI per zone',
           xlabel='NDVI', ylabel='')


#row 3: pipeline  + summary

def _panel_summary(ax, result: ClusterResult):
    n = result.optimal_n

    ax.set_facecolor(BG_SUMMARY)
    ax.set_xlim(0, 1);  ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor('#1e1e1e')
    ax.set_title('Summary', color=TITLE_COL, fontsize=10,
                 pad=5, fontweight='bold')

    # proportional cluster size bars
    BAR_TOP = 0.92;  BAR_H = 0.10;  left = 0.02
    for i in range(n):
        color = _zone_color(i)
        width = (result.cluster_pct[i] / 100) * 0.96
        ax.barh(BAR_TOP, width, left=left, height=BAR_H, color=color, alpha=0.92)
        if result.cluster_pct[i] > 4:
            ax.text(left + width / 2, BAR_TOP, f'{result.cluster_pct[i]:.0f}%',
                    ha='center', va='center', color='white',
                    fontsize=8.5, fontweight='bold')
        left += width

    # detailed cluster info
    sorted_idx = list(np.argsort(result.cluster_ndvi_avg)[::-1])
    row_gap    = min(0.125, 0.52 / max(n, 1))
    row_top    = BAR_TOP - BAR_H / 2 - 0.08

    for rank, cidx in enumerate(sorted_idx):
        color = _zone_color(cidx)
        y     = row_top - rank * row_gap
        ax.text(0.02, y,
            f'{result.crop_names[cidx]}:  '
            f'{result.cluster_counts[cidx]:,} px '
            f'({result.cluster_pct[cidx]:.1f} %)   '
            f'NDVI {result.cluster_ndvi_avg[cidx]:.3f} ± {result.cluster_ndvi_std[cidx]:.3f}   '
            f'Conf {result.cluster_conf_avg[cidx]:.2f}',
            color=color, fontsize=7.8, va='center',
        )

    # separator line
    sep_y = row_top - n * row_gap - 0.01
    ax.axhline(sep_y, xmin=0.02, xmax=0.98,
               color='#2a2a2a', linewidth=0.8)

    # meta info
    meta_y = sep_y - 0.055
    ax.text(0.5, meta_y,
        f'Optimal clusters (BIC): {result.optimal_n}   |  '
        f'NDVI spread: {result.ndvi_diff:.3f}   |   '
        f'Avg confidence: {result.avg_confidence:.2f}',
        ha='center', va='center', color='#888888', fontsize=7.5,
    )

    # final verdict
    verdict_y = meta_y - 0.09
    ax.text(0.5, verdict_y, result.verdict,
        ha='center', va='center', color='#d5d5d5',
        fontsize=8.0, fontstyle='italic',
    )


def _panel_pipeline(ax, chip, chip_enriched, mask_clean, emb_pixels, result, device):
    """Pipeline panel."""
    ax.set_facecolor(BG_SUMMARY)
    ax.set_xlim(0, 1);  ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor('#1e1e1e')
    ax.set_title('Pipeline', color=TITLE_COL, fontsize=10, pad=5, fontweight='bold')

    from config import chip_path
    bic_str = (f'[{min(result.bic_scores):.0f} – {max(result.bic_scores):.0f}]'
               if result.bic_scores else 'N/A')

    lines = [
        f"Device: {device.type.upper()}",
        f"Raw chip: {chip.shape}  to  enriched: {chip_enriched.shape}",
        f"Valid field pixels: {int(mask_clean.sum())}  (post cloud/shadow filter)",
        f"Embeddings: {emb_pixels.shape}  (upsampled Prithvi patches)",
        f"PCA components: {result.field_pca.shape[1]}",
        f"BIC scores: {bic_str}",
        f"Optimal N (BIC): {result.optimal_n}",
        f"File: {chip_path().split('/')[-1].split(chr(92))[-1]}",
    ]
    for i, line in enumerate(lines):
        ax.text(0.04, 0.87 - i * 0.105, line,
                color=LABEL_COL, fontsize=7.5, va='center',
                fontfamily='monospace')


# main visualization function

def build_dashboard(
    rgb, nir_false, ndvi_display, ndvi,
    feature_map,
    chip, chip_enriched,
    mask_224,
    mask_clean,
    emb_pixels,
    result: ClusterResult,
    device,
    save_path: str,
) -> None:
    """Build the entire dashboard with all panels and save it as a PNG file."""

    fig = plt.figure(figsize=(20, 16), facecolor=BG_DARK)
    fig.suptitle(
        f'Prithvi-EO 2.0  ·  Field {FIELD_ID}  ·  {DATE}'
        f'  ·  GMM + PCA + BIC  ({result.optimal_n} zones)',
        fontsize=13, color='white', fontweight='bold', y=0.975,
        fontfamily='sans-serif',
    )

    gs = gridspec.GridSpec(
        4, 3, figure=fig,
        hspace=0.44, wspace=0.26,
        height_ratios=[1, 1, 1, 0.44],
    )

    p = [
        fig.add_subplot(gs[0, 0]),   # 0  True colour
        fig.add_subplot(gs[0, 1]),   # 1  NIR false colour
        fig.add_subplot(gs[0, 2]),   # 2  NDVI
        fig.add_subplot(gs[1, 0]),   # 3  Feature map
        fig.add_subplot(gs[1, 1]),   # 4  BIC curve
        fig.add_subplot(gs[1, 2]),   # 5  PCA scatter
        fig.add_subplot(gs[2, 0]),   # 6  Crop zone map
        fig.add_subplot(gs[2, 1]),   # 7  Confidence
        fig.add_subplot(gs[2, 2]),   # 8  NDVI per zone
        fig.add_subplot(gs[3, :2]),  # 9  Summary
        fig.add_subplot(gs[3, 2]),   # 10 Pipeline
    ]

    _panel_true_colour (p[0],  rgb,    mask_224)
    _panel_nir_false  (p[1],  nir_false,   mask_224)
    _panel_ndvi  (fig, p[2],  ndvi_display, mask_224)
    _panel_feature_map (fig, p[3],  feature_map,  mask_clean)
    _panel_bic_curve  (p[4], result)
    _panel_pca_scatter (p[5], result)
    _panel_crop_map   (p[6],  result,      mask_224)
    _panel_confidence  (fig, p[7],  result, mask_224, mask_clean)
    _panel_ndvi_clusters(p[8],  result)
    _panel_summary (p[9],  result)
    _panel_pipeline(p[10], chip, chip_enriched, mask_clean, emb_pixels, result, device)

    plt.savefig(save_path, dpi=160, bbox_inches='tight',
                facecolor=BG_DARK, edgecolor='none')
    plt.close(fig)
    print(f"Dashboard saved to: {save_path}")
    
    