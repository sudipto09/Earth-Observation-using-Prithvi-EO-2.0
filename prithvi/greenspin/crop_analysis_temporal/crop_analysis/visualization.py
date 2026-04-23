"""
visualization.py

Builds a 4-row analysis dashboard for the Prithvi crop-zone pipeline.

"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.colors import BoundaryNorm, ListedColormap
from clustering import ClusterResult
from config import FIELD_ID, DATES, PATCH_GRID
from encoder import make_patch_mask





CLUSTER_PALETTE = [
    '#27ae60',   # zone 0   green  
    '#e74c3c',   # zone 1   red
    '#3498db',   # zone 2   blue
    '#f1c40f',   # zone 3   yellow
    '#9b59b6',   # zone 4   purple
    '#1abc9c',   # zone 5   teal
    '#e67e22',   # zone 6   orange
    '#ec407a',   # zone 7   pink
]

BG_DARK   = '#0d0d0d'
BG_PANEL  = '#141414'
BG_SUMMARY = '#0d0d0d'
LABEL_COL = '#e0e0e0'
TITLE_COL = '#ffffff'
GRID_COL = '#252525'
BORDER_COL = '#2a2a2a'
GOLD    = '#f39c12'
BOUNDARY  = '#FFD700'
CAPTION  = '#888888'




def _zone_color(idx: int) -> str:
    return CLUSTER_PALETTE[idx % len(CLUSTER_PALETTE)]


def _draw_field_boundary(ax, mask_224: np.ndarray,
                         color: str = BOUNDARY, linewidth: float = 1.6):
    
    ax.contour(mask_224, levels=[0.5], colors=[color],
               linewidths=linewidth, linestyles='solid')


def _style(ax, title: str, subtitle: str = '',
           xlabel: str = '', ylabel: str = '', fontsize: int = 9):
    
    
    pad = 14 if subtitle else 4
    ax.set_title(title, color=TITLE_COL, fontsize=fontsize,
                 pad=pad, fontweight='semibold')
    
    if subtitle:
        
        ax.text(0.5, 1.03, subtitle, transform=ax.transAxes,
                ha='center', va='bottom', color=CAPTION, fontsize=6.5,
                fontstyle='italic')
                
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


def _field_zoom(ax, mask_224: np.ndarray, pad: int = 22):
    
    rows, cols = np.where(mask_224 == 1)
    if rows.size > 0:
        ax.set_xlim(cols.min() - pad, cols.max() + pad)
        ax.set_ylim(rows.max() + pad, rows.min() - pad)


#row 0

def _panel_true_colour(ax, rgb, mask_224):
    ax.imshow(rgb)
    ax.set_xticks([]); ax.set_yticks([])
    _style(ax, 'True Colour  (B4 - B3 - B2)')           
    _draw_field_boundary(ax, mask_224)


def _panel_nir_false(ax, nir_false, mask_224):
    ax.imshow(nir_false)
    ax.set_xticks([]); ax.set_yticks([])
    _style(ax, 'NIR False Colour  (B8 - B4 - B3)')
    _draw_field_boundary(ax, mask_224)


def _panel_ndvi(fig, ax, ndvi_display, mask_224):
    im = ax.imshow(ndvi_display, cmap='YlGn', vmin=0, vmax=1)
    ax.set_xticks([]); ax.set_yticks([])
    _style(ax, 'NDVI')
    _colorbar(fig, im, ax, label='NDVI')
    _draw_field_boundary(ax, mask_224)


# row 1

def _panel_feature_map(fig, ax, feature_map, mask_clean):
    
    patch_mask = make_patch_mask(mask_clean)
    valid_rows, valid_cols= np.where(patch_mask)

    fm= np.where(patch_mask, feature_map, np.nan).copy()
    if not np.all(np.isnan(fm)):
        lo, hi = np.nanmin(fm), np.nanmax(fm)
        if hi > lo:
            fm = (fm - lo) / (hi - lo)
        else:
            fm = np.zeros_like(fm)

    masked = np.ma.masked_invalid(fm)
    im = ax.imshow(masked, cmap='magma', interpolation='nearest', vmin=0, vmax=1)
    im.cmap.set_bad(color=BG_PANEL)

    if valid_rows.size > 0:
        pad = 1
        r0 = max(valid_rows.min() - pad, 0); r1 = min(valid_rows.max() + pad, PATCH_GRID - 1)
        c0 = max(valid_cols.min() - pad, 0); c1= min(valid_cols.max() + pad, PATCH_GRID - 1)
        ax.set_xlim(c0 - 0.5, c1 + 0.5)
        ax.set_ylim(r1 + 0.5, r0 - 0.5)
        for x in range(c0, c1 + 2):
            ax.axvline(x - 0.5, color='white',linewidth=0.25, alpha=0.35)
        for y in range(r0, r1 + 2):
            ax.axhline(y - 0.5, color='white',linewidth=0.25, alpha=0.35)

    for r, c in zip(valid_rows, valid_cols):
        ax.add_patch(plt.Rectangle(
            (c - 0.5, r - 0.5), 1, 1,
            linewidth=1.2, edgecolor=BOUNDARY, facecolor='none',
        ))

    n_patches= int(patch_mask.sum())
    ax.text(0.02, 0.98,
            f'{n_patches} / 196 field patches',
            transform=ax.transAxes, color=BOUNDARY,
            fontsize=7, va='top', fontstyle='italic')

    # inset context map of field location
    axins= ax.inset_axes([0.68, 0.01, 0.30, 0.30])
    ctx = np.zeros((PATCH_GRID, PATCH_GRID))
    ctx[patch_mask] = 1.0
    axins.imshow(ctx, cmap='YlGn', vmin=0, vmax=1.2, interpolation='nearest')
    axins.set_xticks([]); axins.set_yticks([])
    axins.set_title('Field in grid', color=LABEL_COL, fontsize=5, pad=2)
    for sp in axins.spines.values():
        sp.set_edgecolor(BOUNDARY)
    axins.set_facecolor(BG_PANEL)

    _colorbar(fig, im, ax, label='Feature intensity')
    _style(ax, 'Encoder Feature Intensity')


def _panel_bic_curve(ax, result: ClusterResult):
    
    if not result.bic_n_range:
        ax.text(0.5, 0.5,
                'Insufficient field\npixels for BIC',
                ha='center', va='center', color=LABEL_COL,
                fontsize=9, transform=ax.transAxes)
        _style(ax, 'BIC-Cluster Count Selection')
        return

    n_range = result.bic_n_range
    bic   = result.bic_scores
    opt_n  = result.optimal_n
    if opt_n in n_range:
        opt_idx= n_range.index(opt_n)
    else:
        opt_idx = int(np.argmin(bic))
        opt_n= n_range[opt_idx]
    bic_span= (max(bic) - min(bic)) or 1.0

    # BIC line
    ax.plot(n_range, bic, color='#5dade2', linewidth=2.0,
            marker='o', markersize=4.5,
            markerfacecolor='#5dade2', markeredgecolor='none', zorder=3)

    
    ax.axvspan(opt_n, max(n_range) + 0.5, alpha=0.06,
               color='#e74c3c', label='Over-fitting risk')

    # optimal point 
    ax.scatter([opt_n], [bic[opt_idx]], s=120, color=GOLD,
               zorder=5, edgecolors='white', linewidths=0.8)
    ax.axvline(opt_n, color=GOLD, linewidth=1.0, linestyle='--', alpha=0.55)

    # annotation 
    
    is_rising = bic[0] <= bic[-1]
    offset_x = 0.35 if opt_idx < len(n_range) // 2 else -0.35
    offset_y  = -bic_span * 0.14 if is_rising and opt_idx == 0 else bic_span * 0.18
    annot_label = f'Best N = {opt_n}'
    if opt_n == 1:
        annot_label += '  (single zone)'
    ax.annotate(
        annot_label,
        xy  = (opt_n, bic[opt_idx]),
        xytext  = (opt_n + offset_x, bic[opt_idx] + offset_y),
        color = GOLD, fontsize=7.5, fontweight='semibold',
        arrowprops = dict(arrowstyle='->', color=GOLD, lw=0.9),
        ha = 'left' if offset_x > 0 else 'right',
    )

    ax.set_xticks(n_range)
    ax.grid(True, color=GRID_COL, linestyle='--', linewidth=0.6, alpha=0.9)
    _style(ax, 'BIC - Automatic Cluster Count Selection')


def _panel_pca_scatter(ax, result: ClusterResult):
    
    colours = [_zone_color(l) for l in result.pixel_labels]
    two_d = result.field_pca.shape[1] >= 2

    if two_d:
        ax.scatter(result.field_pca[:, 0], result.field_pca[:, 1],
                   c=colours, alpha=0.50, s=10, edgecolors='none', rasterized=True)
        xlabel, ylabel = 'PC 1', 'PC 2'
        title_str = 'Feature Space - PCA'
    else:
        rng= np.random.default_rng(0)
        jitter =rng.uniform(-0.3, 0.3, len(result.pixel_labels))
        ax.scatter(result.field_pca[:, 0], jitter,
                   c=colours, alpha=0.50, s=10, edgecolors='none', rasterized=True)
        xlabel, ylabel = 'PC 1', 'Jitter'
        title_str = 'Feature Space  (PC1)'

    unique_labels = np.unique(result.pixel_labels)

    handles = [
        Patch(facecolor=_zone_color(i),
              label=result.crop_names[i][:28] if len(result.crop_names[i]) > 28
              else result.crop_names[i])
        for i in unique_labels
    ]
    
    ax.legend(handles=handles, fontsize=6.5, facecolor=BG_PANEL,
              edgecolor=BORDER_COL, labelcolor=LABEL_COL,
              loc='center left',bbox_to_anchor= (1.02, 0.5), framealpha=0.85, handlelength=1.2)
    
    
    
    ax.grid(True, color=GRID_COL, linestyle='--', linewidth=0.5, alpha=0.8)
    _style(ax, title_str,
           subtitle='Each dot = one field pixel | separated chunks = distinct crop zones',
           xlabel=xlabel, ylabel=ylabel)


#row 2 

def _panel_crop_map(ax, result: ClusterResult, mask_224):

    n  = result.optimal_n

    
    clrs = [BG_PANEL] + [_zone_color(i) for i in range(n)]
    cmap = ListedColormap(clrs)

    bnorm = BoundaryNorm(np.arange(-1.5, n + 0.5, 1), len(clrs))

    ax.imshow(result.pixel_cluster_map, cmap=cmap, norm=bnorm, interpolation='nearest')
    ax.set_xticks([]); ax.set_yticks([])

    _field_zoom(ax, mask_224)

    handles= [Patch(facecolor=_zone_color(i), label=result.crop_names[i], edgecolor='none')
               for i in range(n)]
    handles.append(Patch(facecolor=BG_PANEL, label='Outside field',
                         edgecolor=BORDER_COL, linewidth=0.5))
    ax.legend(
        handles  = handles,
        loc  = 'center left',
        bbox_to_anchor= (1.02, 0.5),
        fontsize = 6.5,
        facecolor  = BG_PANEL,
        edgecolor= BORDER_COL,
        labelcolor = LABEL_COL,
        framealpha = 0.90,
        borderaxespad = 0.0,
    )
    _draw_field_boundary(ax, mask_224)
    _style(ax, 'Crop Zone Map')


def _panel_confidence(fig, ax, result: ClusterResult, mask_224, mask_clean):

    conf_display = np.where(mask_clean == 1, result.confidence_map, np.nan)
    im = ax.imshow(conf_display, cmap='RdYlGn', vmin=0.5, vmax=1.0,
                   interpolation='nearest')
    ax.set_xticks([]); ax.set_yticks([])
    
    zoom_mask = mask_clean if mask_clean.sum() > 0 else mask_224
    _field_zoom(ax, zoom_mask)

    
    
    CONF_THRESHOLD = 0.7
    low_conf = (result.confidence_map < CONF_THRESHOLD) & (mask_clean == 1)
    if low_conf.any():
        rows, cols = np.where(low_conf)
        ax.scatter(cols, rows, s=1.5, c='white', alpha=0.35,
                   linewidths=0, rasterized=True)

    _draw_field_boundary(ax, mask_224)
    _colorbar(fig, im, ax, label='Confidence')
    _style(ax, 'GMM Assignment Confidence')

def _panel_ndvi_clusters(ax, result: ClusterResult):
    
    n = result.optimal_n
    order = list(np.argsort(result.cluster_ndvi_avg)[::-1])

    if n == 0:
        ax.set_ylim(-0.55, 0.45)
        _style(ax, 'Mean NDVI per Zone')
        return

    for rank, cidx in enumerate(order):
        color = _zone_color(cidx)
        mean_v = float(result.cluster_ndvi_avg[cidx])
        std_v = float(result.cluster_ndvi_std[cidx])
        pct = float(result.cluster_pct[cidx])
        y_pos = n - 1 - rank

        
        ax.barh(y_pos, 1.0, height=0.68, left=0,
                color='#1e1e1e', alpha=1.0, zorder=1)

        
        bar_w = float(np.clip(mean_v, 0.0, 1.0))
        ax.barh(y_pos, bar_w, height=0.68, left=0,
                color=color, alpha=0.88, zorder=2)

       
        err_lo = max(bar_w - std_v, 0.0)
        err_hi = min(bar_w + std_v, 1.0)
        # if err_hi > err_lo:
        #     ax.plot([err_lo, err_hi], [y_pos, y_pos],
        #             color='white', linewidth=1.1, alpha=0.75, zorder=3)
        #     for xe in (err_lo, err_hi):
        #         ax.plot([xe, xe], [y_pos - 0.12, y_pos + 0.12],
        #                 color='white', linewidth=1.0, alpha=0.75, zorder=3)

        
        name_trunc = result.crop_names[cidx]
        if len(name_trunc) > 26:
            name_trunc = name_trunc[:24] + '..'
        label_text = f'{name_trunc}  ({pct:.0f}%)'
        if bar_w > 0.35:
            ax.text(0.015, y_pos, label_text,
                    va='center', ha='left', color='white', fontsize=6.5,
                    fontweight='bold', zorder=4, clip_on=True)
        else:
            ax.text(bar_w + 0.015, y_pos, label_text,
                    va='center', ha='left', color=LABEL_COL, fontsize=6.5,
                    fontweight='bold', zorder=4, clip_on=True)

        
        ax.text(1.06, y_pos, f'{mean_v:.3f} ± {std_v:.3f}',
                va='center', ha='right', color=LABEL_COL, fontsize=7.0,
                zorder=4, clip_on=False)

    for xref, lbl in [(0.2, '0.2'), (0.5, '0.5'), (0.8, '0.8')]:
        ax.axvline(xref, color='#555', linewidth=0.7, linestyle=':', zorder=0)

    ax.set_yticks([])
    ax.set_xlim(0, 1.08)
    ax.set_ylim(-0.6, n - 0.4)

    ax.set_xlabel('Mean NDVI', color=LABEL_COL, fontsize=7.5)
    ax.grid(True, axis='x', color=GRID_COL, linestyle='--',
            linewidth=0.6, alpha=0.9)
    _style(ax, 'Mean NDVI per Zone')



#row 3

def _panel_summary(ax, result: ClusterResult):
    
    n = max(result.optimal_n, 1)
    ax.set_facecolor(BG_SUMMARY)
    ax.set_xlim(0, 1);  ax.set_ylim(0, 1)
    ax.set_xticks([]);  ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor('#1e1e1e')
    ax.set_title('Field Summary', color=TITLE_COL, fontsize=10,
                 pad=5, fontweight='bold')

    #area coverage bar
    BAR_TOP = 0.88       
    BAR_H   = 0.09
    BAR_LEFT_EDGE  = 0.08    
    BAR_RIGHT_EDGE = 0.98
    BAR_FULL_W = BAR_RIGHT_EDGE - BAR_LEFT_EDGE

    ax.text(0.01, BAR_TOP, 'Area\ncoverage',color=CAPTION, fontsize=6.0, va='center', ha='left',fontfamily='monospace')

    left = BAR_LEFT_EDGE
    for i in range(n):
        color = _zone_color(i)
        width = (result.cluster_pct[i] / 100.0) * BAR_FULL_W
        ax.barh(BAR_TOP, width, left=left, height=BAR_H,
                color=color, alpha=0.92)
        if result.cluster_pct[i] > 6 and width > 0.05:
            text_x = left + width / 2
            ax.text(text_x, BAR_TOP,
                    f'{result.cluster_pct[i]:.0f}%',
                    ha='center', va='center',
                    color='white', fontsize=8.0,
                    fontweight='bold', clip_on=True)
        left += width

    #per zone table
    TABLE_TOP = BAR_TOP - BAR_H / 2 - 0.06       # header row y
    TABLE_BOT = 0.30
    
    row_gap   = min(0.085, (TABLE_TOP - TABLE_BOT) / (n + 0.5))

    
    ax.text(0.02, TABLE_TOP,
            f"{'Zone':<30}{'Pixels':>10}{'Area':>8}{'NDVI':>22}{'Conf':>8}",
            color='#777', fontsize=7.0,
            va='center', fontfamily='monospace')

    
    underline_y = TABLE_TOP - row_gap * 0.45
    ax.axhline(underline_y, xmin=0.02, xmax=0.98,
               color='#2a2a2a', linewidth=0.8)

    first_row_y = underline_y - row_gap * 0.55
    sorted_idx = list(np.argsort(result.cluster_ndvi_avg)[::-1])
    for rank, cidx in enumerate(sorted_idx):
        color = _zone_color(cidx)
        y = first_row_y - rank * row_gap
        name  = result.crop_names[cidx][:22]

        line = (
            f"{name:<22}"
            f"{result.cluster_counts[cidx]:>7,} px"
            f"{result.cluster_pct[cidx]:>7.1f}%"
            f"   {result.cluster_ndvi_avg[cidx]:>5.3f} ± {result.cluster_ndvi_std[cidx]:.3f}"
            f"   {result.cluster_conf_avg[cidx]:>4.2f}"
        )
        ax.text(0.02, y, line, color=color, fontsize=7.2,
                va='center', fontfamily='monospace')

    
    sep_y = first_row_y - n * row_gap
    sep_y = max(sep_y, 0.15)                    
    ax.axhline(sep_y, xmin=0.02, xmax=0.98,
               color='#2a2a2a', linewidth=0.8)

    
    meta_y = max(sep_y - 0.05, 0.10)
    ax.text(0.5, meta_y,
            f'Zones (BIC): {result.optimal_n}   |   '
            f'NDVI spread: {result.ndvi_diff:.3f}   |   '
            f'Avg confidence: {result.avg_confidence:.2f}', ha='center', va='center', color='#888888', fontsize=7.5)

    
    verdict_y = max(meta_y - 0.06, 0.03)
    ax.text(0.5, verdict_y, result.verdict,
            ha='center', va='center', color='#d5d5d5', fontsize=8.0, fontstyle='italic')


def _panel_pipeline(ax, chip, chip_enriched, mask_clean, emb_pixels, result, device, meta: dict):
    
    ax.set_facecolor(BG_SUMMARY)
    ax.set_xlim(0, 1);  ax.set_ylim(0, 1)
    ax.set_xticks([]);  ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor('#1e1e1e')
    ax.set_title('Pipeline', color=TITLE_COL,
                 fontsize=10, pad=5, fontweight='bold')

    # temporal date info
    dates = getattr(result, 'temporal_dates', [])
    n_dates = getattr(result, 'n_dates', 0) or len(dates)
    date_span = f'{dates[0]} to {dates[-1]}' if len(dates) >= 2 else str(DATES)

    bic_str = (f'[{min(result.bic_scores):.0f} - {max(result.bic_scores):.0f}]'
               if result.bic_scores else 'N/A')

    lines = [
        ('Device',   device.type.upper()),
        ('Temporal Stack',   f'{n_dates} dates analyzed'),
        ('Valid field pixels', f'{int(mask_clean.sum())}  (post temporal cloud filter)'),
        ('Embeddings',   f'{emb_pixels.shape}  (Prithvi patches)'),
        ('PCA components', str(result.field_pca.shape[1])),
        ('BIC scores',    bic_str),
        ('Optimal N (BIC)',  str(result.optimal_n)),
        ('Span',   date_span),
    ]

    for i, (key, val) in enumerate(lines):
        y = 0.87 - i * 0.105
        ax.text(0.04, y, f'{key}:', color=CAPTION,
                fontsize=7.2, va='center', fontfamily='monospace')
        ax.text(0.38, y, val,       color=LABEL_COL,
                fontsize=7.2, va='center', fontfamily='monospace')


#main function to build the dashboard
def build_dashboard(
    rgb, nir_false, ndvi_display, ndvi,
    feature_map,
    chip, chip_enriched,
    mask_224,
    mask_clean,
    emb_pixels,
    result: ClusterResult,
    device,
    meta: dict,
    save_path: str,
) -> None:
    
    
    dates  = getattr(result, 'temporal_dates', [])
    n_dates = getattr(result, 'n_dates', 0) or len(dates)
    t_label = f'Temporal Stack ({n_dates} Dates)  ·  ' if n_dates else ''

    fig = plt.figure(figsize=(22, 18), facecolor=BG_DARK)
    fig.suptitle(
        f'Prithvi-EO 2.0  |  Field {FIELD_ID}  |  {t_label}'
        f'GMM + PCA + BIC',
        fontsize=13, color='white', fontweight='bold', y=0.975,
    )

    gs = gridspec.GridSpec(
    4, 3, figure=fig,
    hspace=0.65,  
    wspace=0.28,
    height_ratios=[1, 1, 1, 0.55],  
)

    p = [
        fig.add_subplot(gs[0, 0]),   # 0  True colour
        fig.add_subplot(gs[0, 1]),   # 1  NIR false colour
        fig.add_subplot(gs[0, 2]),   # 2  NDVI
        fig.add_subplot(gs[1, 0]),   # 3  Encoder feature map
        fig.add_subplot(gs[1, 1]),   # 4  BIC curve
        fig.add_subplot(gs[1, 2]),   # 5  PCA scatter
        fig.add_subplot(gs[2, 0]),   # 6  Crop zone map
        fig.add_subplot(gs[2, 1]),   # 7  Confidence
        fig.add_subplot(gs[2, 2]),   # 8  NDVI per zone
        fig.add_subplot(gs[3, :2]),  # 9  Summary
        fig.add_subplot(gs[3, 2]),   # 10 Pipeline 
    ]

    _panel_true_colour (p[0],  rgb,mask_224)
    _panel_nir_false  (p[1],  nir_false,  mask_224)
    _panel_ndvi  (fig, p[2],  ndvi_display, mask_224)
    _panel_feature_map (fig, p[3],  feature_map,  mask_clean)
    _panel_bic_curve  (p[4], result)
    _panel_pca_scatter(p[5], result)
    _panel_crop_map  (p[6],  result,mask_224)
    _panel_confidence  (fig, p[7],  result,  mask_224, mask_clean)
    _panel_ndvi_clusters(p[8], result)
    _panel_summary  (p[9],  result)
    _panel_pipeline   (p[10], chip, chip_enriched, mask_clean,emb_pixels, result, device, meta)

    plt.savefig(save_path, dpi=160, bbox_inches='tight',facecolor=BG_DARK, edgecolor='none')
    plt.close(fig)
    print(f'Dashboard saved to: {save_path}')