"""
per_date_clustering.py

Single-date spectral GMM clustering applied independently per acquisition date.
Used as a temporal sanity check: compares per-date cluster counts against the
temporal-stack result and writes a k-comparison timeline chart and grid summary.

"""
from __future__ import annotations

import os
import csv
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import BoundaryNorm, ListedColormap
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from config import (
    MIN_CLUSTERS, MAX_CLUSTERS, PCA_COMPONENT, RANDOM_SEED, FIELD_ID,
)
from spectral import compute_indices, norm as _norm
from clustering import ClusterResult


CLUSTER_PALETTE = [
    '#27ae60',   # phenotype 0   
    '#e74c3c',   # phenotype 1   
    '#3498db',   # phenotype 2   
    '#f1c40f',   # phenotype 3  
    '#9b59b6',   # phenotype 4   
    '#1abc9c',   # phenotype 5  
    '#e67e22',   # phenotype 6   
    '#ec407a',   # phenotype 7   
]

BG_DARK   = '#0d0d0d'
BG_PANEL  = '#141414'
LABEL_COL = '#e0e0e0'
TITLE_COL = '#ffffff'
BORDER_COL = '#2a2a2a'
BOUNDARY  = '#FFD700'
CAPTION   = '#888888'
GOLD      = '#f39c12'


SINGLE_DATE_MAX_K = 20




def _zone_color(idx: int) -> str:
    return CLUSTER_PALETTE[idx % len(CLUSTER_PALETTE)]


def _draw_field_boundary(ax, mask_224: np.ndarray,
                         color: str = BOUNDARY, linewidth: float = 1.4):
    ax.contour(mask_224, levels=[0.5], colors=[color],
               linewidths=linewidth, linestyles='solid')


def _field_zoom(ax, mask_224: np.ndarray, pad: int = 22):
    rows, cols = np.where(mask_224 == 1)
    if rows.size > 0:
        ax.set_xlim(cols.min() - pad, cols.max() + pad)
        ax.set_ylim(rows.max() + pad, rows.min() - pad)



_NDVI_BINS   = [0.65, 0.50, 0.35, 0.20, 0.08]   
_NDVI_COLORS = [
    '#27ae60',   # dense         NDVI >= 0.65
    '#2ecc71',   # hi-veg        0.50 - 0.65
    '#f1c40f',   # mod-veg       0.35 - 0.50
    '#e67e22',   # sparse        0.20 - 0.35
    '#e74c3c',   # bare          0.08 - 0.20
    '#95a5a6',   # non-veg       < 0.08
]
_NDVI_BIN_LABELS = [
    'dense  (NDVI >= 0.65)',
    'hi-veg  (0.50 - 0.65)',
    'mod-veg  (0.35 - 0.50)',
    'sparse  (0.20 - 0.35)',
    'bare  (0.08 - 0.20)',
    'non-veg  (< 0.08)',
]


def _ndvi_to_bin(ndvi_val: float) -> int:
    for b, thresh in enumerate(_NDVI_BINS):
        if ndvi_val >= thresh:
            return b
    return len(_NDVI_BINS)


def _build_semantic_label_map(
    label_map_sorted: np.ndarray,
    cluster_ndvi_avg: list[float],
    n_clusters: int,
) -> tuple[np.ndarray, list[str]]:
    

    semantic_map  = np.full_like(label_map_sorted, -1)
    cluster_colors: list[str] = []

    for i in range(n_clusters):
        bin_idx = _ndvi_to_bin(cluster_ndvi_avg[i])
        semantic_map[label_map_sorted == i] = bin_idx
        cluster_colors.append(_NDVI_COLORS[bin_idx])

    return semantic_map, cluster_colors


def _semantic_cmap():
    
    colors = [BG_PANEL] + _NDVI_COLORS
    cmap   = ListedColormap(colors)
    bnorm  = BoundaryNorm(np.arange(-1.5, len(_NDVI_COLORS) + 0.5, 1), len(colors))
    return cmap, bnorm


#result container

@dataclass
class SingleDateResult:
    date:     str
    label_map:   np.ndarray    
    semantic_map:  np.ndarray   
    n_clusters:   int         
    bic_n_range:  list[int]
    bic_scores:   list[float]
    cluster_ndvi_avg: list[float]
    cluster_pct:   list[float]
    cluster_colors:   list[str]      
    ndvi_spread:  float
    silhouette:  float
    n_valid_pixels: int
    cloud_pct:    float
    skipped_reason:  str | None = None   

#single-date GMM pipeline

def _cluster_single_date(
    chip_t:    np.ndarray,   
    cloud_mask_t:  np.ndarray,    
    field_mask:  np.ndarray,    
    date_str:   str,
    min_valid_pixels: int = 50,
) -> SingleDateResult:

    H, W = field_mask.shape
    label_map  = np.full((H, W), -1, dtype=np.int32)
    semantic_map= np.full((H, W), -1, dtype=np.int32)

    valid   = (field_mask == 1) & (cloud_mask_t == 0)
    n_field = int((field_mask == 1).sum())
    n_valid = int(valid.sum())
    cloud_pct = (
        float(((field_mask == 1) & (cloud_mask_t == 1)).sum()) / max(n_field, 1) * 100
    )

    if n_valid < min_valid_pixels:
        return SingleDateResult(
            date       = date_str,
            label_map     = label_map,
            semantic_map   = semantic_map,
            n_clusters  = 0,
            bic_n_range = [],
            bic_scores   = [],
            cluster_ndvi_avg = [],
            cluster_pct  = [],
            cluster_colors  = [],
            ndvi_spread  = 0.0,
            silhouette   = 0.0,
            n_valid_pixels = n_valid,
            cloud_pct  = cloud_pct,
            skipped_reason  = f'{n_valid} clean field pixels ',
        )

    # 6 raw bands + 4 spectral indices (NDVI, NDWI, SAVI, NDRE)
    indices = compute_indices(chip_t)
    extra  = np.stack([indices['ndvi'], indices['ndwi'],indices['savi'], indices['ndre']], axis=0)
    feats_all = np.concatenate([chip_t, extra], axis=0)    

    feats = feats_all.reshape(feats_all.shape[0], -1).T  
    valid_flat  = valid.ravel()
    feats_valid = feats[valid_flat]                             
    ndvi_valid  = indices['ndvi'].ravel()[valid_flat]

    # PCA whiten
    feats_std = StandardScaler().fit_transform(feats_valid)
    n_comp  = min(PCA_COMPONENT, feats_std.shape[1] - 1, n_valid - 1)
    n_comp  = max(n_comp, 1)
    pca = PCA(n_components=n_comp, random_state=RANDOM_SEED)
    feats_pca = pca.fit_transform(feats_std)

    
    upper = min(SINGLE_DATE_MAX_K, max(MIN_CLUSTERS, n_valid // 100))
    upper  = max(upper, MIN_CLUSTERS)
    n_range = list(range(MIN_CLUSTERS, upper + 1))

    bic_scores: list[float] = []
    fitted: dict = {}
    for n in n_range:
        gmm = GaussianMixture(
            n_components   = n,
            covariance_type = 'full',
            random_state  = RANDOM_SEED,
            n_init   = 3,
            reg_covar   = 1e-3,
        )
        gmm.fit(feats_pca)
        bic_scores.append(float(gmm.bic(feats_pca)))
        fitted[n] = gmm

    optimal_n= n_range[int(np.argmin(bic_scores))]
    gmm  = fitted[optimal_n]
    labels  = gmm.predict(feats_pca)

    
    unique = np.unique(labels)
    if len(unique) < optimal_n:
        remap  = {old: new for new, old in enumerate(unique)}
        labels = np.array([remap[l] for l in labels], dtype=int)
        optimal_n = len(unique)

    # cluster 0 = greenest
    cluster_ndvi = np.array([
        float(np.mean(ndvi_valid[labels == i])) if (labels == i).any() else 0.0
        for i in range(optimal_n)
    ])
    order = np.argsort(cluster_ndvi)[::-1]
    rank_map = {old: new for new, old in enumerate(order)}

    flat_out = label_map.ravel().copy()
    flat_out[valid_flat] = labels
    label_map_raw = flat_out.reshape(H, W)

    label_map_sorted = label_map_raw.copy()
    for old, new in rank_map.items():
        label_map_sorted[label_map_raw == old] = new
    label_map = label_map_sorted

    cluster_ndvi_sorted = [float(cluster_ndvi[order[i]]) for i in range(optimal_n)]
    cluster_counts  = [int((label_map == i).sum()) for i in range(optimal_n)]
    total     = sum(cluster_counts) or 1
    cluster_pct  = [c / total * 100 for c in cluster_counts]
    ndvi_spread= (
        max(cluster_ndvi_sorted) - min(cluster_ndvi_sorted)
        if optimal_n > 1 else 0.0
    )

    
    semantic_map, cluster_colors = _build_semantic_label_map(
        label_map, cluster_ndvi_sorted, optimal_n,
    )

    sil = 0.0
    if optimal_n > 1 and n_valid > optimal_n:
        labs_sorted = label_map.ravel()[valid_flat]
        try:
            sil = float(silhouette_score(
                feats_pca, labs_sorted,
                sample_size  = min(5000, n_valid),
                random_state = RANDOM_SEED,
            ))
        except Exception:
            sil = 0.0

    return SingleDateResult(
        date    = date_str,
        label_map    = label_map,
        semantic_map  = semantic_map,
        n_clusters = optimal_n,
        bic_n_range= n_range,
        bic_scores  = bic_scores,
        cluster_ndvi_avg = cluster_ndvi_sorted,
        cluster_pct  = cluster_pct,
        cluster_colors  = cluster_colors,
        ndvi_spread  = ndvi_spread,
        silhouette   = sil,
        n_valid_pixels = n_valid,
        cloud_pct  = cloud_pct,
    )


#plot helpers

def _style(ax, title: str, subtitle: str = ''):
    pad = 22 if subtitle else 8
    ax.set_title(title, color=TITLE_COL, fontsize=10,
                 fontweight='semibold', pad=pad)
    if subtitle:
        ax.text(0.5, 1.03, subtitle, transform=ax.transAxes,
                ha='center', va='bottom', color=CAPTION, fontsize=7,
                fontstyle='italic')
    ax.set_facecolor(BG_PANEL)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER_COL)


#per-date 2-panel comparison PNG 

def _save_per_date_comparison(
    sd_result:   SingleDateResult,
    chip_t: np.ndarray,
    field_mask:  np.ndarray,
    cloud_mask_t: np.ndarray,
    out_path:  str,
):
    rgb  = np.stack([_norm(chip_t[2]), _norm(chip_t[1]), _norm(chip_t[0])], axis=-1)
    cloudy_field = (cloud_mask_t == 1) & (field_mask == 1)

    # 3 columns: RGB | zone map | legend-only axes
    # legend column is narrow (width_ratio 1) so it never overlaps the maps
    fig, axes = plt.subplots(
        1, 3,
        figsize     = (16, 6),
        facecolor   = BG_DARK,
        gridspec_kw = {'wspace': 0.05, 'width_ratios': [2, 2, 1]},
    )

    # panel 1 : true colour
    ax = axes[0]
    ax.imshow(rgb)
    ax.set_xticks([]); ax.set_yticks([])
    _draw_field_boundary(ax, field_mask)
    _field_zoom(ax, field_mask)
    _style(ax, f'True Colour  -  {sd_result.date}', subtitle='B4 - B3 - B2')
    if cloudy_field.any():
        ax.contourf(cloudy_field.astype(float),
                    levels=[0.5, 1.5], hatches=['////'],
                    colors=['#555555'], alpha=0.30)

    # panel 2 : single-date zones 
    ax = axes[1]
    if sd_result.skipped_reason:
        ax.set_facecolor(BG_PANEL)
        ax.text(0.5, 0.5,
                f'Skipped\n{sd_result.skipped_reason}',
                ha='center', va='center', color=CAPTION,
                fontsize=9, transform=ax.transAxes, fontstyle='italic')
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(BORDER_COL)
        _style(ax, 'k = -')
    else:
        cmap, bnorm = _semantic_cmap()
        ax.imshow(sd_result.semantic_map, cmap=cmap, norm=bnorm, interpolation='nearest')
        ax.set_xticks([]); ax.set_yticks([])
        _draw_field_boundary(ax, field_mask)
        _field_zoom(ax, field_mask)
        if cloudy_field.any():
            ax.contourf(cloudy_field.astype(float),
                        levels=[0.5, 1.5], hatches=['////'],
                        colors=['#555555'], alpha=0.30)

        handles = [
            Patch(
                facecolor = sd_result.cluster_colors[i],
                label     = (f'Cluster {i + 1}  '
                             f'NDVI {sd_result.cluster_ndvi_avg[i]:.2f}  '
                             f'({sd_result.cluster_pct[i]:.0f}%)'),
                edgecolor = 'none',
            )
            for i in range(sd_result.n_clusters)
        ]
        if cloudy_field.any():
            handles.append(
                Patch(facecolor='#555555', alpha=0.5,
                      label=f'Cloud / shadow  ({sd_result.cloud_pct:.0f}%)')
            )
        _style(ax,
               f'k = {sd_result.n_clusters}',
               subtitle=(f'NDVI spread {sd_result.ndvi_spread:.2f}  |  '
                         f'silhouette {sd_result.silhouette:.2f}  |  '
                         f'{sd_result.n_valid_pixels} clean field px'))

    # panel 3 : legend column - blank axes used purely as a legend host
    ax_leg = axes[2]
    ax_leg.set_facecolor(BG_DARK)
    ax_leg.set_xticks([]); ax_leg.set_yticks([])
    for sp in ax_leg.spines.values():
        sp.set_visible(False)
    if not sd_result.skipped_reason:
        ax_leg.legend(
            handles        = handles,
            loc            = 'center left',
            fontsize       = 8,
            facecolor      = BG_PANEL,
            edgecolor      = BORDER_COL,
            labelcolor     = LABEL_COL,
            framealpha     = 0.92,
            borderpad      = 1.0,
            labelspacing   = 0.6,
        )

    fig.suptitle(
        f'Field {FIELD_ID}  |  {sd_result.date}  |  Single-date spectral clustering',
        fontsize=12, color=TITLE_COL, fontweight='bold', y=1.01,
    )

    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor=BG_DARK, edgecolor='none')
    plt.close(fig)


#grid of all dates  

def _save_grid_summary(
    results:    list[SingleDateResult],
    field_mask: np.ndarray,
    out_path:   str,
):
    n = len(results)
    ncols = 6
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize  = (3.2 * ncols, 3.5 * nrows),
        facecolor  = BG_DARK,
        gridspec_kw = {'wspace': 0.06, 'hspace': 0.35},
    )
    axes = np.atleast_2d(axes)

    cmap, bnorm = _semantic_cmap()

    for idx, sd in enumerate(results):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        ax.set_facecolor(BG_PANEL)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(BORDER_COL)

        if sd.skipped_reason:
            ax.text(0.5, 0.5, 'skipped',
                    ha='center', va='center', color=CAPTION,
                    fontsize=8, transform=ax.transAxes, fontstyle='italic')
            ax.set_title(f'{sd.date}\nk = -',
                         color=LABEL_COL, fontsize=8, pad=4)
            continue

        ax.imshow(sd.semantic_map, cmap=cmap, norm=bnorm, interpolation='nearest')
        _draw_field_boundary(ax, field_mask, linewidth=0.9)
        _field_zoom(ax, field_mask, pad=18)
        ax.set_title(
            f'{sd.date}\nk = {sd.n_clusters},  spread {sd.ndvi_spread:.2f}',
            color=LABEL_COL, fontsize=8, pad=4,
        )

    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].axis('off')

    
    legend_handles = [
        Patch(facecolor=_NDVI_COLORS[i], label=_NDVI_BIN_LABELS[i], edgecolor='none')
        for i in range(len(_NDVI_COLORS))
    ]
    fig.legend(
        handles  = legend_handles,
        loc  = 'lower center',
        ncol   = 6,
        fontsize   = 8,
        facecolor   = BG_PANEL,
        edgecolor = BORDER_COL,
        labelcolor = LABEL_COL,
        framealpha = 0.92,
        bbox_to_anchor = (0.5, -0.01),
    )

    fig.suptitle(
        f'Field {FIELD_ID}  :  Per-Date Single-Date Phenotype Clustering  ({n} dates)\n',
        
        fontsize=12, color=TITLE_COL, fontweight='bold', y=1.01,
    )

    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor=BG_DARK, edgecolor='none')
    plt.close(fig)


#k-per-date timeline chart

def _save_k_timeline(
    results: list[SingleDateResult],
    temporal_n: int,
    out_path: str,
):
    dates = [r.date for r in results]
    ks = [r.n_clusters for r in results]
    spreads = [r.ndvi_spread for r in results]

    fig, ax1 = plt.subplots(
        figsize = (max(12, 0.38 * len(dates)), 5),
        facecolor = BG_DARK,
    )
    ax1.set_facecolor(BG_PANEL)

    bar_colors = [_zone_color(k - 1) if k > 0 else '#333333' for k in ks]
    ax1.bar(
        dates, ks,
        color     = bar_colors,
        edgecolor = BORDER_COL,
        linewidth = 0.6,
        zorder  = 2,
        label  = 'Single-date k values',
    )
    line1 = ax1.axhline(temporal_n, color=GOLD, linewidth=2.0, linestyle='--',
                        label=f'Temporal-stack k = {temporal_n}', zorder=3)
    ax1.set_xlabel('Date', color=LABEL_COL, fontsize=9)
    ax1.set_ylabel('Number of clusters', color=LABEL_COL, fontsize=9)
    valid_ks = [k for k in ks if k > 0]
    ax1.set_ylim(0, max(max(valid_ks) if valid_ks else 1, temporal_n) + 1)
    ax1.set_yticks(range(0, int(ax1.get_ylim()[1]) + 1))
    ax1.tick_params(axis='x', labelrotation=70, colors=LABEL_COL, labelsize=7)
    ax1.tick_params(axis='y', colors=LABEL_COL, labelsize=8)
    for sp in ax1.spines.values():
        sp.set_edgecolor(BORDER_COL)
    ax1.grid(axis='y', color='#444444', linestyle='--', linewidth=0.5, zorder=0)

    for i, r in enumerate(results):
        if r.skipped_reason:
            ax1.text(i, 0.15, 'x', ha='center', va='bottom',
                     color='#555555', fontsize=7)

    ax2 = ax1.twinx()
    line2, = ax2.plot(dates, spreads, color='#5dade2', marker='o', markersize=4,
             linewidth=1.2, label='NDVI spread (single-date)', zorder=4)
    ax2.set_ylabel('NDVI spread  (max - min cluster mean)',
                   color='#5dade2', fontsize=9)
    ax2.set_ylim(0, max(max(spreads) * 1.15, 0.1))
    ax2.tick_params(axis='y', colors='#5dade2', labelsize=8)
    for sp in ax2.spines.values():
        sp.set_edgecolor(BORDER_COL)

    ax1.set_title(
        f'Field {FIELD_ID}  |  Single-date k vs. temporal-stack k  '
        f'(temporal k = {temporal_n})',
        color=TITLE_COL, fontsize=11, fontweight='semibold', pad=8,
    )

    # only the 2 reference items - bar colours are readable from the y-axis labels
    ax1.legend(
        handles    = [line1, line2],
        loc        = 'upper right',
        fontsize   = 8,
        facecolor  = BG_PANEL,
        edgecolor  = BORDER_COL,
        labelcolor = LABEL_COL,
        framealpha = 0.92,
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor=BG_DARK, edgecolor='none')
    plt.close(fig)


#main entry point

def run_per_date_clustering(
    chip_temporal:np.ndarray,       
    cloud_masks:  np.ndarray,       
    field_mask:  np.ndarray,       
    used_dates:  list[str],
    temporal_result:  ClusterResult,
    output_dir:   str,
    min_valid_pixels: int = 50,
) -> list[SingleDateResult]:

    os.makedirs(output_dir, exist_ok=True)
    comparison_dir = os.path.join(output_dir, 'per_date_maps')
    os.makedirs(comparison_dir, exist_ok=True)

    print(f'\nPer-date clustering for {len(used_dates)} dates  ')
    

    results: list[SingleDateResult] = []

    for t, date_str in enumerate(used_dates):

        sd = _cluster_single_date(
            chip_t  = chip_temporal[t],
            cloud_mask_t = cloud_masks[t],
            field_mask  = field_mask,
            date_str  = date_str,
            min_valid_pixels = min_valid_pixels,
        )
        results.append(sd)

        if sd.skipped_reason:
            print(f'  {date_str}  SKIPPED  ({sd.skipped_reason})')
        else:
            print(f'  {date_str}  k={sd.n_clusters}  '
                  f'NDVI spread={sd.ndvi_spread:.3f}  '
                  f'sil={sd.silhouette:.2f}  '
                  f'valid={sd.n_valid_pixels}px  '
                  f'cloudy={sd.cloud_pct:.0f}%')

        out_png = os.path.join(comparison_dir, f'phenotype_map_{date_str}.png')
        _save_per_date_comparison(
            sd_result = sd,
            chip_t  = chip_temporal[t],
            field_mask= field_mask,
            cloud_mask_t = cloud_masks[t],
            out_path  = out_png,
        )

    _save_grid_summary(
        results,
        field_mask,
        os.path.join(output_dir, 'per_date_zones_grid.png'),
    )
    _save_k_timeline(
        results,
        temporal_result.optimal_n,
        os.path.join(output_dir, 'k_comparison_timeline.png'),
    )

    csv_path = os.path.join(output_dir, 'per_date_summary.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['date', 'k_single_date', 'k_temporal', 'ndvi_spread',
                    'silhouette', 'n_valid_pixels', 'cloud %',
                    'cluster_ndvi_means', 'cluster_pcts'])
        for r in results:
            w.writerow([
                r.date,
                r.n_clusters,
                temporal_result.optimal_n,
                f'{r.ndvi_spread:.4f}',
                f'{r.silhouette:.4f}',
                r.n_valid_pixels,
                f'{r.cloud_pct:.1f}',
                ';'.join(f'{v:.3f}' for v in r.cluster_ndvi_avg),
                ';'.join(f'{v:.1f}'  for v in r.cluster_pct),
            ])

    # summary
    valid_results = [r for r in results if not r.skipped_reason]
    if valid_results:
        ks  = [r.n_clusters for r in valid_results]
        spreads = [r.ndvi_spread for r in valid_results]
        
        print(f'  Single-date k :  min={min(ks)}  max={max(ks)}  '
              f'median={int(np.median(ks))}  mean={np.mean(ks):.1f}')
        
        
        n_match = sum(1 for k in ks if k == temporal_result.optimal_n)
        print(f'  Dates whose single-date k matches temporal k : '
              f'{n_match} / {len(valid_results)}')
    print(f'\n  Outputs written to : {output_dir}\n')

    return results