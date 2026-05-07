"""
reporting/panels.py

Individual matplotlib panel drawing functions for the batch report figure.
Each function draws into pre-created Axes objects passed by batch_report.py.
Includes: KPI cards, classification bars, scatter panels, ranked table, thumbnails.


"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle, Patch

from reporting.theme import Theme, THEME
from reporting.metrics import BatchMetrics, detect_outlier_fields



_PHENO_COLORS = [
    '#27ae60',  # P0  greenest
    '#e74c3c', # P1
    '#3498db',  # P2
    '#f1c40f', # P3
    '#9b59b6',  # P4
    '#1abc9c', # P5
    '#e67e22',   # P6
    '#ec407a',  # P7
]
_BOUNDARY_COLOR = '#FFD700'


def _field_bbox(mask: np.ndarray, pad: int = 18) -> tuple[int, int, int, int]:
    
    rows, cols = np.where(mask > 0.5)
    if rows.size == 0:
        return 0, mask.shape[0], 0, mask.shape[1]
    r0 = max(int(rows.min()) - pad, 0)
    r1 = min(int(rows.max()) + pad + 1, mask.shape[0])
    c0 = max(int(cols.min()) - pad, 0)
    c1 = min(int(cols.max()) + pad + 1, mask.shape[1])
    return r0, r1, c0, c1


#KPI card

def draw_kpi_card(
    ax: Axes, value: str, label: str,
    accent: str | None = None, theme: Theme = THEME,
) -> None:
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    accent = accent or theme.color_kpi_value
    ax.add_patch(Rectangle((0.02, 0.05), 0.96, 0.9,
                            facecolor=theme.color_bg_panel,
                            edgecolor=theme.color_border,
                            linewidth=0.6, zorder=0))
    ax.add_patch(Rectangle((0.02, 0.05), 0.04, 0.9,
                            facecolor=accent, edgecolor='none', zorder=1))
    ax.text(0.5, 0.62, value, ha='center', va='center',
            fontsize=theme.font_size_kpi, fontweight='bold',
            color=accent, transform=ax.transAxes)
    ax.text(0.5, 0.25, label, ha='center', va='center',
            fontsize=theme.font_size_kpi_lbl,
            color=theme.color_kpi_label, transform=ax.transAxes)


#Classification bar chart 

def panel_classification_bars(
    ax: Axes, metrics: BatchMetrics, theme: Theme = THEME,
) -> None:
    if not metrics.label_counts:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes, color=theme.color_text_muted)
        ax.axis('off'); return

    order  = ['homogeneous', 'intra-crop', 'multi-crop', 'weakly-variable']
    labels = [l for l in order if l in metrics.label_counts]
    values = [metrics.label_counts[l] for l in labels]
    colors = [theme.label_color(l) for l in labels]

    bars = ax.bar(labels, values, color=colors,
                  edgecolor=theme.color_text, linewidth=0.8,
                  alpha=theme.panel_alpha)
    ax.set_title(f'Field classification (segmented fields)', loc='left')
    ax.set_ylabel('count')
    ax.tick_params(axis='x', rotation=15)
    ax.grid(True, axis='y', alpha=0.3); ax.set_axisbelow(True)

    for bar, v in zip(bars, values):
        pct = 100 * v / max(metrics.n_total, 1)
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.02,
                f'{v}\n({pct:.0f}%)',
                ha='center', va='bottom',
                fontsize=theme.font_size_small, fontweight='bold',
                color=theme.color_text)
    ax.set_ylim(0, max(values) * 1.20)


#Segmentation signature

def panel_segmentation_signature(
    ax: Axes, classifications: list,
    theme: Theme = THEME, label_outliers_only: bool = True,
) -> None:
    if not classifications:
        ax.axis('off'); return

    outliers = (detect_outlier_fields(classifications) if label_outliers_only
                else {c.field_id for c in classifications})

    xs = np.array([c.ndvi_diff for c in classifications])
    ys = np.array([c.fragmentation for c in classifications])
    cs = [theme.label_color(c.label) for c in classifications]

    x_max = max(float(xs.max()), 0.30) * 1.10
    y_max = max(float(ys.max()), 1.5)  * 1.10
    x_thresh, y_thresh = 0.15, 5.0

    ax.axvline(x_thresh, color=theme.color_grid, linestyle='--', linewidth=0.8, zorder=0)
    ax.axhline(y_thresh, color=theme.color_grid, linestyle='--', linewidth=0.8, zorder=0)

    qstyle = dict(fontsize=theme.font_size_small - 1,
                  color=theme.color_text_muted, alpha=0.8, style='italic')
    ax.text(x_thresh * 0.05, y_max * 0.95, 'homogeneous', ha='left', va='top', **qstyle)
    ax.text(x_max * 0.98, y_max * 0.95, 'multi-crop\n(contiguous)', ha='right', va='top', **qstyle)
    if y_max > y_thresh:
        ax.text(x_max * 0.98, y_max * 0.05, 'scattered\nmulti-crop', ha='right', va='bottom', **qstyle)
        ax.text(x_thresh * 0.05, y_max * 0.05, 'stress patches', ha='left', va='bottom', **qstyle)

    ax.scatter(xs, ys, c=cs, s=60,
               edgecolor=theme.color_text, linewidth=0.5,
               alpha=theme.panel_alpha, zorder=3)

    for c, x, y in zip(classifications, xs, ys):
        if c.field_id in outliers:
            ax.annotate(str(c.field_id), (x, y),
                        fontsize=theme.font_size_small - 1,
                        xytext=(4, 4), textcoords='offset points',
                        color=theme.color_text)

    ax.set_xlabel('NDVI spread'); ax.set_ylabel('fragmentation')
    ax.set_title('Segmentation signature', loc='left')
    ax.set_xlim(0, x_max); ax.set_ylim(0, y_max)
    ax.grid(True, alpha=0.3); ax.set_axisbelow(True)


#Cluster quality 

def panel_cluster_quality(
    ax: Axes, classifications: list, theme: Theme = THEME,
) -> None:
    if not classifications:
        ax.axis('off'); return

    by_n: dict[int, list] = {}
    for c in classifications:
        by_n.setdefault(c.n_phenotypes_effective, []).append(c)

    n_values = sorted(by_n.keys())
    for n in n_values:
        group = by_n[n]
        sils  = np.array([c.silhouette for c in group])
        jitter = np.random.RandomState(n).uniform(-0.12, 0.12, size=len(group))
        xs = np.full_like(sils, float(n)) + jitter
        cs = [theme.label_color(c.label) for c in group]
        ax.scatter(xs, sils, c=cs, s=50,
                   edgecolor=theme.color_text, linewidth=0.4,
                   alpha=theme.panel_alpha, zorder=3)
        ax.hlines(float(np.median(sils)), n - 0.25, n + 0.25,
                  color=theme.color_text, linewidth=1.8, zorder=4)

    all_sils = [c.silhouette for c in classifications]
    ax.set_ylim(max(min(all_sils) - 0.05, -0.1),
                min(max(all_sils) + 0.05,  1.05))
    ax.set_xticks(n_values)
    ax.set_xlabel('effective phenotypes'); ax.set_ylabel('silhouette')
    ax.set_title('Cluster quality (median bars)', loc='left')
    ax.grid(True, axis='y', alpha=0.3); ax.set_axisbelow(True)


#table

def panel_ranked_table(
    ax: Axes, classifications: list,
    max_rows: int = 12, theme: Theme = THEME,
) -> None:
    from field_classifier import is_segmented

    segmented = [c for c in classifications if is_segmented(c)]
    multi = sorted([c for c in segmented if c.label == 'multi-crop'],
                   key=lambda c: (-c.ndvi_diff, -c.silhouette))
    intra = sorted([c for c in segmented if c.label == 'intra-crop'],
                   key=lambda c: (-c.ndvi_diff, -c.silhouette))

    rows_data: list = []
    i = j = 0
    while len(rows_data) < max_rows and (i < len(multi) or j < len(intra)):
        if i < len(multi): rows_data.append(multi[i]); i += 1
        if len(rows_data) >= max_rows: break
        if j < len(intra): rows_data.append(intra[j]); j += 1

    if not rows_data:
        ax.text(0.5, 0.5, 'No segmented fields', ha='center', va='center',
                transform=ax.transAxes, color=theme.color_text_muted)
        ax.axis('off'); return

    headers    = ['FID', 'class', 'n_eff', 'NDVI_sp', 'sil', 'frag', 'conf']
    cell_text  = []
    cell_colors = []

    for c in rows_data:
        tint = theme.label_color(c.label) + '22'
        cell_text.append([
            str(c.field_id), c.label,
            str(c.n_phenotypes_effective),
            f'{c.ndvi_diff:.3f}', f'{c.silhouette:.2f}',
            f'{c.fragmentation:.1f}', f'{c.avg_confidence:.2f}',
        ])
        row_cols = [theme.color_bg_panel] * 7
        row_cols[1] = tint
        cell_colors.append(row_cols)

    ax.axis('off')
    table = ax.table(
        cellText=cell_text, colLabels=headers,
        cellColours=cell_colors, cellLoc='center', loc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(theme.font_size_small)
    table.scale(1.0, 1.4)

    for j_idx in range(len(headers)):
        table[(0, j_idx)].set_text_props(fontweight='bold', color=theme.color_text)
        table[(0, j_idx)].set_facecolor(theme.color_bg)

    title = f'Top {len(rows_data)} segmented fields'
    if len(segmented) > max_rows:
        title += f'  (of {len(segmented)})'
    ax.set_title(title, loc='left', pad=10)


#Field thumbnail 

def panel_field_thumbnail(
    ax_main:  Axes,
    ax_spark: Axes | None,
    classification,
    rgb:   np.ndarray,           
    cluster_map:  np.ndarray,           
    ndvi_trajectory:  list | np.ndarray | None,
    theme:   Theme = THEME,
    mask: np.ndarray | None = None,
    trajectory_dates: list[str] | None = None,
    ndvi_trajectory_std: list | np.ndarray | None = None,
) -> None:
   
    conf   = classification.avg_confidence
    border = (theme.color_homogeneous if conf >= 0.85
              else theme.color_intra_crop if conf >= 0.70
              else theme.color_multi_crop)

    
    effective_mask = mask
    if effective_mask is None or effective_mask.sum() == 0:
        effective_mask = (cluster_map >= 0).astype(np.float32)

    if effective_mask.sum() > 0:
        r0, r1, c0, c1 = _field_bbox(effective_mask, pad=18)
    else:
        r0, c0 = 0, 0
        r1, c1 = rgb.shape[0], rgb.shape[1]

    img_crop  = rgb[r0:r1, c0:c1]
    cmap_crop = cluster_map[r0:r1, c0:c1]
    mask_crop = effective_mask[r0:r1, c0:c1]

    
    
    H_c, W_c = img_crop.shape[:2]
    ax_main.imshow(np.zeros((H_c, W_c, 3), dtype=np.float32),
                   interpolation='nearest', aspect='equal')

    # NIR background only inside the field
    nir_inside = img_crop.copy()
    if nir_inside.ndim == 3:
        outside = mask_crop < 0.5
        nir_inside[outside] = 0.0
    ax_main.imshow(nir_inside, interpolation='nearest', aspect='equal')

    #cluster overlay
    n_pheno  = classification.n_phenotypes_raw
    overlay  = np.zeros((H_c, W_c, 4), dtype=np.float32)

    for i in range(n_pheno):
        rgba = mcolors.to_rgba(_PHENO_COLORS[i % len(_PHENO_COLORS)])
        overlay[cmap_crop == i] = rgba

    
    overlay[mask_crop < 0.5] = (0, 0, 0, 0)

    ax_main.imshow(overlay, alpha=0.48, interpolation='nearest')

    
    if effective_mask.sum() > 0:
        ax_main.contour(effective_mask[r0:r1, c0:c1],
                        levels=[0.5], colors=[_BOUNDARY_COLOR],
                        linewidths=1.3, linestyles='solid')

    
    legend_handles = [
        Patch(facecolor=_PHENO_COLORS[i % len(_PHENO_COLORS)],
              label=f'P{i + 1}', edgecolor='none')
        for i in range(n_pheno)
    ]
    ax_main.legend(
        handles=legend_handles, loc='lower right',
        fontsize=theme.font_size_footer,
        facecolor=theme.color_bg_panel, edgecolor=theme.color_border,
        framealpha=0.85, handlelength=0.9, handleheight=0.8,
        borderpad=0.4, labelspacing=0.3,
    )

    ax_main.set_xticks([]); ax_main.set_yticks([])
    for spine in ax_main.spines.values():
        spine.set_edgecolor(border); spine.set_linewidth(2.5); spine.set_visible(True)

    label_color = {
        'multi-crop': theme.color_multi_crop,
        'intra-crop': theme.color_intra_crop,
    }.get(classification.label, theme.color_text_muted)

    ax_main.set_title(
        f'FID {classification.field_id}    {classification.label}   '
        f'n={classification.n_phenotypes_effective}  '
        f'NDVI sp={classification.ndvi_diff:.3f}',
        fontsize=theme.font_size_small, loc='left', color=label_color,
    )

    #sparkline
    if ax_spark is None:
        return

    
    if ndvi_trajectory is None:
        ax_spark.axis('off')
        return

    if isinstance(ndvi_trajectory, np.ndarray):
        traj_list = ([ndvi_trajectory.tolist()] if ndvi_trajectory.ndim == 1
                     else ndvi_trajectory.tolist())
    else:
        traj_list = list(ndvi_trajectory)

    
    traj_list = traj_list[:n_pheno]

    T = max((len(t) for t in traj_list), default=0)
    if T == 0:
        ax_spark.axis('off')
        ax_spark.text(0.5, 0.5, 'no trajectory',
                      ha='center', va='center',
                      transform=ax_spark.transAxes,
                      fontsize=theme.font_size_footer,
                      color=theme.color_text_muted)
        return

    x = np.arange(T)
    ax_spark.set_facecolor(theme.color_bg_panel)

    # Handle std trajectories if provided
    if isinstance(ndvi_trajectory_std, np.ndarray):
        std_list = ([ndvi_trajectory_std.tolist()] if ndvi_trajectory_std.ndim == 1
                    else ndvi_trajectory_std.tolist())
    else:
        std_list = list(ndvi_trajectory_std) if ndvi_trajectory_std else [None] * len(traj_list)
    
    std_list = std_list[:n_pheno]

    for i, traj in enumerate(traj_list):
        vals  = np.array(traj[:T], dtype=float)
        color = _PHENO_COLORS[i % len(_PHENO_COLORS)]
        
        # Plot mean trajectory
        ax_spark.plot(x, vals, color=color, linewidth=1.5,
                      marker='.', markersize=3.5, label=f'P{i + 1}', zorder=3)
        
        # Plot std shading if available
        if i < len(std_list) and std_list[i]:
            std_vals = np.array(std_list[i][:T], dtype=float)
            valid = ~np.isnan(vals) & ~np.isnan(std_vals)
            if valid.sum() > 1:
                lo = np.where(valid, vals - std_vals, np.nan)
                hi = np.where(valid, vals + std_vals, np.nan)
                ax_spark.fill_between(x, lo, hi, color=color, alpha=0.12, linewidth=0)
        else:
            # Fallback if no std: just fill from mean
            valid = ~np.isnan(vals)
            if valid.sum() > 1:
                ax_spark.fill_between(x, vals, alpha=0.10, color=color, where=valid)

    
    all_nan = np.all(
        np.isnan(np.array([t[:T] for t in traj_list], dtype=float)), axis=0)
    for xi, is_nan in enumerate(all_nan):
        if is_nan:
            ax_spark.axvline(xi, color=theme.color_grid,
                             linewidth=0.5, linestyle=':', zorder=1)

    
    if trajectory_dates and len(trajectory_dates) == T:
        labels = [d[5:] for d in trajectory_dates]   
        tick_step = max(1, T // 8)
        tick_pos  = list(range(0, T, tick_step))
        ax_spark.set_xticks(tick_pos)
        ax_spark.set_xticklabels(
            [labels[i] for i in tick_pos],
            rotation=40, ha='right',
            fontsize=theme.font_size_footer - 1,
        )
    else:
        ax_spark.set_xticks([])

    ax_spark.set_xlim(-0.5, T - 0.5)
    ax_spark.set_ylim(0.0, 1.0)
    ax_spark.set_yticks([0.0, 0.5, 1.0])
    ax_spark.tick_params(labelsize=theme.font_size_footer - 1)
    ax_spark.yaxis.set_tick_params(pad=1)
    ax_spark.grid(axis='y', alpha=0.25, linewidth=0.5)
    ax_spark.set_axisbelow(True)

    for spine in ax_spark.spines.values():
        spine.set_visible(False)

    ax_spark.text(0.01, 0.97, 'NDVI trajectory',
                  transform=ax_spark.transAxes,
                  fontsize=theme.font_size_footer,
                  color=theme.color_text_muted, va='top')

    
    if len(traj_list) > 1:
        ax_spark.legend(
            fontsize=theme.font_size_footer - 1,
            facecolor=theme.color_bg_panel,
            edgecolor=theme.color_border,
            framealpha=0.8, loc='upper right',
            handlelength=0.8, handleheight=0.7,
            borderpad=0.3, labelspacing=0.2,
            ncol=min(len(traj_list), 3),   
        )