"""
reporting/batch_report.py

Assembles the complete batch summary figure from all panel functions and saves
to disk. build_batch_report() is the sole public API. Figure height is computed
dynamically from the number of table rows and thumbnail rows requested.

"""
from __future__ import annotations

import datetime as dt
import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from reporting.theme import THEME, apply_matplotlib_defaults
from reporting.metrics import (
    compute_batch_metrics,
    select_representative_fields,
)
from reporting.panels import (
    draw_kpi_card,
    panel_classification_bars,
    panel_segmentation_signature,
    panel_cluster_quality,
    panel_ranked_table,
    panel_field_thumbnail,
)


_H_TITLE  = 0.7
_H_KPI  = 1.1
_H_PANEL= 4.0
_H_TABLE_ROW = 0.32
_H_TABLE_HDR = 0.55
_H_THUMB = 3.2
_H_SPARK = 1.2
_H_THUMB_GAP = 0.3
_H_FOOTER= 0.4

_FIG_WIDTH = 18.0
_N_COLS = 6


def _build_figure_layout(n_table_rows: int, n_thumb_rows: int):
    table_h  = _H_TABLE_HDR + n_table_rows * _H_TABLE_ROW
    thumbs_h = n_thumb_rows * (_H_THUMB + _H_SPARK + _H_THUMB_GAP)
    total_h  = _H_TITLE + _H_KPI + _H_PANEL + table_h + thumbs_h + _H_FOOTER

    height_ratios = (
        [_H_KPI, _H_PANEL, table_h]
        + [_H_THUMB, _H_SPARK] * n_thumb_rows
    )

    fig = plt.figure(figsize=(_FIG_WIDTH, total_h))
    gs  = GridSpec(
        len(height_ratios), _N_COLS,
        figure = fig,
        height_ratios = height_ratios,
        hspace  = 0.65,
        wspace = 0.30,
        top= 1.0 - (_H_TITLE / total_h),
        bottom = _H_FOOTER / total_h,
        left= 0.04,
        right = 0.98,
    )
    return fig, gs, total_h


def build_batch_report(
    classifications: list,
    thumbnail_data: dict[int, dict],
    save_path: str,
    *,
    region_name:  str  = 'unspecified',
    season: str = 'unspecified',
    model_name:  str  = 'Prithvi-EO',
    n_dates:  int | None = None,
    n_thumbnails: int    = 6,
    max_table_rows:int = 12,
    n_total_processed: int | None = None,   
    theme= THEME,
) -> None:
    
    if not classifications:
        print('no classifications to plot')
        return

    apply_matplotlib_defaults(theme)

    metrics = compute_batch_metrics(classifications)
    representatives = select_representative_fields(
        classifications, target_total=n_thumbnails,
    )
    representatives = [
        c for c in representatives if c.field_id in thumbnail_data
    ][:n_thumbnails]

    n_thumb_rows = max(1, math.ceil(max(len(representatives), 1) / 3))

    from field_classifier import is_segmented
    n_seg= sum(1 for c in classifications if is_segmented(c))
    n_table_rows = max(1, min(n_seg, max_table_rows))

    fig, gs, total_h = _build_figure_layout(n_table_rows, n_thumb_rows)

    
    n_display = n_total_processed if n_total_processed else metrics.n_total
    n_seg_display = metrics.n_segmented

    title_y = 1.0 - (_H_TITLE * 0.30) / total_h
    sub_y= 1.0 - (_H_TITLE * 0.75) / total_h
    n_dates_str = f'{n_dates} dates' if n_dates else 'n/a'
    fig.text(0.04, title_y,
             'Phenotype Clustering - Batch Report',
             fontsize=18, fontweight='bold',
             color=theme.color_text, ha='left', va='center')
    fig.text(0.04, sub_y,
             f'{region_name}  |  season {season}  |  {n_dates_str} | '
    #          fontsize=11, color=theme.color_text_muted, ha='left', va='center')
    # fig.text(0.04, sub_y - 0.04,
             f' {n_display} fields processed  |  {n_seg_display} segmented',
             fontsize=11, color=theme.color_text_muted, ha='left', va='center')

    #KPI 
    pct_seg = 100.0 * n_seg_display / max(n_display, 1)
    kpi_cards = [
        (str(n_display),'fields processed',  theme.color_kpi_value),
        (str(n_seg_display),'segmented', theme.color_multi_crop),
        (f'{pct_seg:.0f}%','segmented %', theme.color_intra_crop),
        (f'{metrics.avg_silhouette:.2f}', 'avg silhouette', theme.color_homogeneous),
        (f'{metrics.avg_confidence:.2f}', 'avg confidence', theme.color_kpi_value),
    ]
    spans = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 6)]
    for (lo, hi), (val, lbl, accent) in zip(spans, kpi_cards):
        ax = fig.add_subplot(gs[0, lo:hi])
        draw_kpi_card(ax, val, lbl, accent=accent, theme=theme)

    #analytical panels
    panel_classification_bars(
        fig.add_subplot(gs[1, 0:2]), metrics, theme=theme)
    panel_segmentation_signature(
        fig.add_subplot(gs[1, 2:4]), classifications, theme=theme)
    panel_cluster_quality(
        fig.add_subplot(gs[1, 4:6]), classifications, theme=theme)

    #ranked table
    panel_ranked_table(
        fig.add_subplot(gs[2, :]),
        classifications, max_rows=max_table_rows, theme=theme)

    #  thumbnail
    for idx, c in enumerate(representatives):
        thumb_row = idx // 3
        col  = (idx % 3) * 2
        row_main  = 3 + thumb_row * 2
        row_spark = row_main + 1

        data = thumbnail_data.get(c.field_id)
        if data is None:
            continue

        ax_main= fig.add_subplot(gs[row_main,  col:col + 2])
        ax_spark = fig.add_subplot(gs[row_spark, col:col + 2])

        panel_field_thumbnail(
            ax_main= ax_main,
            ax_spark = ax_spark,
            classification   = c,
            rgb = data['rgb'],
            cluster_map = data['cluster_map'],
            ndvi_trajectory  = data.get('ndvi_trajectory'),
            mask = data.get('mask'),
            trajectory_dates = data.get('trajectory_dates'),
            theme  = theme,
        )

    # #footer
    # timestamp   = dt.datetime.now().strftime('%Y-%m-%d %H:%M')
    # n_dates_str = f'{n_dates} dates' if n_dates else 'n/a'
    # footer = (
    #     f'Generated {timestamp}   |   model: {model_name}   |   '
    #     f'{n_dates_str}   |   region: {region_name}   |   season: {season}'
    # )
    # fig.text(0.5, _H_FOOTER * 0.35 / total_h, footer,
    #          ha='center', va='bottom',
    #          fontsize=theme.font_size_footer,
    #          color=theme.color_text_muted)

    fig.savefig(save_path, dpi=theme.fig_dpi, facecolor=theme.color_bg,
                bbox_inches='tight')
    plt.close(fig)
    print(f'  Batch report saved: {save_path}')