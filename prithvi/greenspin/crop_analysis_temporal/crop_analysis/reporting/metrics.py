"""
reporting/metrics.py

Computes BatchMetrics aggregates (counts, averages) across all FieldClassification
objects. select_representative_fields() picks the most clearly segmented fields
for thumbnail display in the report, prioritising multi-crop over intra-crop.


"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from field_classifier import FieldClassification


@dataclass
class BatchMetrics:
    
    n_total: int
    n_segmented:   int
    n_homogeneous: int
    n_multi_crop: int
    n_intra_crop:  int
    n_weak: int

    pct_segmented: float
    avg_silhouette: float
    avg_ndvi_spread: float
    avg_confidence: float
    avg_fragmentation: float

    label_counts:dict = field(default_factory=dict)


def compute_batch_metrics(classifications: list) -> BatchMetrics:
    
    if not classifications:
        return BatchMetrics(
            n_total=0, n_segmented=0, n_homogeneous=0,
            n_multi_crop=0, n_intra_crop=0, n_weak=0,
            pct_segmented=0.0, avg_silhouette=0.0,
            avg_ndvi_spread=0.0, avg_confidence=0.0,
            avg_fragmentation=0.0, label_counts={},
        )

    label_counts: dict[str, int] = {}
    for c in classifications:
        label_counts[c.label] = label_counts.get(c.label, 0) + 1

    n_total  = len(classifications)
    n_multi = label_counts.get('multi-crop', 0)
    n_intra  = label_counts.get('intra-crop', 0)
    n_homo  = label_counts.get('homogeneous', 0)
    n_weak  = label_counts.get('weakly-variable', 0)
    n_segmented  = n_multi + n_intra

    return BatchMetrics(
        n_total  = n_total,
        n_segmented = n_segmented,
        n_homogeneous  = n_homo,
        n_multi_crop = n_multi,
        n_intra_crop   = n_intra,
        n_weak  = n_weak,
        pct_segmented = 100.0 * n_segmented / max(n_total, 1),
        avg_silhouette  = float(np.mean([c.silhouette for c in classifications])),
        avg_ndvi_spread = float(np.mean([c.ndvi_diff for c in classifications])),
        avg_confidence = float(np.mean([c.avg_confidence for c in classifications])),
        avg_fragmentation = float(np.mean([c.fragmentation for c in classifications])),
        label_counts  = label_counts,
    )


def select_representative_fields(
    classifications: list,
    n_per_class: int = 2,
    target_total: int | None = None,
) -> list:
    
    by_class: dict[str, list] = {}
    for c in classifications:
        by_class.setdefault(c.label, []).append(c)

    
    for label in by_class:
        by_class[label].sort(key=lambda c: (-c.ndvi_diff, -c.silhouette))

    representatives: list = []
    priority = ['multi-crop', 'intra-crop', 'weakly-variable']

    
    for label in priority:
        if label in by_class:
            representatives.extend(by_class[label][:n_per_class])

    
    if target_total is not None and len(representatives) < target_total:
        already = {id(c) for c in representatives}
        leftover = []
        for label in priority:
            if label in by_class:
                leftover.extend(by_class[label][n_per_class:])
        leftover.sort(key=lambda c: (-c.ndvi_diff, -c.silhouette))
        for c in leftover:
            if id(c) not in already:
                representatives.append(c)
                if len(representatives) >= target_total:
                    break

    return representatives


def detect_outlier_fields(
    classifications: list,
    iqr_multiplier: float = 1.5,
) -> set[int]:
    
    if len(classifications) < 4:
        return {c.field_id for c in classifications}

    ndvi = np.array([c.ndvi_diff for c in classifications])
    frag= np.array([c.fragmentation for c in classifications])

    def _iqr_bounds(x: np.ndarray) -> tuple[float, float]:
        q1, q3 = np.percentile(x, [25, 75])
        iqr = q3 - q1
        return q1 - iqr_multiplier * iqr, q3 + iqr_multiplier * iqr

    nlo, nhi =_iqr_bounds(ndvi)
    flo, fhi= _iqr_bounds(frag)

    outliers = set()
    for i, c in enumerate(classifications):
        if ndvi[i] < nlo or ndvi[i] > nhi or frag[i] < flo or frag[i] > fhi:
            outliers.add(c.field_id)
    return outliers