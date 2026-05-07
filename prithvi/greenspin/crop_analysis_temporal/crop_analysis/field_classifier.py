"""
field_classifier.py

Rule-based classifier: maps a ClusterResult to one of four agronomic categories
(homogeneous / weakly-variable / intra-crop / multi-crop) using NDVI spread,
fragmentation index, and silhouette score. is_segmented() gates the full pipeline.

"""
from dataclasses import dataclass
import numpy as np
from scipy.ndimage import label as ndlabel

from clustering import ClusterResult


# thresholds  

STRONG_NDVI_SEP   = 0.15   #land-cover difference
MODERATE_NDVI_SEP = 0.05   # within-crop variability
MIN_SILHOUETTE    = 0.10   # below this = clusters not well separated
MIN_PHENOTYPE_PCT = 5.0    
FRAG_MULTI_CROP_MAX = 5.0  #  (multi-crop signature)
FRAG_INTRA_HIGH     = 15.0 # scattered patches 


# result container

@dataclass
class FieldClassification:
    field_id: int
    label: str                
    n_phenotypes_effective: int   
    n_phenotypes_raw: int     
    ndvi_diff: float
    silhouette: float
    db_score: float
    dominant_pct: float
    fragmentation: float
    avg_confidence: float
    n_field_pixels: int
    verdict: str         


# fragmentation 

def _compute_fragmentation(
    cluster_map: np.ndarray,
    n: int,
    min_blob_size: int = 20,
) -> float:
    
    if n <= 1:
        return 0.0

    total_components = 0
    for i in range(n):
        binary = (cluster_map == i).astype(int)
        if binary.sum() == 0:
            continue
        _, num = ndlabel(binary)
       
        if num > 0:
            labeled, _ = ndlabel(binary)
            sizes = np.bincount(labeled.ravel())[1:]   
            num   = int((sizes >= min_blob_size).sum())
        total_components += num

    return total_components / max(n, 1)


# main classifier

def classify_field(
    field_id: int,
    result: ClusterResult,
) -> FieldClassification:
    
    n_raw = result.optimal_n
    pcts = result.cluster_pct
    ndvi_diff = result.ndvi_diff
    sil    = result.silhouette

    # phenotypes that cover >= MIN_PHENOTYPE_PCT
    big_phenotypes = [i for i, p in enumerate(pcts) if p >= MIN_PHENOTYPE_PCT]
    n_eff = len(big_phenotypes)

    fragmentation = _compute_fragmentation(result.pixel_cluster_map, n_raw)

    n_field_px = int((result.pixel_cluster_map >= 0).sum())
    dominant_pct = float(max(pcts)) if pcts else 100.0

    #logic
    if n_eff <= 1:
        label = 'homogeneous'
        verdict = (f'Single dominant phenotype ({dominant_pct:.0f}% of field). '
                   f'Likely uniform crop / single land cover.')

    elif (not (sil != sil) and sil < MIN_SILHOUETTE) and ndvi_diff < MODERATE_NDVI_SEP:  
        label = 'weakly-variable'
        verdict = (f'GMM found {n_raw} phenotypes but cluster separation is weak '
                   f'(silhouette={sil:.2f}, NDVI spread={ndvi_diff:.3f}). '
                   f'Treat as effectively homogeneous.')

    elif ndvi_diff >= STRONG_NDVI_SEP and fragmentation <= FRAG_MULTI_CROP_MAX:
        label = 'multi-crop'
        verdict = (f'{n_eff} phenotypes with strong NDVI separation '
                   f'({ndvi_diff:.3f}) and contiguous blobs (frag={fragmentation:.1f}). '
                   f'Likely genuinely multi-cropped or split land use.')

    elif ndvi_diff >= STRONG_NDVI_SEP:
        # strong NDVI diff but scattered  
        label = 'multi-crop'
        verdict = (f'{n_eff} phenotypes with strong NDVI separation '
                   f'({ndvi_diff:.3f}) but scattered (frag={fragmentation:.1f}). '
                   f'Possible interleaved crops or salt-and-pepper land cover.')

    elif ndvi_diff >= MODERATE_NDVI_SEP and fragmentation >= FRAG_INTRA_HIGH:
        label = 'intra-crop'
        verdict = (f'{n_eff} phenotypes with moderate NDVI spread '
                   f'({ndvi_diff:.3f}) and high fragmentation '
                   f'(frag={fragmentation:.1f}). Likely stress patches / '
                   f'within-field variability of one crop.')

    elif ndvi_diff >= MODERATE_NDVI_SEP:
        label = 'intra-crop'
        verdict = (f'{n_eff} phenotypes with moderate NDVI spread '
                   f'({ndvi_diff:.3f}, frag={fragmentation:.1f}). '
                   f'Within-field variability — same crop, different vigour.')

    else:
        label = 'weakly-variable'
        verdict = (f'{n_eff} phenotypes but NDVI spread is small '
                   f'({ndvi_diff:.3f}). Likely temporal-trajectory subtleties '
                   f'rather than distinct crop classes.')

    return FieldClassification(
        field_id  = field_id,
        label  = label,
        n_phenotypes_effective = n_eff,
        n_phenotypes_raw  = n_raw,
        ndvi_diff = ndvi_diff,
        silhouette = sil,
        db_score = result.db_score,
        dominant_pct = dominant_pct,
        fragmentation = fragmentation,
        avg_confidence = result.avg_confidence,
        n_field_pixels = n_field_px,
        verdict   = verdict,
    )




def is_segmented(classification: FieldClassification) -> bool:
    
    return classification.label in ('multi-crop', 'intra-crop')