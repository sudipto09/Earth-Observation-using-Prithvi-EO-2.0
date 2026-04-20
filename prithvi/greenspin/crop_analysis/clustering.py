"""
clustering.py

Performs :
    1. Normalise features (embeddings + spectral + spatial)
    2. Mask to field pixels only
    3. PCA whitening for dimensionality reduction
    4. Fit GMMs with different cluster counts, select optimal N via BIC
    5. Map cluster labels and confidence back to 2D spatial grids
    6. Compute per-cluster statistics and overall summary metrics
    
"""
from dataclasses import dataclass
import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from config import MIN_CLUSTERS, MAX_CLUSTERS, PCA_COMPONENT, CHIP_SIZE, RANDOM_SEED


# results dataclass to hold all outputs from the clustering 

@dataclass
class ClusterResult:
    pixel_labels: np.ndarray    
    confidence:  np.ndarray    
    pixel_cluster_map: np.ndarray    
    confidence_map: np.ndarray    
    field_pca:   np.ndarray    
    pca_model:  PCA

    # BIC selection
    optimal_n: int           
    bic_n_range: list[int]     
    bic_scores: list[float]   

    # cluster identity
    greener_idx:  int    
    crop_names:   list[str]   

    # per-cluster statistics 
    cluster_counts: list[int]
    cluster_ndvi_avg: list[float]
    cluster_ndvi_std:list[float]  
    cluster_conf_avg: list[float]
    cluster_pct:  list[float]

    # summary metrics
    avg_confidence: float
    ndvi_diff:  float 
    verdict:    str


# bic helper

def _select_optimal_n(
    field_pca: np.ndarray,
) -> tuple[int, list[int], list[float]]:
    n_samples   = field_pca.shape[0]
    max_feasible = min(MAX_CLUSTERS, n_samples // 5)
    upper     = max(max_feasible, MIN_CLUSTERS)         
    n_range   = list(range(MIN_CLUSTERS, upper + 1))

    bic_scores: list[float] = []
    for n in n_range:
        gmm = GaussianMixture(
            n_components    = n,
            covariance_type = 'full',
            random_state    = RANDOM_SEED,
            n_init          = 3,
            reg_covar       = 1e-3,
        )
        gmm.fit(field_pca)
        bic_scores.append(float(gmm.bic(field_pca)))

    optimal_n = n_range[int(np.argmin(bic_scores))]
    return optimal_n, n_range, bic_scores


#main clustering function

def run_clustering(
    emb_pixels:    np.ndarray,   # (H*W, embed_dim)
    chip_enriched: np.ndarray,   # (10, H, W)
    mask_224:      np.ndarray,   # (H, W)  binary float32
    ndvi:          np.ndarray,   # (H, W)
) -> ClusterResult:
    """Full clustering pipeline: normalise - PCA - BIC-adaptive GMM - stats."""

    spectral_pixels = chip_enriched.reshape(10, -1).T   # (H*W, 10)

    # normalize 
    emb_norm      = StandardScaler().fit_transform(emb_pixels)
    spectral_norm = StandardScaler().fit_transform(spectral_pixels)

    H, W  = mask_224.shape
    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    coords   = np.stack([xs.ravel(), ys.ravel()], axis=1)
    coords_norm = StandardScaler().fit_transform(coords) * 0.5   # damped spatial weight

    # combine all features for clustering
    combined        = np.concatenate([emb_norm, spectral_norm, coords_norm], axis=1)
    field_mask_flat = mask_224.ravel() == 1
    field_combined  = combined[field_mask_flat]

    #pca whitening 
    n_samples  = field_combined.shape[0]
    n_components = min(PCA_COMPONENT, n_samples - 1, field_combined.shape[1] - 1)
    n_components = max(n_components, 1)
    pca   = PCA(n_components=n_components, random_state=RANDOM_SEED)
    field_pca = pca.fit_transform(field_combined)

    n_samples = field_pca.shape[0]

    # handle edge cases
    if n_samples < 5:
       
        pixel_labels = np.zeros(n_samples, dtype=int)
        confidence   = np.ones(n_samples)
        optimal_n   = 1
        bic_n_range = []
        bic_scores  = []

    elif n_samples < MIN_CLUSTERS * 5:
        
        gmm = GaussianMixture(
            n_components=1, covariance_type='diag',
            random_state=RANDOM_SEED, reg_covar=1e-3,
        )
        gmm.fit(field_pca)
        pixel_labels = gmm.predict(field_pca)
        confidence = np.ones(n_samples)
        optimal_n   = 1
        bic_n_range  = [1]
        bic_scores  = [float(gmm.bic(field_pca))]

    else:
        
        optimal_n, bic_n_range, bic_scores = _select_optimal_n(field_pca)

        gmm_final = GaussianMixture(
            n_components    = optimal_n,
            covariance_type = 'full',
            random_state   = RANDOM_SEED,
            n_init     = 5,
            reg_covar    = 1e-3,
        )
        gmm_final.fit(field_pca)
        pixel_labels = gmm_final.predict(field_pca)
        confidence = gmm_final.predict_proba(field_pca).max(axis=1)

    # map pixel labels and confidence back to full spatial grid
    pixel_cluster_map                     = np.full(CHIP_SIZE * CHIP_SIZE, optimal_n, dtype=int)
    pixel_cluster_map[field_mask_flat]    = pixel_labels
    pixel_cluster_map                     = pixel_cluster_map.reshape(CHIP_SIZE, CHIP_SIZE)

    confidence_map            = np.full(CHIP_SIZE * CHIP_SIZE, np.nan)
    confidence_map[field_mask_flat]   = confidence
    confidence_map    = confidence_map.reshape(CHIP_SIZE, CHIP_SIZE)

    # compute per-cluster statistics and summary 
    field_ndvi = ndvi.ravel()[field_mask_flat]
    n_actual  = len(np.unique(pixel_labels))   

    cluster_counts   = [int(np.sum(pixel_labels == i))                for i in range(n_actual)]
    cluster_ndvi_avg  = [float(np.mean(field_ndvi[pixel_labels == i])) for i in range(n_actual)]
    cluster_ndvi_std = [float(np.std( field_ndvi[pixel_labels == i])) for i in range(n_actual)]
    cluster_conf_avg  = [float(np.mean(confidence[pixel_labels == i])) for i in range(n_actual)]
    cluster_pct     = [c / len(pixel_labels) * 100                   for c in cluster_counts]

    
    while len(cluster_counts)   < optimal_n: cluster_counts.append(0)
    while len(cluster_ndvi_avg) < optimal_n: cluster_ndvi_avg.append(0.0)
    while len(cluster_ndvi_std) < optimal_n: cluster_ndvi_std.append(0.0)
    while len(cluster_conf_avg) < optimal_n: cluster_conf_avg.append(0.0)
    while len(cluster_pct)      < optimal_n: cluster_pct.append(0.0)

    
    ndvi_order  = list(np.argsort(cluster_ndvi_avg)[::-1])   # index 0 = highest NDVI
    greener_idx = int(ndvi_order[0])

    crop_names  = [''] * optimal_n
    for rank, cidx in enumerate(ndvi_order):
        if optimal_n == 1:
            crop_names[cidx] = 'Zone 1 (uniform field)'
        elif rank == 0:
            crop_names[cidx] = f'Zone {rank + 1} :high NDVI'
        elif rank == optimal_n - 1:
            crop_names[cidx] = f'Zone {rank + 1} :low NDVI'
        else:
            crop_names[cidx] = f'Zone {rank + 1}'

    # summary
    ndvi_diff = float(max(cluster_ndvi_avg) - min(cluster_ndvi_avg))
    avg_conf  = float(np.mean(confidence))

    if optimal_n == 1:
        verdict = 'Single homogeneous zone : no significant crop separation detected.'
    elif ndvi_diff > 0.15:
        verdict = (f'{optimal_n} distinct zones : strong spectral separation '
                   f'(NDVI spread {ndvi_diff:.3f}) : likely distinct crop types.')
    elif ndvi_diff > 0.05:
        verdict = (f'{optimal_n} zones · moderate separation '
                   f'(NDVI spread {ndvi_diff:.3f}) : possible mixed crop types or stress.')
    else:
        verdict = (f'{optimal_n} weakly-separated zones '
                   f'(NDVI spread {ndvi_diff:.3f}) : may reflect micro-variability.')

    return ClusterResult(
        pixel_labels    = pixel_labels,
        confidence  = confidence,
        pixel_cluster_map = pixel_cluster_map,
        confidence_map = confidence_map,
        field_pca  = field_pca,
        pca_model  = pca,
        optimal_n     = optimal_n,
        bic_n_range   = bic_n_range,
        bic_scores  = bic_scores,
        greener_idx = greener_idx,
        crop_names   = crop_names,
        cluster_counts = cluster_counts,
        cluster_ndvi_avg = cluster_ndvi_avg,
        cluster_ndvi_std = cluster_ndvi_std,
        cluster_conf_avg = cluster_conf_avg,
        cluster_pct    = cluster_pct,
        avg_confidence  = avg_conf,
        ndvi_diff    = ndvi_diff,
        verdict    = verdict,
    )