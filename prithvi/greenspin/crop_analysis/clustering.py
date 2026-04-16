"""
clustering.py
-------------
Implements the clustering pipeline: normalisation, masking, PCA, GMM clustering, and computation of cluster statistics.
Defines a ClusterResult dataclass to hold all outputs and metrics from the clustering process.
"""
from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from config import N_CLUSTERS, PCA_COMPONENT, CHIP_SIZE, RANDOM_SEED


#result container

@dataclass
class ClusterResult:
    pixel_labels:   np.ndarray   
    confidence:  np.ndarray   
    pixel_cluster_map: np.ndarray   
    confidence_map:   np.ndarray   
    field_pca:       np.ndarray   
    pca_model:     PCA
    greener_idx:    int
    crop_names:     list[str]
    
    cluster_counts: list[int]
    cluster_ndvi_avg: list[float]
    cluster_conf_avg: list[float]
    cluster_pct:   list[float]
    avg_confidence:   float
    ndvi_diff:     float
    verdict:      str


#clustering pipeline

def run_clustering(
    emb_pixels:    np.ndarray,   # (H*W, embed_dim)
    chip_enriched: np.ndarray,   # (10, H, W)
    mask_224:    np.ndarray,   # (H, W)  binary float32
    ndvi:      np.ndarray,   # (H, W)
) -> ClusterResult:
    """clustering pipeline: normalise  mask -> PCA -> GMM -> stats."""

    spectral_pixels = chip_enriched.reshape(10, -1).T   # (H*W, 10)

    # normalise
    emb_norm   = StandardScaler().fit_transform(emb_pixels)
    spectral_norm = StandardScaler().fit_transform(spectral_pixels)
    
    # spatial features 
    H, W = mask_224.shape
    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

    coords = np.stack([xs.ravel(), ys.ravel()], axis=1)
    coords_norm = StandardScaler().fit_transform(coords)

    # control influence
    SPATIAL_WEIGHT = 0.5
    coords_norm *= SPATIAL_WEIGHT

    # combine and mask to field pixels only
    combined = np.concatenate([emb_norm, spectral_norm, coords_norm], axis=1)
    field_mask_flat  = mask_224.ravel() == 1
    field_combined   = combined[field_mask_flat]

    # PCA
    
    n_samples = field_combined.shape[0]
    n_components = min(PCA_COMPONENT, n_samples - 1, 5)
    pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
    field_pca = pca.fit_transform(field_combined)
    
    n_samples = field_pca.shape[0]

    # GMM
    if n_samples < 5:
        pixel_labels = np.zeros(n_samples, dtype=int)
        confidence = np.ones(n_samples)

    elif n_samples < 20:
        gmm = GaussianMixture(
            n_components=1,
            covariance_type='diag',
            random_state=RANDOM_SEED,
            reg_covar=1e-3
        )
        gmm.fit(field_pca)
        pixel_labels = gmm.predict(field_pca)
        confidence = np.ones(n_samples)

    else:
        gmm = GaussianMixture(
            n_components=N_CLUSTERS,
            covariance_type='full',
            random_state=RANDOM_SEED,
            n_init=5,
            reg_covar=1e-3
        )
        gmm.fit(field_pca)
        pixel_labels = gmm.predict(field_pca)
        confidence = gmm.predict_proba(field_pca).max(axis=1)

    # map back to 2D spatial grids
    pixel_cluster_map       = np.full(CHIP_SIZE * CHIP_SIZE, 2, dtype=int)
    pixel_cluster_map[field_mask_flat] = pixel_labels
    pixel_cluster_map      = pixel_cluster_map.reshape(CHIP_SIZE, CHIP_SIZE)

    confidence_map         = np.full(CHIP_SIZE * CHIP_SIZE, np.nan)
    confidence_map[field_mask_flat] = confidence
    confidence_map      = confidence_map.reshape(CHIP_SIZE, CHIP_SIZE)
    # per-cluster stats
    field_ndvi   = ndvi.ravel()[field_mask_flat]
    n_actual     = len(np.unique(pixel_labels))           
    cluster_counts   = [int(np.sum(pixel_labels == i))        for i in range(n_actual)]
    cluster_ndvi_avg = [float(np.mean(field_ndvi[pixel_labels == i])) for i in range(n_actual)]
    cluster_conf_avg = [float(np.mean(confidence[pixel_labels == i])) for i in range(n_actual)]
    cluster_pct      = [c / len(pixel_labels) * 100            for c in cluster_counts]

    # pad stats for missing clusters 
    while len(cluster_counts)   < N_CLUSTERS: cluster_counts.append(0)
    while len(cluster_ndvi_avg) < N_CLUSTERS: cluster_ndvi_avg.append(0.0)
    while len(cluster_conf_avg) < N_CLUSTERS: cluster_conf_avg.append(0.0)
    while len(cluster_pct)      < N_CLUSTERS: cluster_pct.append(0.0)

    # assign readable crop names
    greener_idx     = int(np.argmax(cluster_ndvi_avg))
    crop_names     = ['', '']
    crop_names[greener_idx]  = 'Crop A (higher NDVI)'
    crop_names[1-greener_idx]= 'Crop B (lower NDVI)'

    ndvi_diff= abs(cluster_ndvi_avg[0] - cluster_ndvi_avg[1])
    avg_conf  = float(np.mean(confidence))
    verdict = (
        'Strong spectral separation: likely two distinct crop types.'
        if ndvi_diff > 0.08 else
        'Weak spectral separation: could be stress zones rather than different crops.'
    )

    return ClusterResult(
        pixel_labels   = pixel_labels,
        confidence    = confidence,
        pixel_cluster_map = pixel_cluster_map,
        confidence_map  = confidence_map,
        field_pca      = field_pca,
        pca_model     = pca,
        greener_idx   = greener_idx,
        crop_names     = crop_names,
        cluster_counts   = cluster_counts,
        cluster_ndvi_avg = cluster_ndvi_avg,
        cluster_conf_avg = cluster_conf_avg,
        cluster_pct= cluster_pct,
        avg_confidence  = avg_conf,
        ndvi_diff        = ndvi_diff,
        verdict       = verdict,
    )