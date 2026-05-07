"""
clustering.py

Core BIC-optimal GMM phenotype clustering. Fuses Prithvi embeddings, temporal
NDVI statistics, spectral features, and spatial coordinates into one feature matrix,
then runs PCA whitening, GMM model selection, trajectory-based cluster merging,
and phenotype naming. Returns a ClusterResult dataclass with all outputs.

"""
from dataclasses import dataclass, field
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from config import (
    MAX_CLUSTERS, MIN_CLUSTERS, PCA_COMPONENT, RANDOM_SEED,
    TEMPORAL_STAT_WEIGHT,
)


#result container

@dataclass
class ClusterResult:
    # pixel-level outputs
    pixel_labels:  np.ndarray        
    confidence:  np.ndarray        
    pixel_cluster_map: np.ndarray      
    confidence_map: np.ndarray        

    # dimensionality-reduction state
    field_pca: np.ndarray              
    pca_model:  PCA

    # BIC selection
    optimal_n: int
    bic_n_range: list[int]
    bic_scores:list[float]
    
    greener_idx:int                   # index of the highest-NDVI phenotype
    crop_names: list[str]

    # per phenotype stats
    cluster_counts: list[int]
    cluster_ndvi_avg: list[float]
    cluster_ndvi_std:list[float]
    cluster_conf_avg:list[float]
    cluster_pct:list[float]
   
    # summary
    avg_confidence: float
    ndvi_diff:float
    verdict:str

    # cluster quality metrics
    silhouette: float = 0.0          
    db_score: float = 0.0         

    temporal_dates: list[str] = field(default_factory=list)
    n_dates:    int       = 0


#bic selection helper

def _select_optimal_n(
    field_pca: np.ndarray,
    effective_max: int = MAX_CLUSTERS,
    min_n: int = MIN_CLUSTERS,
    n_field_patches: int = 196,       
) -> tuple[int, list[int], list[float]]:

    n_samples = field_pca.shape[0]
    max_feasible = min(effective_max, n_samples // 5)
    upper = max(max_feasible, min_n)
    n_range = list(range(min_n, upper + 1))

    bic_scores: list[float] = []
    for n in n_range:
        gmm = GaussianMixture(
            n_components=n,
            covariance_type='full',
            random_state=RANDOM_SEED,
            n_init=3,
            reg_covar=1e-3,
        )
        gmm.fit(field_pca)
        bic_scores.append(float(gmm.bic(field_pca)))

    if n_field_patches < 30:
        penalty_weight = max(0.0, 1.0 - n_field_patches / 30.0)
        bic_scores = [
            s + penalty_weight * 0.05 * abs(s) * (n - min_n)
            for n, s in zip(n_range, bic_scores)
        ]

    optimal_n = n_range[int(np.argmin(bic_scores))]
    return optimal_n, n_range, bic_scores


#main clustering pipeline

def run_clustering(
    emb_pixels: np.ndarray,
    chip_enriched: np.ndarray,
    mask_224:  np.ndarray,
    ndvi:   np.ndarray,
    n_field_patches: int = 196,
    temporal_stats: np.ndarray | None = None,
) -> ClusterResult:
    
    H, W = mask_224.shape
    C = chip_enriched.shape[0]
    spectral_pixels = chip_enriched.reshape(C, -1).T

    
    if n_field_patches < 10:
        effective_max = max(1, min(MAX_CLUSTERS, n_field_patches // 2))
        
    else:
        effective_max = MAX_CLUSTERS

    
    emb_weight = 1.0 if n_field_patches >= 10 else max(0.1, n_field_patches / 20.0)

    # normalize features
    emb_norm  = StandardScaler().fit_transform(emb_pixels) * emb_weight
    spectral_norm = StandardScaler().fit_transform(spectral_pixels)

    # spatial coordinates
    ys, xs   = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    coords = np.stack([xs.ravel(), ys.ravel()], axis=1)
    coords_norm = StandardScaler().fit_transform(coords) * 0.5

    # (NDVI trajectory + embedding variability) 
    if temporal_stats is not None:
        ts_norm = StandardScaler().fit_transform(temporal_stats) * TEMPORAL_STAT_WEIGHT
        combined = np.concatenate([emb_norm, ts_norm, spectral_norm, coords_norm], axis=1)
    else:
        combined = np.concatenate([emb_norm, spectral_norm, coords_norm], axis=1)

    field_mask_flat = mask_224.ravel() > 0
    field_combined = combined[field_mask_flat]
    n_samples = field_combined.shape[0]

    if n_samples == 0:
        raise ValueError("No valid field pixels after masking.")
    
    #pca whitening
    n_components= min(PCA_COMPONENT, n_samples - 1, field_combined.shape[1] - 1)
    n_components = max(n_components, 1)

    pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
    field_pca = pca.fit_transform(field_combined)

   
    BIMODAL_IQR_THRESH = 0.20   
    forced_min = MIN_CLUSTERS
    if temporal_stats is not None:
        ts_field_pre = temporal_stats[field_mask_flat]   # (n_samples, n_dims)
        field_mean_ndvi = ts_field_pre[:, 0]             # col 0 = mean NDVI per pixel
        ndvi_p25 = float(np.percentile(field_mean_ndvi, 25))
        ndvi_p75 = float(np.percentile(field_mean_ndvi, 75))
        if (ndvi_p75 - ndvi_p25) > BIMODAL_IQR_THRESH:
            forced_min = max(MIN_CLUSTERS, 2)

    #gmm with BIC
    if n_samples < 5:
        pixel_labels= np.zeros(n_samples, dtype=int)
        confidence= np.ones(n_samples)
        optimal_n  = 1
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
        bic_n_range= [1]
        bic_scores= [float(gmm.bic(field_pca))]

    else:
        optimal_n, bic_n_range, bic_scores = _select_optimal_n(
            field_pca, effective_max, min_n=forced_min, n_field_patches=n_field_patches
        )

        gmm_final= GaussianMixture(
            n_components= optimal_n,
            covariance_type = 'full',
            random_state = RANDOM_SEED,
            n_init = 5,
            reg_covar  = 1e-3,
        )
        gmm_final.fit(field_pca)
        pixel_labels = gmm_final.predict(field_pca)
        confidence = gmm_final.predict_proba(field_pca).max(axis=1)

        # remap labels 
        unique_labels = np.unique(pixel_labels)
        if len(unique_labels) < optimal_n:
            remap = {old: new for new, old in enumerate(unique_labels)}
            pixel_labels = np.array([remap[l] for l in pixel_labels], dtype=int)
            optimal_n = len(unique_labels)

    # maps results back to pixel space
   
    pixel_cluster_map = np.full(H * W, -1, dtype=int)
    pixel_cluster_map[field_mask_flat] = pixel_labels
    pixel_cluster_map  = pixel_cluster_map.reshape(H, W)

    confidence_map = np.full(H * W, np.nan, dtype=np.float32)
    confidence_map[field_mask_flat] = confidence
    confidence_map = confidence_map.reshape(H, W)

    
    field_ndvi = ndvi.ravel()[field_mask_flat]

    cluster_counts: list[int] = []
    cluster_ndvi_avg: list[float] = []
    cluster_ndvi_std: list[float] = []
    cluster_conf_avg: list[float] = []
    for i in range(optimal_n):
        sel = pixel_labels == i
        cluster_counts.append(int(sel.sum()))
        if sel.any():
            cluster_ndvi_avg.append(float(np.mean(field_ndvi[sel])))
            cluster_ndvi_std.append(float(np.std(field_ndvi[sel])))
            cluster_conf_avg.append(float(np.mean(confidence[sel])))
        else:
            cluster_ndvi_avg.append(0.0)
            cluster_ndvi_std.append(0.0)
            cluster_conf_avg.append(0.0)
    cluster_pct = [c / len(pixel_labels) * 100 for c in cluster_counts]

    ndvi_diff = float(max(cluster_ndvi_avg) - min(cluster_ndvi_avg))

  
    TRAJ_CORR_THRESH  = 0.97   
    NDVI_MERGE_MAX    = 0.10   

    def _rebuild_stats(labels, n, ndvi_vals, conf_vals):
        counts, avgs, stds, confs = [], [], [], []
        for i in range(n):
            sel = labels == i
            counts.append(int(sel.sum()))
            if sel.any():
                avgs.append(float(np.mean(ndvi_vals[sel])))
                stds.append(float(np.std(ndvi_vals[sel])))
                confs.append(float(np.mean(conf_vals[sel])))
            else:
                avgs.append(0.0); stds.append(0.0); confs.append(0.0)
        pcts = [c / max(len(labels), 1) * 100 for c in counts]
        return counts, avgs, stds, confs, pcts

    if optimal_n > 1 and temporal_stats is not None:
        ts_field = temporal_stats[field_mask_flat]       # (n_samples, n_dims)
        do_merge = True
        while do_merge and optimal_n > 1:
            do_merge = False

            cluster_ts = np.array([
                ts_field[pixel_labels == i].mean(axis=0)
                if (pixel_labels == i).any()
                else np.zeros(ts_field.shape[1])
                for i in range(optimal_n)
            ])                                           # (optimal_n, n_dims)

            best_pair = None
            best_r    = -1.0
            for i in range(optimal_n):
                for j in range(i + 1, optimal_n):
                    
                    ndvi_dist = abs(cluster_ndvi_avg[i] - cluster_ndvi_avg[j])
                    if ndvi_dist >= NDVI_MERGE_MAX:
                        continue                        

                    vi, vj = cluster_ts[i], cluster_ts[j]
                    denom  = np.std(vi) * np.std(vj)
                    if denom < 1e-9:
                        r = 1.0 if ndvi_dist < NDVI_MERGE_MAX else -1.0
                    else:
                        r = float(np.corrcoef(vi, vj)[0, 1])
                    if r > best_r:
                        best_r, best_pair = r, (i, j)

            if best_r >= TRAJ_CORR_THRESH and best_pair is not None:
                i_keep, i_drop = best_pair
                pixel_labels[pixel_labels == i_drop] = i_keep
                pixel_labels[pixel_labels > i_drop] -= 1
                optimal_n -= 1

                pixel_cluster_map = np.full(H * W, -1, dtype=int)
                pixel_cluster_map[field_mask_flat] = pixel_labels
                pixel_cluster_map = pixel_cluster_map.reshape(H, W)

                cluster_counts, cluster_ndvi_avg, cluster_ndvi_std, \
                    cluster_conf_avg, cluster_pct = _rebuild_stats(
                        pixel_labels, optimal_n, field_ndvi, confidence)
                ndvi_diff = float(max(cluster_ndvi_avg) - min(cluster_ndvi_avg))
                do_merge  = True

    
    if optimal_n > 1 and n_samples > optimal_n:
        sample_size = min(5000, n_samples)
        sil = float(silhouette_score(
            field_pca, pixel_labels,
            sample_size=sample_size, random_state=RANDOM_SEED,
        ))
        db = float(davies_bouldin_score(field_pca, pixel_labels))
    else:
        sil = float('nan')
        db  = 0.0

    #phenotype naming by NDVI rank
    ndvi_order = list(np.argsort(cluster_ndvi_avg)[::-1])    # 0 = highest NDVI
    greener_idx = int(ndvi_order[0])

    crop_names = [''] * optimal_n

    def _ndvi_label(v: float) -> str:
        
        if v >= 0.65: return 'dense'
        if v >= 0.50: return 'hi-veg'
        if v >= 0.35: return 'mod-veg'
        if v >= 0.20: return 'sparse'
        if v >= 0.08: return 'bare'
        return 'non-veg'

    from collections import Counter
    all_labels = [_ndvi_label(v) for v in cluster_ndvi_avg]
    dominant_label, dominant_count = Counter(all_labels).most_common(1)[0]
    dominant_frac  = dominant_count / max(len(all_labels), 1)
    texture_driven = dominant_frac >= 0.6 and ndvi_diff < 0.15
    ndvi_driven    = ndvi_diff >= 0.15

    for rank, cidx in enumerate(ndvi_order):
        ndvi_val = cluster_ndvi_avg[cidx]
        short = _ndvi_label(ndvi_val)
        if optimal_n == 1:
            crop_names[cidx] = f'Phenotype 1 ({short})'
        elif rank == 0:
            crop_names[cidx] = f'Phenotype {rank + 1} : high NDVI ({short})'
        elif rank == optimal_n - 1:
            crop_names[cidx] = f'Phenotype {rank + 1} : low NDVI ({short})'
        elif texture_driven:
           
            crop_names[cidx] = f'Phenotype {rank + 1} (temporal-{rank + 1})'
        else:
            crop_names[cidx] = f'Phenotype {rank + 1} ({short})'

    #summary
    avg_conf= float(np.mean(confidence))

    if optimal_n == 1:
        verdict = 'Single homogeneous phenotype : no significant temporal separation detected.'
    elif ndvi_diff > 0.15:
        verdict = (
            f'{optimal_n} distinct phenotypes | strong NDVI separation '
            f'(spread {ndvi_diff:.3f}) | likely distinct crop types or strong stress.'
        )
    elif ndvi_diff > 0.05:
        if texture_driven:
            
            verdict = (
                f'{optimal_n} phenotypes | temporally-driven separation '
                f'(NDVI spread {ndvi_diff:.3f}, dominant class: {dominant_label}) | '
                f'similar greenness but different seasonal trajectories.'
            )
        else:
            verdict = (
                f'{optimal_n} phenotypes | moderate NDVI separation '
                f'(spread {ndvi_diff:.3f}) | possible mixed crop types or stress patches.'
            )
    else:
        verdict = (
            f'{optimal_n} weakly-separated phenotypes '
            f'(NDVI spread {ndvi_diff:.3f}) | micro-variability or temporal trajectory pattern.'
        )

    return ClusterResult(
        pixel_labels   = pixel_labels,
        confidence   = confidence,
        pixel_cluster_map = pixel_cluster_map,
        confidence_map  = confidence_map,
        field_pca  = field_pca,
        pca_model   = pca,
        optimal_n  = optimal_n,
        bic_n_range= bic_n_range,
        bic_scores    = bic_scores,
        greener_idx  = greener_idx,
        crop_names      = crop_names,
        cluster_counts   = cluster_counts,
        cluster_ndvi_avg= cluster_ndvi_avg,
        cluster_ndvi_std = cluster_ndvi_std,
        cluster_conf_avg = cluster_conf_avg,
        cluster_pct = cluster_pct,
        avg_confidence  = avg_conf,
        ndvi_diff  =  ndvi_diff,
        verdict  = verdict,
        silhouette  = sil,
        db_score   = db,
    )