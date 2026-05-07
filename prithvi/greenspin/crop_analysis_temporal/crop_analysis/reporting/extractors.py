"""
reporting/extractors.py

Extracts per-phenotype NDVI mean and std trajectories from raw chip arrays for
use in report thumbnail sparklines. No matplotlib dependency - pure data extraction.
extract_thumbnail_data(bundle) is the main convenience wrapper.


"""
from __future__ import annotations
import numpy as np


def extract_ndvi_trajectory(
    chip_temporal: np.ndarray,  
    cloud_masks: np.ndarray,   
    field_mask: np.ndarray,   
) -> np.ndarray:
   
    T = chip_temporal.shape[0]
    field_flat = field_mask.ravel() == 1
    out = np.full(T, np.nan, dtype=np.float32)

    for t in range(T):
        valid = (cloud_masks[t].ravel() == 0) & field_flat
        if not valid.any():
            continue
        B4 = chip_temporal[t, 2].ravel()[valid].astype(np.float32)
        B8 = chip_temporal[t, 3].ravel()[valid].astype(np.float32)
        out[t] = float(np.mean((B8 - B4) / (B8 + B4 + 1e-6)))

    return out


def extract_per_phenotype_trajectories(
    chip_temporal: np.ndarray,   
    cloud_masks:np.ndarray,   
    field_mask:  np.ndarray,   
    cluster_map:np.ndarray,   
    n_phenotypes:  int,
) -> list[list[float]]:
    
    T  = chip_temporal.shape[0]
    B4 = chip_temporal[:, 2, :, :].astype(np.float32)   
    B8 = chip_temporal[:, 3, :, :].astype(np.float32)
    ndvi_t = (B8 - B4) / (B8 + B4 + 1e-6)               

    trajectories: list[list[float]] = []
    for i in range(n_phenotypes):
        pheno_mask = (cluster_map == i)                  
        traj: list[float] = []
        for t in range(T):
            valid = pheno_mask & (cloud_masks[t] == 0)
            if valid.any():
                traj.append(float(np.mean(ndvi_t[t][valid])))
            else:
                traj.append(float('nan'))
        trajectories.append(traj)

    return trajectories


def extract_per_phenotype_trajectories_with_std(
    chip_temporal: np.ndarray,   
    cloud_masks: np.ndarray,   
    field_mask: np.ndarray,   
    cluster_map: np.ndarray,   
    n_phenotypes: int,
) -> tuple[list[list[float]], list[list[float]]]:
    
    
    T = chip_temporal.shape[0]
    B4 = chip_temporal[:, 2, :, :].astype(np.float32)   
    B8 = chip_temporal[:, 3, :, :].astype(np.float32)
    ndvi_t = (B8 - B4) / (B8 + B4 + 1e-6)               

    traj_mean: list[list[float]] = []
    traj_std: list[list[float]] = []
    
    for i in range(n_phenotypes):
        pheno_mask = (cluster_map == i)                  
        mean_vals: list[float] = []
        std_vals: list[float] = []
        
        for t in range(T):
            valid = pheno_mask & (cloud_masks[t] == 0)
            if valid.any():
                vals = ndvi_t[t][valid]
                mean_vals.append(float(np.mean(vals)))
                std_vals.append(float(np.std(vals)))
            else:
                mean_vals.append(float('nan'))
                std_vals.append(float('nan'))
        
        traj_mean.append(mean_vals)
        traj_std.append(std_vals)

    return traj_mean, traj_std


def _make_nir_false(chip: np.ndarray) -> np.ndarray:
    
    def _norm(arr: np.ndarray) -> np.ndarray:
        lo, hi = np.percentile(arr, 2), np.percentile(arr, 98)
        return np.clip((arr - lo) / (hi - lo + 1e-9), 0.0, 1.0)
    return np.stack([_norm(chip[3]), _norm(chip[2]), _norm(chip[1])], axis=-1)


def extract_thumbnail_data(bundle: dict) -> dict:
    
    result  = bundle['result']
    chip_disp  = bundle.get('chip_composite', bundle['chip_temporal'][0])
    nir_false = _make_nir_false(chip_disp)

    trajectories = extract_per_phenotype_trajectories(
        chip_temporal = bundle['chip_temporal'],
        cloud_masks = bundle['cloud_masks'],
        field_mask = bundle['mask_224_clean'],
        cluster_map = result.pixel_cluster_map,
        n_phenotypes= result.optimal_n,
    )

    return {
        'rgb': nir_false,
        'cluster_map':  result.pixel_cluster_map,
        'mask': bundle['mask_224_clean'],
        'ndvi_trajectory': trajectories,
        'trajectory_dates': bundle.get('used_dates', []),
    }