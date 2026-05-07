"""
encoder.py

Interfaces with the Prithvi-EO Vision Transformer encoder. Extracts (T, 196, 1024)
patch tokens, computes temporal embedding statistics (mean/std/range over time),
and upsamples patch embeddings to pixel resolution for clustering.


"""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

from config import (
    TEMPORAL_REPEATS, PATCH_GRID, CHIP_SIZE, PATCH_MASK_THRESHOLD, INFRA_NDVI_THRESH,RANDOM_SEED
)

def build_input_tensor(temporal_chips: np.ndarray, device: torch.device) -> torch.Tensor:
 
    t_tensor = torch.from_numpy(temporal_chips).float()
    
    t_tensor = t_tensor.permute(1, 0, 2, 3).unsqueeze(0)
    return t_tensor.to(device)


def extract_patch_tokens(model, input_tensor: torch.Tensor, n_dates: int | None = None) -> np.ndarray:
    with torch.no_grad():
        last_block = model.forward_features(input_tensor)[-1]
        patch_tokens = last_block[:, 1:, :]
        t= n_dates if n_dates is not None else TEMPORAL_REPEATS
        patch_tokens = patch_tokens.reshape(
            1, t, PATCH_GRID * PATCH_GRID, -1
        )
    return patch_tokens.squeeze(0).cpu().numpy()   


def average_patch_tokens(patch_tokens_temporal: np.ndarray) -> np.ndarray:
   
    return patch_tokens_temporal.mean(axis=0)


def extract_temporal_emb_stats(
    patch_tokens_temporal: np.ndarray,
    n_pca_dims: int = 16,
) -> np.ndarray:
    
    T, P, D = patch_tokens_temporal.shape

    
    n_comp = min(n_pca_dims, D - 1, T * P - 1, T - 1)
    n_comp = max(n_comp, 1)

    flat = patch_tokens_temporal.reshape(T * P, D)
    pca_emb  = PCA(n_components=n_comp, random_state=RANDOM_SEED)
    flat_reduced = pca_emb.fit_transform(flat)              
    reduced = flat_reduced.reshape(T, P, n_comp)       

    mean_t = reduced.mean(axis=0)                         
    std_t = reduced.std(axis=0)
    range_t = reduced.max(axis=0) - reduced.min(axis=0)

    return np.concatenate([mean_t, std_t, range_t], axis=1) 


def upsample_embeddings(embeddings: np.ndarray) -> np.ndarray:
    
    embed_dim = embeddings.shape[1]
    emb_spatial = torch.tensor(
        embeddings.T.reshape(1, embed_dim, PATCH_GRID, PATCH_GRID),
        dtype=torch.float32,
    )
    emb_up = F.interpolate(
        emb_spatial,
        size=(CHIP_SIZE, CHIP_SIZE),
        mode='bilinear',
        align_corners=False,
    )
    return emb_up.squeeze(0).reshape(embed_dim, -1).T.numpy()


def make_patch_mask(mask_224: np.ndarray, threshold: float = PATCH_MASK_THRESHOLD) -> np.ndarray:
    
    """A patch is True when >= threshold fraction of its pixels are field.

    """
    patch_size = CHIP_SIZE // PATCH_GRID
    patch_mask= np.zeros((PATCH_GRID, PATCH_GRID), dtype=bool)

    for r in range(PATCH_GRID):
        for c in range(PATCH_GRID):
            tile = mask_224[
                r * patch_size : (r + 1) * patch_size,
                c * patch_size : (c + 1) * patch_size,
            ]
            patch_mask[r, c] = tile.mean() >= threshold

    return patch_mask


def mask_patch_embeddings(
    embeddings: np.ndarray,        
    mask_224: np.ndarray,     
    chip: np.ndarray | None = None,  
    threshold: float = PATCH_MASK_THRESHOLD,
    infra_ndvi_thresh: float = INFRA_NDVI_THRESH,
) -> np.ndarray:
    
    patch_mask = make_patch_mask(mask_224, threshold=threshold)   

    if chip is not None:
        B4  = chip[2].astype(np.float32)
        B8   = chip[3].astype(np.float32)
        ndvi_chip = (B8 - B4) / (B8 + B4 + 1e-6)
        patch_size = CHIP_SIZE // PATCH_GRID

        for r in range(PATCH_GRID):
            for c in range(PATCH_GRID):
                if not patch_mask[r, c]:
                    continue
                rs = slice(r * patch_size, (r + 1) * patch_size)
                cs = slice(c * patch_size, (c + 1) * patch_size)
                field_px_in_patch = mask_224[rs, cs]
                ndvi_in_patch = ndvi_chip[rs, cs]
                
                field_ndvi_vals= ndvi_in_patch[field_px_in_patch == 1]
                if field_ndvi_vals.size > 0 and field_ndvi_vals.mean() < infra_ndvi_thresh:
                    patch_mask[r, c] = False   

    flat_mask= patch_mask.ravel()
    masked= embeddings.copy()
    masked[~flat_mask] = 0.0
    return masked                           


def make_feature_map(
    embeddings: np.ndarray,
    mask_224: np.ndarray | None = None,
    mode: str = 'l2',           
) -> np.ndarray:
    
    if embeddings.ndim == 2:
        if mode == 'l2':
            raw_vals = np.linalg.norm(embeddings, axis=1)
        elif mode == 'pca':
            raw_vals = PCA(n_components=1).fit_transform(embeddings).reshape(-1) 
            raw_vals= np.abs(raw_vals)
        else:
            raw_vals = np.linalg.norm(embeddings, axis=1)
    else:
        raw_vals= embeddings

    raw_map =raw_vals.reshape(PATCH_GRID, PATCH_GRID)

    if mask_224 is None:
        return raw_map

    patch_mask= make_patch_mask(mask_224, threshold=PATCH_MASK_THRESHOLD)
    masked_map = raw_map.astype(float)
    masked_map[~patch_mask] = np.nan

    if np.all(np.isnan(masked_map)):
        return masked_map

    lo= np.nanmin(masked_map)
    hi= np.nanmax(masked_map)
    masked_map =(masked_map - lo)/(hi - lo + 1e-9)

    return masked_map