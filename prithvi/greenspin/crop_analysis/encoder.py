"""
encoder.py

Contains functions to build the Prithvi input tensor, extract patch tokens, upsample to pixel-level embeddings, and create a PCA-based feature map for visualisation.
"""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

from config import TEMPORAL_REPEATS, PATCH_GRID, CHIP_SIZE


def build_input_tensor(chip: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Convert raw chip (C, H, W) to Prithvi input tensor of shape (1, C, T, H, W). 
    Repeats the chip across T time steps and moves to device.    
    """
    return (
        torch.from_numpy(chip).float()
        .unsqueeze(0)           
        .unsqueeze(2)           
        .repeat(1, 1, TEMPORAL_REPEATS, 1, 1)  
        .to(device)
    )


def extract_patch_tokens(
    model,
    input_tensor: torch.Tensor,
) -> np.ndarray:
    """
    Extract patch tokens from the Prithvi model. Returns a numpy array. 
    """
    with torch.no_grad():
        last_block  = model.forward_features(input_tensor)[-1]
        patch_tokens = last_block[:, 1:, :]                                # drop CLS
        patch_tokens = (
            patch_tokens
            .reshape(1, TEMPORAL_REPEATS, PATCH_GRID * PATCH_GRID, -1)
            .mean(dim=1)                                                    # avg over time
        )
    return patch_tokens.squeeze(0).cpu().numpy()   # (196, 1024)


def upsample_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """
    Upsample patch embeddings to pixel-level using bilinear interpolation.
    Input shape: (196, D), Output shape: (H*W, D).
    """
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


def make_patch_mask(mask_224: np.ndarray, threshold: float = 0.05) -> np.ndarray:
    """
    A patch is considered 'field' if >= threshold fraction of its pixels are field.
    a patch needs at least 30% field coverage to be included
    
    """
    patch_size = CHIP_SIZE // PATCH_GRID      # 16 pixels per patch side
    patch_mask = np.zeros((PATCH_GRID, PATCH_GRID), dtype=bool)

    for r in range(PATCH_GRID):
        for c in range(PATCH_GRID):
            tile = mask_224[
                r * patch_size : (r + 1) * patch_size,
                c * patch_size : (c + 1) * patch_size,
            ]
            patch_mask[r, c] = tile.mean() >= threshold

    return patch_mask


def mask_patch_embeddings(
    embeddings: np.ndarray,   # (196, D)
    mask_224:  np.ndarray,   # (H, W)  binary float32
    threshold: float = 0.05,
) -> np.ndarray:
    """
    Zero-out patch embeddings whose footprint lies outside the field boundary.
    
    """
    patch_mask = make_patch_mask(mask_224, threshold=threshold)   # (14, 14) bool
    flat_mask  = patch_mask.ravel()                               # (196,)
    masked     = embeddings.copy()
    masked[~flat_mask] = 0.0
    return masked                                                 # (196, D)


def make_feature_map(
    embeddings: np.ndarray,
    mask_224: np.ndarray | None = None,
    mode: str = 'l2',           # l2 or pca
) -> np.ndarray:
    """
    mode=l2  -> L2 norm of each patch embedding  (better for small fields)
    mode=pca -> first PCA component               (better when many patches)
    """
    if embeddings.ndim == 2:
        if mode == 'l2':
            raw_vals = np.linalg.norm(embeddings, axis=1)
        elif mode == 'pca':
            raw_vals = PCA(n_components=1).fit_transform(embeddings).reshape(-1)
            raw_vals = np.abs(raw_vals)
    else:
        raw_vals = embeddings

    raw_map = raw_vals.reshape(PATCH_GRID, PATCH_GRID)

    if mask_224 is None:
        return raw_map

    patch_mask = make_patch_mask(mask_224)
    masked_map = raw_map.astype(float)
    masked_map[~patch_mask] = np.nan

    if np.all(np.isnan(masked_map)):
        return masked_map

    lo = np.nanmin(masked_map)
    hi = np.nanmax(masked_map)
    masked_map = (masked_map - lo) / (hi - lo + 1e-9)

    return masked_map