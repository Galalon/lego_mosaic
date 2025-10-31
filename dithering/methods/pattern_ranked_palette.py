import numpy as np
from utils.consts import CHANNEL_NOISE_STD
from utils.pattern_utils import generate_pattern


def pattern_ranked_palette_dither_core(img_cs, palette_cs, color_space='LAB',scale=1.0, pattern='bayer',
                                       pattern_params={'level': 3}):
    """
    Core function for ranked palette dithering.
    """
    H, W, _ = img_cs.shape
    img_flat = img_cs.reshape(-1, 3)
    P = palette_cs.shape[0]

    # Get noise std weights for color space and invert them for weighting
    weights = np.array(CHANNEL_NOISE_STD.get(color_space.upper(), [1.0, 1.0, 1.0]), dtype=np.float32)
    inv_weights = 1.0 / weights

    # Compute weighted distances between image pixels and palette
    diff = img_flat[:, None, :] - palette_cs[None, :, :]  # (H*W, P, 3)
    weighted_diff = diff * inv_weights  # (H*W, P, 3)
    dists = np.linalg.norm(weighted_diff, axis=2)  # (H*W, P)

    map_vals = generate_pattern(pattern, (H, W, 1), pattern_params=pattern_params)[:, :, 0]

    # Apply scale to map values
    map_vals = np.clip(map_vals, 0, scale)

    # Ranked index selection
    rank_indices = np.clip((map_vals * (P - 1)).astype(int), 0, P - 1)
    sorted_idx = np.argsort(dists, axis=1)  # (H*W, P)
    chosen_idx = sorted_idx[np.arange(H * W), rank_indices.flatten()]

    # Return reshaped palette image in color space (as RGB mapping will be handled after)
    chosen_palette_cs = palette_cs[chosen_idx]
    return chosen_palette_cs.reshape(H, W, 3)
