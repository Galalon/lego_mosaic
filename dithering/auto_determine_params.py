import numpy as np

from skimage.color import rgb2lab, rgb2hsv
from utils.color_utils import prepare_palette

METHOD_PARAMS = {
    "pattern_direction_vectors": {"supported_color_spaces": ['RGB'], "post_processing_func": lambda x: x / 3, },
    "pattern_multiple_channels": {"supported_color_spaces": ['RGB', 'HSV', 'LAB'],
                                  "post_processing_func": lambda x: x / 3},
    "pattern_ranked_palette": {"supported_color_spaces": ['RGB'], "post_processing_func": lambda x: x},
    "pattern_single_channel": {"supported_color_spaces": ['RGB', 'HSV', 'LAB'], "post_processing_func": lambda x: x},
    "error_diffusion_raster": {"supported_color_spaces": ['RGB', 'LAB'], "post_processing_func": lambda x: 1},
    "error_diffusion_hilbert": {"supported_color_spaces": ['RGB', 'LAB'], "post_processing_func": lambda x: 1},
}


def compute_palette_statistics(pallete_hex):
    def normalize_lab(lab):
        lab_norm = lab.copy()
        lab_norm[:, 0] /= 100.0  # L ∈ [0, 100]
        lab_norm[:, 1] = (lab_norm[:, 1] + 128) / 255.0  # a ∈ [-128, 127]
        lab_norm[:, 2] = (lab_norm[:, 2] + 128) / 255.0  # b ∈ [-128, 127]
        return lab_norm

    def compute_pairwise_distances(colors):
        N = colors.shape[0]
        diffs = colors[:, None, :] - colors[None, :, :]  # (N, N, C)
        dists = np.linalg.norm(diffs, axis=2)
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)  # Only upper triangle
        return dists[mask]

    result = {}

    stats = {}
    rgb, _ = prepare_palette(pallete_hex)
    lab = normalize_lab(rgb2lab(rgb))
    hsv = rgb2hsv(rgb)

    color_spaces = {
        'RGB': rgb,
        'LAB': lab,
        'HSV': hsv
    }

    for cs_name, data in color_spaces.items():
        dists = compute_pairwise_distances(data)
        stats[cs_name] = {
            'avg': float(np.mean(dists)),
            'std': float(np.std(dists))
        }

    return stats


def determine_scale_and_color_space(method_name, pallete_hex):
    stats = compute_palette_statistics(pallete_hex)
    supported_cs, post_process = METHOD_PARAMS[method_name]['supported_color_spaces'], METHOD_PARAMS[method_name][
        'post_processing_func']
    min_std = np.inf
    best_cs = None
    threshold = None
    for cs in supported_cs:
        std = stats[cs]['std']
        if std < min_std:
            best_cs = cs
            min_std = std
            threshold = stats[cs]['avg']
    threshold = post_process(threshold)
    return best_cs, threshold
