import numpy as np
from utils.consts import CHANNEL_NOISE_STD
from utils.pattern_utils import generate_pattern


def pattern_multiple_channels_dither_core(img_cs, palette_cs, color_space='RGB', scale=1.0, pattern='bayer',
                                          pattern_params={'level': 3}):
    H, W, C = img_cs.shape
    channel_scales = np.array(CHANNEL_NOISE_STD.get(color_space.upper(), [1.0] * C), dtype=np.float32)

    pattern_array = generate_pattern(pattern_name=pattern, shape=img_cs.shape, pattern_params=pattern_params)
    if pattern == 'bayer':
        pattern_array = (pattern_array - pattern_array.min()) / (pattern_array.max() - pattern_array.min())
        pattern_array = pattern_array * 2 - 1  # Normalize to [-1, 1]
    out = img_cs.copy()

    out += scale * pattern_array * channel_scales

    return out
