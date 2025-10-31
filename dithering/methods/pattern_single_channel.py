import numpy as np
from utils.consts import CHANNEL_NOISE_STD
from utils.pattern_utils import generate_pattern


def pattern_single_channel_dither_core(img_cs, palette_cs, color_space='LAB', scale=1.0, channel_name='L',
                                       channel_scales=None, pattern='bayer', pattern_params={'level': 3}):
    C = img_cs.shape[2]
    H, W = img_cs.shape[:2]

    # Get channel index
    channel_names = {
        "RGB": ["R", "G", "B"],
        "HSV": ["Hue", "Sat", "Val"],
        "LAB": ["L", "a", "b"]
    }
    try:
        ch_idx = channel_names[color_space.upper()].index(channel_name)
    except (KeyError, ValueError):
        raise ValueError(f"Channel name '{channel_name}' not found in color space '{color_space}'")

    if channel_scales is None:
        channel_scales = np.array(CHANNEL_NOISE_STD.get(color_space.upper(), [1.0] * C), dtype=np.float32)
    else:
        channel_scales = np.array(channel_scales, dtype=np.float32)

    # Direction vector in color space, scaled
    direction_vec = np.zeros(C, dtype=np.float32)
    direction_vec[ch_idx] = channel_scales[ch_idx]

    # Create Bayer matrix and tile to match image
    B = generate_pattern(pattern_name=pattern, shape=(H, W, 1), pattern_params=pattern_params)

    # Apply offset in named channel direction
    offset = scale * B * direction_vec[None, None, :]
    return img_cs + offset
