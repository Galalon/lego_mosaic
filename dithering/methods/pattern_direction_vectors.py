import numpy as np
from utils.consts import CHANNEL_NOISE_STD
from utils.pattern_utils import generate_pattern
import cv2


def pattern_direction_vectors_dither_core(img_cs, palette_cs, color_space='RGB', scale=1.0, pattern='bayer',
                                          pattern_params={'level': 3}):
    H, W, C = img_cs.shape
    pattern_mat = generate_pattern(shape=(H, W, 1), pattern_name=pattern, pattern_params=pattern_params)[:, :, 0]
    pattern_norm = (pattern_mat - pattern_mat.min()) / (pattern_mat.max() - pattern_mat.min())
    if pattern == 'blue_noise':
        pattern_norm = cv2.equalizeHist((pattern_norm * 255).astype(np.uint8))
        pattern_norm = pattern_norm.astype(float) / 255

    channel_weights = np.array(CHANNEL_NOISE_STD.get(color_space.upper(), [1.0] * C), dtype=np.float32)
    out = img_cs.copy()

    if color_space.upper() == 'RGB':
        R = np.array([1.0, 0.0, 0.0])
        G = np.array([0.0, 1.0, 0.0])
        B = np.array([0.0, 0.0, 1.0])
        if pattern == 'bayer':
            dir_tile = np.array([[G, B], [R, G]])
            dir_tile = np.tile(dir_tile, (H // 2, W // 2, 1))[:H, :W]
            channel_weights[1] /= 2  # reduce green influence because it is shown twice as often
        else:
            dir_vectors = np.stack([R, G, B])
            index_tile = pattern_norm * 255 % 3
            dir_tile = dir_vectors[(index_tile - 1).astype(int)] 
        direction_vectors = dir_tile * channel_weights * pattern_norm[:, :, np.newaxis]
        out = out + (direction_vectors * scale)
        out = out / out.mean() *img_cs.mean() # preserve mean brightness


    elif color_space.upper() == 'LAB':
        theta = pattern_norm * 2 * np.pi
        direction_vectors = np.zeros((H, W, 3), dtype=np.float32)
        direction_vectors[..., 1] = np.cos(theta) * 0.5
        direction_vectors[..., 2] = np.sin(theta) * 0.5
        direction_vectors *= channel_weights
        out = out + (direction_vectors * scale)

    else:
        raise ValueError(f"Color space {color_space} not supported in direction_vectors_core")
    

    return out
