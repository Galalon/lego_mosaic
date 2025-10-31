import numpy as np
from utils.color_utils import rgb_to_cs
from utils.pattern_utils import create_bayer_matrix

def bayer_dither_direction_rgb_guided_by_hsv_dither_core(img_cs, palette_cs, level=3, scale=1.0, **kwargs):
    # img_cs is actually RGB even though pipeline thinks it's color-space-transformed
    img_rgb = img_cs.copy()
    H, W, _ = img_rgb.shape
    img_hsv = rgb_to_cs(img_rgb, 'HSV')

    B = create_bayer_matrix(level)
    B_norm = (B + 1) / 2
    N = B.shape[0]

    alpha = 2 * scale

    for y in range(H):
        for x in range(W):
            V = img_hsv[y, x, 2]
            beta = B_norm[y % N, x % N]

            if V < 0.3:
                d = np.array([-0.1, -0.3, 0.0])
            elif V > 0.7:
                d = np.array([0.1, 0.2, 0.0])
            else:
                d = np.zeros(3)

            perturb = beta * alpha * d * 255
            img_rgb[y, x] = np.clip(img_rgb[y, x] + perturb, 0, 255)

    return img_rgb
