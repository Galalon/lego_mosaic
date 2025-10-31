import numpy as np
from utils.consts import CHANNEL_NOISE_STD
from utils.pattern_utils import generate_blue_noise_fft, generate_bayer_matrices, tile_matrix

def noise_dither_core(img_cs, palette_cs,color_space='RGB', noise_scalar=0.1):
    channel_scales = np.array(CHANNEL_NOISE_STD[color_space.upper()])
    H, W, C = img_cs.shape

    noise = np.stack([
        generate_blue_noise_fft(max(H, W), low_freq=50, high_freq=250)[:H,:W]
        for _ in range(C)
    ], axis=2) * (channel_scales * noise_scalar)

    return img_cs + noise

