import numpy as np
from utils.consts import CHANNEL_NOISE_STD, KERNEL_DICT
from dithering.quantization import quantize_pixel


def error_diffusion_raster_dither_core(img_cs, palette_cs, color_space='LAB',scale = 1.0, kernel_name='floyd-steinberg'):


    assert color_space.upper() != 'HSV', 'HSV not supported for error diffusion'

    H, W, _ = img_cs.shape
    buf = img_cs.copy()

    weights = 1.0 / np.array(CHANNEL_NOISE_STD[color_space.upper()], dtype=np.float32)
    kernel = KERNEL_DICT[kernel_name]

    for y in range(H):
        for x in range(W):
            old_pixel = buf[y, x]
            new_pixel = quantize_pixel(old_pixel, palette_cs, weights)
            err = old_pixel - new_pixel
            buf[y, x] = new_pixel

            for dx, dy, k in kernel:
                nx, ny = x + int(dx), y + int(dy)
                if 0 <= nx < W and 0 <= ny < H:
                    buf[ny, nx] += err * k

    buf = scale* buf + (1-scale)* img_cs
    return buf
