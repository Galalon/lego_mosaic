import numpy as np


def quantize_pixel(pixel, palette, weights=None):
    diff = palette - pixel
    if weights is not None:
        diff = diff * weights
    dists = np.linalg.norm(diff, axis=1)
    return palette[np.argmin(dists)]


def quantize_image(img_rgb, palette_rgb, weights=None):
    """
    Quantize an image to the closest palette color per pixel, using optional per-channel weights.

    Args:
        img_rgb (H, W, 3): Image in RGB color space, float32.
        palette_rgb (N, 3): Palette colors in RGB, float32.
        weights (3,) or None: Optional per-channel weights (e.g. 1/std per channel).

    Returns:
        quantized_img (H, W, 3): uint8 image with quantized palette colors.
    """
    h, w, _ = img_rgb.shape
    img_flat = img_rgb.reshape(-1, 3)

    weights = np.asarray(weights if weights is not None else [1.0, 1.0, 1.0], dtype=np.float32)

    diff = (img_flat[:, None, :] - palette_rgb[None, :, :]) * weights
    distances = np.sum(np.abs(diff), axis=2)
    closest_indices = np.argmin(distances, axis=1)
    quantized_flat = palette_rgb[closest_indices]

    return quantized_flat.reshape((h, w, 3)).astype(np.uint8)
