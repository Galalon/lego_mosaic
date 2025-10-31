import numpy as np
from hilbertcurve.hilbertcurve import HilbertCurve
from utils.color_utils import clamp_img
from utils.consts import CHANNEL_NOISE_STD


def error_diffusion_hilbert_dither_core(img_cs, palette_cs, color_space,scale=1.0, decay=1.0):
    """
    Hilbert curve error diffusion core function compatible with the pipeline.

    Args:
        img_cs: Image in the specified color space (H, W, 3) as float32
        palette_cs: Palette in the same color space (P, 3)
        color_space: 'RGB', 'LAB', etc.
        order: Order of the Hilbert curve
        decay: Error diffusion decay factor
        scale: Error strength multiplier

    Returns:
        Modified image in color space (H, W, 3)
    """
    H, W, _ = img_cs.shape
    order = int(np.ceil(np.log2(max(H, W))))

    hilbert_max = 2 ** order
    hilbert = HilbertCurve(p=order, n=2)

    coords = [hilbert.point_from_distance(i) for i in range(hilbert_max ** 2)]
    coords = [(x, y) for x, y in coords if x < W and y < H][:H * W]
    flat_indices = [(y, x) for x, y in coords]

    buf_flat = np.array([img_cs[y, x] for y, x in flat_indices], dtype=np.float32)

    weights = 1.0 / np.array(CHANNEL_NOISE_STD.get(color_space.upper(), [1.0, 1.0, 1.0]), dtype=np.float32)

    max_lookahead = 4
    decay_factors = decay ** np.arange(1, max_lookahead + 1)
    decay_factors /= decay_factors.sum()

    for i in range(len(buf_flat)):
        old_pixel = buf_flat[i]
        diff = palette_cs - old_pixel
        dists = np.linalg.norm(diff * weights, axis=1)
        new_pixel = palette_cs[np.argmin(dists)]
        err = old_pixel - new_pixel
        buf_flat[i] = new_pixel

        for j, d in enumerate(decay_factors):
            tgt_idx = i + j + 1
            if tgt_idx < len(buf_flat):
                buf_flat[tgt_idx] += err * d * scale

    # Reconstruct full image buffer
    out_buf = np.zeros_like(img_cs)
    for (y, x), val in zip(flat_indices, buf_flat):
        out_buf[y, x] = val

    return clamp_img(out_buf, color_space)
