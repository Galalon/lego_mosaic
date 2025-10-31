import numpy as np
from utils.color_utils import rgb_to_cs, cs_to_rgb, clamp_img
from dithering.quantization import quantize_image

import importlib
import inspect
from pathlib import Path
from dithering.methods.pattern_direction_vectors import pattern_direction_vectors_dither_core
from dithering.methods.pattern_multiple_channels import pattern_multiple_channels_dither_core
from dithering.methods.pattern_ranked_palette import pattern_ranked_palette_dither_core
from dithering.methods.pattern_single_channel import pattern_single_channel_dither_core
from dithering.methods.error_diffusion_raster import error_diffusion_raster_dither_core
from dithering.methods.error_diffusion_hilbert import error_diffusion_hilbert_dither_core

DITHERING_METHODS = {
    "pattern_direction_vectors": {
        "func": pattern_direction_vectors_dither_core,
        "color_space": {'type': 'list', 'values': ['RGB', 'LAB']},
        "scale": {'type': 'range', 'values': [0, 1]},
        'pattern': {'type': 'pattern', 'values': ['bayer', 'blue_noise']}
    },
    "pattern_multiple_channels": {
        'func': pattern_multiple_channels_dither_core,
        "color_space": {'type': 'list', 'values': ['RGB', 'LAB', 'HSV']},
        "scale": {'type': 'range', 'values': [0, 1]},
        'pattern': {'type': 'pattern', 'values': ['bayer', 'blue_noise']}
    },
    "pattern_ranked_palette": {
        'func': pattern_ranked_palette_dither_core,
        "color_space": {'type': 'list', 'values': ['RGB', 'LAB', 'HSV']},
        "scale": {'type': 'range', 'values': [0, 1]},
        'pattern': {'type': 'pattern', 'values': ['bayer', 'blue_noise']}
    },
    "pattern_single_channel": {
        'func': pattern_single_channel_dither_core,
        "color_space": {'type': 'list', 'values': ['RGB', 'LAB', 'HSV']},
        "scale": {'type': 'range', 'values': [0, 1]},
        'pattern': {'type': 'pattern', 'values': ['bayer', 'blue_noise']},
        'channel_name': {'type': 'list', 'values': ["R", "G", "B", "Hue", "Sat", "Val", "L", "a", "b"]}
    },
    "error_diffusion_raster": {
        'func': error_diffusion_raster_dither_core,
        "color_space": {'type': 'list', 'values': ['RGB', 'LAB']},
        "scale": {'type': 'range', 'values': [0, 1]},
        'kernel_name': {'type': 'list',
                        'values': ['floyd-steinberg', 'jarvis-judice-ninke', 'sierra', 'atkinson', 'stucki']},
    },
    "error_diffusion_hilbert": {
        'func': error_diffusion_hilbert_dither_core,
        "color_space": {'type': 'list', 'values': ['RGB', 'LAB']},
        "scale": {'type': 'range', 'values': [0, 1]},
        'decay': {'type': 'range', 'values': [0, 1]},
    },

}


def dither(img_rgb, rgb_palette, color_space, method_name, scale, **kwargs):
    core_func = DITHERING_METHODS.get(method_name)['func']
    if core_func is None:
        raise ValueError(f"Unknown dithering method: {method_name}")

    return run_dithering_pipeline(
        img_rgb,
        rgb_palette,
        color_space,
        scale,
        dither_core_func=core_func,
        **kwargs
    )


def run_dithering_pipeline(
        img_rgb,
        rgb_palette,
        color_space,
        scale,
        dither_core_func,
        clamp=True,
        quantize_weights=None,
        **core_kwargs
):
    """
    Runs the general dithering pipeline:
      - Converts image & palette to the chosen color space
      - Calls `dither_core_func` which performs the core dithering logic on image in color space
      - Clamps the result (optional)
      - Converts back to RGB
      - Quantizes with optional weights
    """
    img_rgb = img_rgb.astype(np.float32)
    img_cs = rgb_to_cs(img_rgb, color_space)

    palette_cs = rgb_to_cs(rgb_palette[None, ...], color_space).reshape(-1, 3)

    # Run the core dithering method on the color space image (float)
    out_cs = dither_core_func(img_cs, palette_cs, color_space,scale, **core_kwargs)

    if clamp:
        out_cs = clamp_img(out_cs, color_space)

    img_rgb_out = cs_to_rgb(out_cs, color_space)
    quantized = quantize_image(img_rgb_out, rgb_palette, weights=quantize_weights)

    return quantized
