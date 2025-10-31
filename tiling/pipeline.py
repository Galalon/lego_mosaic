import numpy as np
import cv2
from tqdm import tqdm
from tiling.tilemap import TileMap
from tiling.methods.fill_with_ones import fill_with_ones_core
from tiling.methods.greedy_fill import greedy_fill_core
from collections import defaultdict


TILING_METHODS = {
    "fill_with_ones": {
        "func": fill_with_ones_core,
        "params": {}  # Can be expanded later
    },
    "greedy_fill": {
        "func": greedy_fill_core,
        "params": {}
    },
}






def generate_tile_shapes_with_color(tile_sizes, color):
    """
    Generates a list of tile shapes with the specified color.
    Includes both the base and transposed versions if the tile is not square.

    Args:
        tile_sizes: List of (height, width) tuples.
        color: RGB tuple.

    Returns:
        List of (tile_shape, tile_color) tuples.
    """
    tile_list = []
    for h, w in tile_sizes:
        # base = np.ones((h, w), dtype=int)
        tile_list.append(((h,w), color))
        if h != w:
            tile_list.append(((w,h), color))
    return tile_list


def find_isolated_tiles(color_mask, tile_shape):
    kernel = np.ones(tile_shape)
    tile_kernel = np.zeros((tile_shape[0] + 2, tile_shape[1] + 2))
    tile_kernel[1:-1, 1:-1] = kernel

    tile_kernel_d = cv2.dilate(tile_kernel.astype(np.uint8), np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8))
    tile_kernel[tile_kernel != 1] = -tile_kernel_d[tile_kernel != 1].astype(float)

    match_mask = cv2.filter2D(color_mask.astype(float), -1, tile_kernel, anchor=(1, 1), borderType=cv2.BORDER_CONSTANT)
    isolated_tiles_mask = match_mask == kernel.sum()


    anchor = (tile_shape[1] - 1, tile_shape[0] - 1)
    isolated_tiles_mask_ = cv2.dilate(isolated_tiles_mask.astype(np.uint8), kernel, anchor=anchor, iterations=1)

    return isolated_tiles_mask, isolated_tiles_mask_


def tile_image(quantized_img, palette_rgb, tile_specs, method_name, use_isolated=True, **method_kwargs):
    H, W = quantized_img.shape[:2]
    tile_map = TileMap((H, W))
    if len(tile_specs) == 1 and tile_specs[0] == (1,1):
        method_name = "fill_with_ones"
        use_isolated = False

    assert (1,1) in tile_specs, 'tiling must have 1x1 tile to assure tiling'


    tile_method_entry = TILING_METHODS.get(method_name)
    if tile_method_entry is None:
        raise ValueError(f"Unknown tiling method: {method_name}")

    tile_func = tile_method_entry['func']

    for c in tqdm(palette_rgb):
        color_mask = (quantized_img == c).all(axis=2).astype(float)
        if color_mask.sum() == 0:
            continue

        tiles_with_shapes = generate_tile_shapes_with_color(tile_specs, c)

        if use_isolated:
            for tile_shape, tile_color in tiles_with_shapes:
                iso_mask, iso_mask_dilated = find_isolated_tiles(color_mask, tile_shape)
                color_mask -= iso_mask_dilated.astype(float)
                for idx in np.argwhere(iso_mask):
                    tile_map.add_tile(tile_shape, tile_color, tuple(idx))

        if color_mask.sum() > 0:
            tile_func(color_mask, tiles_with_shapes, tile_map, **method_kwargs)

    return tile_map
