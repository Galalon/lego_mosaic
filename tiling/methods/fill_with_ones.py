import numpy as np

def fill_with_ones_core(color_mask, tiles_with_shapes, tile_map, **kwargs):
    """
    Fills the tile map with 1x1 tiles wherever color_mask == 1.

    Args:
        color_mask: 2D array with 1s where a tile should be placed.
        tiles_with_shapes: List of (tile_shape, tile_color) tuples.
        tile_map: TileMap instance.
    """
    for tile_shape, tile_color in tiles_with_shapes:
        if tile_shape == (1, 1):
            for y, x in np.argwhere(color_mask == 1):
                tile_map.add_tile(tile_shape, tile_color, (y, x))
            return
    raise ValueError("tiles_with_shapes must contain a (1x1) tile shape.")
