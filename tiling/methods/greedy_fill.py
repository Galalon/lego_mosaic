import numpy as np
import scipy.signal

def greedy_fill_core(color_mask, tiles_with_shapes, tile_map, **kwargs):
    """
    Greedily tiles a binary image (color_mask) using provided tile shapes (as tuples).

    Args:
        color_mask (np.ndarray): 2D binary array (0s and 1s) to be tiled.
        tiles_with_shapes (list): List of ((h, w), tile_color) tuples.
        tile_map (TileMap): An object to record placed tiles via its add_tile method.
    """
    if not isinstance(color_mask, np.ndarray) or color_mask.ndim != 2:
        raise ValueError("color_mask must be a 2D NumPy array.")

    current_mask = color_mask.astype(bool).copy()

    while np.any(current_mask):
        best_approx_density = np.inf
        best_tile = None

        for tile_shape, tile_color in tiles_with_shapes:
            h, w = tile_shape
            tile_mask = np.ones((h, w), dtype=int)

            flipped_tile = np.flipud(np.fliplr(tile_mask)).astype(float)
            convolution_result = scipy.signal.convolve2d(current_mask.astype(float), flipped_tile, mode='same')
            approx_density = (convolution_result == tile_mask.sum()).sum() / tile_mask.sum()

            if 0 < approx_density < best_approx_density:
                best_approx_density = approx_density
                best_tile = (tile_mask, tile_color)

        if best_tile is None:
            # No suitable tile found, break loop to avoid infinite loop
            break

        tile_mask, tile_color = best_tile
        tile_area = tile_mask.sum()
        tile_h, tile_w = tile_mask.shape

        # Find valid placements using convolution (mode='valid')
        flipped_tile = np.flipud(np.fliplr(tile_mask)).astype(float)
        convolution_result = scipy.signal.convolve2d(current_mask.astype(float), flipped_tile, mode='valid')
        valid_placement_origins = np.argwhere(convolution_result == tile_area)

        if len(valid_placement_origins) == 0:
            # No valid placement for best tile, remove from candidates and continue
            tiles_with_shapes = [t for t in tiles_with_shapes if (np.ones(t[0], dtype=int) != tile_mask).any()]
            continue

        tiles_to_add_this_round = []
        temp_round_coverage = np.zeros_like(current_mask, dtype=bool)

        for r, c in valid_placement_origins:
            temp_slice = temp_round_coverage[r : r + tile_h, c : c + tile_w]
            # Make sure shapes align if partial slices (at edges)
            tile_slice = tile_mask[0:temp_slice.shape[0], 0:temp_slice.shape[1]]

            if np.any(tile_slice & temp_slice):
                continue

            tiles_to_add_this_round.append((r, c))
            temp_round_coverage[r : r + tile_h, c : c + tile_w][tile_slice == 1] = True

        for r, c in tiles_to_add_this_round:
            tile_map.add_tile(tile_mask.shape, tile_color, (r, c))
            current_mask[r : r + tile_h, c : c + tile_w][tile_mask == 1] = False
