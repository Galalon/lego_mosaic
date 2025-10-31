import numpy as np
from collections import defaultdict

class TileMap:
    def __init__(self, grid_size):
        self.grid_size = grid_size  # (rows, cols)
        H, W = grid_size
        self.map = np.full((H, W, 3), fill_value=255, dtype=np.uint8)  # white RGB canvas
        self.occupancy = np.zeros((H, W), dtype=bool)  # True if occupied
        self.tiles = []  # List of (shape, color, top_left)

    def can_place(self, shape, top_left):
        h, w = shape
        y0, x0 = top_left
        if y0 + h > self.grid_size[0] or x0 + w > self.grid_size[1]:
            return False
        return not np.any(self.occupancy[y0:y0 + h, x0:x0 + w])

    def add_tile(self, shape, color, top_left):
        h, w = shape
        y0, x0 = top_left
        if not self.can_place(shape, top_left):
            return False
        self.map[y0:y0 + h, x0:x0 + w] = color
        self.occupancy[y0:y0 + h, x0:x0 + w] = True
        self.tiles.append((shape, color, top_left))
        return True

    def remove_tile(self, top_left):
        for i, (shape, color, pos) in enumerate(self.tiles):
            if pos == top_left:
                h, w = shape
                y0, x0 = pos
                self.map[y0:y0 + h, x0:x0 + w] = 255
                self.occupancy[y0:y0 + h, x0:x0 + w] = False
                self.tiles.pop(i)
                return True
        return False

    def count_tiles(self):
        return len(self.tiles)
    def count_tiles_by_shape_and_color(self, inv_dict=None):
        """
        Count how many times each (normalized_shape, color) pair appears in the tile list.
        Transposed shapes (e.g., (2,3) and (3,2)) are considered the same.

        Args:
            tile_list: List of tuples like ((h, w), color_array)

        Returns:
            Dictionary with keys (normalized_shape, color) and values = counts
        """
        counts = defaultdict(int)

        for shape, color, _ in self.tiles:
            normalized_shape = tuple(sorted(shape))
            color_key = tuple(color) if inv_dict is None else inv_dict[tuple(color)] # Convert np.array to hashable type
            counts[(normalized_shape, color_key)] += 1

        return dict(counts)

    def render(self, divider_color=(0, 0, 0), scale=10):
        """
        Render the tilemap with 1-pixel-wide borders between tiles.

        Args:
            divider_color: RGB tuple for the grid lines
            scale: Zoom factor

        Returns:
            A rendered RGB image (H*scale, W*scale, 3)
        """
        H, W = self.grid_size
        img = np.ones((H * scale, W * scale, 3), dtype=np.uint8) * 255

        for (shape, color, top_left) in self.tiles:
            h, w = shape
            y0, x0 = top_left

            for dy in range(h):
                for dx in range(w):
                    y, x = y0 + dy, x0 + dx
                    sy, sx = y * scale, x * scale
                    img[sy:sy + scale, sx:sx + scale] = color

                    # Borders
                    if dy == 0:
                        img[sy, sx:sx + scale] = divider_color  # top
                    if dy == h - 1:
                        img[sy + scale - 1, sx:sx + scale] = divider_color  # bottom
                    if dx == 0:
                        img[sy:sy + scale, sx] = divider_color  # left
                    if dx == w - 1:
                        img[sy:sy + scale, sx + scale - 1] = divider_color  # right

        return img
