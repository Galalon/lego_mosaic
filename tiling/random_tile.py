import numpy as np
import matplotlib.pyplot as plt
from random import randint, choice
from tilemap import TileMap

# Assume the optimized TileMap class is already defined above
# Example tile sizes and colors
TILE_SIZES = [(1, 1), (1, 2), (2, 2), (2, 3)]
TILE_COLORS = [
    (255, 0, 0),    # Red
    (0, 255, 0),    # Green
    (0, 0, 255),    # Blue
    (255, 255, 0),  # Yellow
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Cyan
    (120, 120, 120) # Gray
]

def fill_canvas_randomly(tilemap, max_attempts=10000):
    attempts = 0
    while attempts < max_attempts:
        size = choice(TILE_SIZES)
        color = choice(TILE_COLORS)
        y = randint(0, tilemap.grid_size[0] - 1)
        x = randint(0, tilemap.grid_size[1] - 1)
        if tilemap.add_tile(size, color, (y, x)):
            continue
        attempts += 1

# Create a tilemap (e.g., 20x20 grid)
tilemap = TileMap(grid_size=(20, 20))
fill_canvas_randomly(tilemap)

# Render and show the result
img = tilemap.render(scale=20)
plt.imshow(img)
plt.axis('off')
plt.title(f"Rendered TileMap with {tilemap.count_tiles()} tiles")
plt.show()
