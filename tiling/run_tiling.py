from dithering.quantization import quantize_image
from utils.tile_data import COLOR_PALETTES
from utils.color_utils import prepare_palette
from tiling.pipeline import tile_image
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    input_path = r"C:\Users\sgala\Documents\Lenna_(test_image).png"  # Update path
    img = Image.open(input_path).convert("RGB")
    img_array = np.array(img)
    quantized = quantize_image(img_array,prepare_palette(COLOR_PALETTES['lego'][0]))
    tile_specs = [(1, 1), (1, 2), (1, 3), (1, 4), (1, 6), (1, 8), (2, 2), (2, 3), (2, 4), (2, 6)]

    tilemap = tile_image(quantized, prepare_palette(COLOR_PALETTES['lego'])[0], tile_specs, "greedy_fill", use_isolated=True)

    img = tilemap.render(scale=20)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Rendered TileMap with {tilemap.count_tiles()} tiles")
    plt.show()