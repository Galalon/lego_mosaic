import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from utils.color_utils import prepare_palette
from dithering.pipeline import dither
from tiling.pipeline import tile_image
from utils.tile_data import COLOR_PALETTES, TILE_PALLETES
from dithering.auto_determine_params import determine_scale_and_color_space


def generate_lego_mosaic(
        image_path,
        palette_name,
        resize_factor=None,
        dithering_method="error_diffusion_hilbert",
        color_space="LAB",
        dither_scale=1,
        dither_kwargs=None,
        tiling_method="greedy_fill",
        use_isolated=True,
        render_scale=20,
        show=True,
        auto_determine_params='normal'
):
    """
    Generates a LEGO mosaic from an image using dithering + tile fitting.

    Args:
        image_path (str): Path to the input image.
        palette_name (str): Name of palette from PALETTES dict.
        dithering_method (str): Name of registered dithering method.
        color_space (str): Color space used for processing ('RGB', 'LAB', etc).
        dither_kwargs (dict): Extra parameters for the dithering method.
        tiling_method (str): Name of registered tiling method.
        use_isolated (bool): Whether to try placing isolated tiles first.
        render_scale (int): Pixel scale of each LEGO tile in the rendered image.
        show (bool): If True, displays the rendered result via matplotlib.

    Returns:
        tuple: (TileMap object, rendered image as NumPy array)
    """

    if dither_kwargs is None:
        dither_kwargs = {}

    # --- Load image ---
    img = Image.open(image_path).convert("RGB")
    if resize_factor is not None:
        new_size = (int(img.width * resize_factor), int(img.height * resize_factor))
        img = img.resize(new_size, Image.LANCZOS)
    img_array = np.array(img)

    # --- Prepare palette ---
    palette_rgb, inv_dict = prepare_palette(COLOR_PALETTES[palette_name])
    if auto_determine_params is not None:
        color_space, dither_scale = determine_scale_and_color_space(dithering_method,
                                                                    pallete_hex=COLOR_PALETTES[palette_name])
        if auto_determine_params == 'minimal':
            dither_scale /= 2
        elif auto_determine_params == 'aggressive':
            dither_scale *= 2

    # --- Apply dithering + quantization ---
    quantized = dither(
        img_rgb=img_array,
        rgb_palette=palette_rgb,
        color_space=color_space,
        scale=dither_scale,
        method_name=dithering_method,
        **dither_kwargs
    )

    # --- Apply LEGO tiling ---
    tile_map = tile_image(
        quantized_img=quantized,
        palette_rgb=palette_rgb,
        tile_specs=TILE_PALLETES[palette_name],
        method_name=tiling_method,
        use_isolated=use_isolated
    )

    # --- Render final image ---
    rendered_img = tile_map.render(scale=render_scale)

    if show:
        plt.figure(figsize=(rendered_img.shape[1] / 100, rendered_img.shape[0] / 100))
        plt.imshow(rendered_img)
        plt.axis("off")
        plt.title(f"Rendered Mosaic with {tile_map.count_tiles()} tiles")
        plt.show()

    return tile_map, rendered_img, tile_map.count_tiles_by_shape_and_color(inv_dict)


if __name__ == "__main__":
        tilemap, img, tile_count = generate_lego_mosaic(
        image_path=r"C:\Users\sgala\Documents\Lenna_(test_image).png",
        palette_name="lego_extended",
        dithering_method="error_diffusion_hilbert",
        color_space="LAB",
        dither_scale=0.5,
        dither_kwargs=None,
        tiling_method="greedy_fill",
        use_isolated=True,
        render_scale=20,
        show=True,
        resize_factor=1 / 4
    )
        print(tile_count)
