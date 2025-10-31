import os
import json
import argparse
from PIL import Image
from lego_mosaic_pipeline import generate_lego_mosaic

from utils.tile_data import COLOR_PALETTES, TILE_PALLETES


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LEGO Mosaic Generator")

    parser.add_argument("--input", type=str, default="input/Lenna_(test_image).png", help="Path to input image")
    parser.add_argument("--output", type=str, default="output/result.png", help="Path to save rendered output image")
    parser.add_argument("--palette", type=str, default="lego_extended", choices=COLOR_PALETTES.keys(), help="Color palette name")
    parser.add_argument("--dither_method", type=str, default='error_diffusion_raster', help="Name of the dithering method")
    parser.add_argument("--color_space", type=str, default="LAB", choices=["RGB", "LAB", "HSV"], help="Color space to use")
    parser.add_argument("--tile_method", type=str, default="greedy_fill", help="Tiling method name")
    parser.add_argument("--use_isolated", action="store_true", help="Enable isolated tile placement before fill")
    parser.add_argument("--resize", type=int, default=96, help="Resize input image height to N pixels (maintains aspect ratio)")
    parser.add_argument("--strength", type=float, default=1, help="Dither strength factor (e.g., 0.8)")
    parser.add_argument("--render_scale", type=int, default=20, help="Rendering scale for final output image")
    parser.add_argument("--auto_determine_params", type=str, default="normal", choices=["normal", "minimal", "agressive",None],
                        help="Automatically choose optimal dithering parameters (WIP)")

    parser.add_argument(
        "--dither_args", type=str, default="{}",
        help='Extra dithering method kwargs as JSON string, e.g. \'{"kernel_name": "floyd-steinberg"}\''
    )

    args = parser.parse_args()

    try:
        dither_kwargs = json.loads(args.dither_args)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON for --dither_args: {e}")
    
    resize_factor = args.resize / Image.open(args.input).height

    tilemap, _, tile_count = generate_lego_mosaic(
        image_path=args.input,
        palette_name=args.palette,
        dithering_method=args.dither_method,
        color_space=args.color_space,
        tiling_method=args.tile_method,
        use_isolated=args.use_isolated,
        resize_factor=resize_factor,
        dither_scale=args.strength,
        dither_kwargs=dither_kwargs,
        show=False,
        render_scale=args.render_scale,
        auto_determine_params=args.auto_determine_params
    )

    # Save rendered image
    rendered_img = tilemap.render(scale=args.render_scale)
    Image.fromarray(rendered_img).save(args.output)
    print(f"Saved mosaic image to {args.output}")

    # Save tile count
    base_path, _ = os.path.splitext(args.output)
    tile_info_path = f"{base_path}_tile_count.txt"
    with open(tile_info_path, "w") as f:
        f.write(f"{tile_count}\n")
    print(f"Tile count saved to {tile_info_path}")
