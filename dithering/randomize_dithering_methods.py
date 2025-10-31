import os
import random
import numpy as np
from PIL import Image
from dithering.pipeline import dither, DITHERING_METHODS
from utils.pattern_utils import PATTERN_REGISTRY
from utils.tile_data import COLOR_PALETTES


# --- Param Sampler Utilities ---

def sample_param(param_type, values):
    if param_type == 'range':
        lo, hi = values
        return random.uniform(lo, hi)
    elif param_type == 'list':
        return random.choice(values)
    elif param_type == 'pattern':
        return random.choice(values)
    else:
        raise ValueError(f"Unknown parameter type: {param_type}")


def get_valid_channels(color_space):
    return {
        "RGB": ["R", "G", "B"],
        "LAB": ["L", "a", "b"],
        "HSV": ["Hue", "Sat", "Val"]
    }.get(color_space.upper(), [])


def random_dither_params(method_name):
    method_info = DITHERING_METHODS.get(method_name)
    if not method_info:
        raise ValueError(f"Method '{method_name}' not found in registry")

    params = {}

    # Handle color_space first if it exists
    if 'color_space' in method_info:
        cs_meta = method_info['color_space']
        color_space = sample_param(cs_meta['type'], cs_meta['values'])
        params['color_space'] = color_space
    else:
        color_space = None

    for key, meta in method_info.items():
        if key in ('func', 'color_space'):
            continue

        if key == 'channel_name':
            if not color_space:
                continue
            valid_channels = get_valid_channels(color_space)
            if not valid_channels:
                continue
            params[key] = random.choice(valid_channels)

        elif key == 'pattern':
            # Choose pattern type and its params
            pattern_type = sample_param('pattern', meta['values'])
            params['pattern'] = pattern_type

            pattern_info = PATTERN_REGISTRY.get(pattern_type, {})
            pat_params_meta = pattern_info.get('params', {})

            pattern_params = {
                pat_k: sample_param(pat_meta['type'], pat_meta['values'])
                for pat_k, pat_meta in pat_params_meta.items()
            }
            params['pattern_params'] = pattern_params

        else:
            params[key] = sample_param(meta['type'], meta['values'])

    return params



# --- Main ---

def main():
    input_path = r"C:\Users\sgala\Documents\Lenna_(test_image).png"  # Update path
    output_dir = "test_outputs_random"
    os.makedirs(output_dir, exist_ok=True)

    img = Image.open(input_path).convert("RGB")

    for i in range(20):
        # Resize randomly
        scale = random.uniform(0.1, 1.1)
        new_size = (int(img.width * scale), int(img.height * scale))
        img_resized = img.resize(new_size, Image.LANCZOS)
        img_array = np.array(img_resized)

        method = random.choice(list(DITHERING_METHODS.keys()))
        palette_name = random.choice(list(COLOR_PALETTES.keys()))
        palette = COLOR_PALETTES[palette_name]

        kwargs = random_dither_params(method)
        color_space = kwargs.pop('color_space', 'RGB')

        try:
            out = dither(img_array, palette, color_space=color_space, method_name=method,scale=scale, **kwargs)
            out_img = Image.fromarray(out)

            # Include pattern name if applicable
            pattern_str = kwargs.get('pattern', '')
            param_str = '_'.join(
                f"{k}{str(v)[:6]}" for k, v in kwargs.items() if k not in ['pattern_params']
            )
            fname = f"{i+1:02d}_{method}_{color_space}_{palette_name}_{pattern_str}_{param_str}.png"
            out_img.save(os.path.join(output_dir, fname))
            print(f"✅ Saved {fname}")

        except Exception as e:
            print(f"❌ Failed for method={method}, space={color_space}, palette={palette_name}: {e}")


if __name__ == "__main__":
    main()
