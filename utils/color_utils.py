import numpy as np
import cv2
from skimage.color import rgb2lab, rgb2hsv

def clamp_img(img, color_space):
    if color_space == 'RGB':
        return np.clip(img, 0, 255)
    elif color_space == 'HSV':
        img[..., 0] = img[..., 0] % 180  # hue wrap
        img[..., 1:] = np.clip(img[..., 1:], 0, 255)
        return img
    elif color_space == 'LAB':
        img[..., 0] = np.clip(img[..., 0], 0, 255)
        img[..., 1:] = np.clip(img[..., 1:], -128, 127)

        return img
    else:
        raise ValueError(f"Unsupported color space {color_space}")


def rgb_to_cs(img_rgb, color_space):
    img_rgb = img_rgb.astype(np.float32) / 255.0

    if color_space == "RGB":
        return img_rgb

    elif color_space == "HSV":
        img_bgr = cv2.cvtColor((img_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 0] /= 179.0  # Normalize H
        hsv[..., 1] /= 255.0  # Normalize S
        hsv[..., 2] /= 255.0  # Normalize V
        return hsv

    elif color_space == "LAB":
        img_bgr = cv2.cvtColor((img_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab[..., 0] /= 255.0  # L in [0, 1]
        lab[..., 1] = (lab[..., 1] - 128.0) / 127.0  # a in [-1, 1]
        lab[..., 2] = (lab[..., 2] - 128.0) / 127.0  # b in [-1, 1]
        return lab

    else:
        raise ValueError(f"Unsupported color space: {color_space}")


def cs_to_rgb(img_cs, color_space):
    if color_space == "RGB":
        img_rgb = np.clip(img_cs, 0, 1)
        return (img_rgb * 255).astype(np.uint8)

    elif color_space == "HSV":
        hsv = img_cs.copy()
        hsv[..., 0] = (hsv[..., 0] * 179.0) % 180.0
        hsv[..., 1] = np.clip(hsv[..., 1] * 255.0, 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2] * 255.0, 0, 255)
        hsv_uint8 = hsv.astype(np.uint8)
        bgr = cv2.cvtColor(hsv_uint8, cv2.COLOR_HSV2BGR)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    elif color_space == "LAB":
        lab = img_cs.copy()
        lab[..., 0] = np.clip(lab[..., 0] * 255.0, 0, 255)
        lab[..., 1] = np.clip(lab[..., 1] * 127.0 + 128.0, 0, 255)
        lab[..., 2] = np.clip(lab[..., 2] * 127.0 + 128.0, 0, 255)
        lab_uint8 = lab.astype(np.uint8)
        bgr = cv2.cvtColor(lab_uint8, cv2.COLOR_LAB2BGR)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    else:
        raise ValueError(f"Unsupported color space: {color_space}")

def hex_to_rgb(hex_color):
    hex_color = hex_color.strip("#")
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def prepare_palette(hex_dict: dict[str:str]):
    return np.array([hex_to_rgb(c) for c in hex_dict.values()]), {hex_to_rgb(v) : k for k, v in hex_dict.items()}


