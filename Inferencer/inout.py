import numpy as np
from PIL import Image


def load_image(input, mode="L"):
    with Image.open(input) as source:
        converted = source.convert(mode)
        profile = converted.info.get("icc_profile")
        image = np.asarray(converted)
    if mode == "L":
        image = image[..., None]
    unit = image.astype(np.float32) / 255.0
    if image.dtype != np.uint8:
        np.clip(unit, 0.0, 1.0, out=unit)
    return unit, profile


def to_pil_image(output, mode="L"):
    if mode == "L" and output.ndim == 3 and output.shape[-1] == 1:
        output = output[..., 0]
    rounded = output * 255.0
    np.rint(rounded, out=rounded)
    return Image.fromarray(rounded.astype(np.uint8), mode=mode)


def save_image(output, filename, mode="L", icc_profile=None):
    to_pil_image(output, mode).save(filename, icc_profile=icc_profile)
