# SPDX-FileCopyrightText: 2025 João Chrisóstomo
# SPDX-FileCopyrightText: 2026 perennialtech contributors
# SPDX-License-Identifier: MIT

import numpy as np
import cv2


def rgb2ycbcr_bt709(image):
    """
    Convert RGB image to YCbCr following BT.709 standard.

    Args:
    Image (numpy.ndarray): RGB image with channels last.

    Returns:
    Image (numpy.ndarray): YCbCr image with channels last.
    """
    image = np.clip(image, 0.0, 1.0)
    matrix = np.array([[0.2126, 0.7152, 0.0722], [-0.114572, -0.385428, 0.5], [0.5, -0.454153, -0.045847]], dtype=np.float32)
    offset = np.array([0.0, 0.5, 0.5], dtype=np.float32)
    ycbcr_image = image @ matrix.T + offset
    return np.clip(ycbcr_image, 0.0, 1.0)


def ycbcr2rgb_bt709(image):
    """
    Convert YCbCr image to RGB following BT.709 standard.

    Args:
    Image (numpy.ndarray): YCbCr image with channels last.

    Returns:
    Image (numpy.ndarray): RGB image with channels last.
    """
    image = np.clip(image, 0.0, 1.0)
    image[:, :, 1] -= 0.5
    image[:, :, 2] -= 0.5
    matrix = np.array([[1.0, 0.0, 1.5748], [1.0, -0.1873, -0.4681], [1.0, 1.8556, 0.0]], dtype=np.float32)
    rgb_image = image @ matrix.T
    return np.clip(rgb_image, 0.0, 1.0)


def rgb2ycbcr_bt601(image):
    """
    Convert RGB image to YCbCr following BT.601 standard.

    Args:
    Image (numpy.ndarray): RGB image with channels last.

    Returns:
    Image (numpy.ndarray): YCbCr image with channels last.
    """
    image = np.clip(image, 0.0, 1.0)
    bt601_matrix = np.array([[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5], [0.5, -0.418688, -0.081312]], dtype=np.float32)
    bt601_offset = np.array([0.0, 0.5, 0.5], dtype=np.float32)
    ycbcr = image @ bt601_matrix.T + bt601_offset
    return np.clip(ycbcr, 0.0, 1.0)


def ycbcr2rgb_bt601(image):
    """
    Convert YCbCr image to RGB following BT.601 standard.

    Args:
    Image (numpy.ndarray): YCbCr image with channels last.

    Returns:
    Image (numpy.ndarray): RGB image with channels last.
    """
    image = np.clip(image, 0.0, 1.0)
    image[:, :, 1] -= 0.5
    image[:, :, 2] -= 0.5
    matrix = np.array([[1.0, 0.0, 1.402], [1.0, -0.344136, -0.714136], [1.0, 1.772, 0.0]], dtype=np.float32)
    rgb_image = image @ matrix.T
    return np.clip(rgb_image, 0.0, 1.0)


def bilinear(image, scaling_factor):
    image = cv2.resize(
        image,
        None,
        fx=scaling_factor,
        fy=scaling_factor,
        interpolation=cv2.INTER_LINEAR_EXACT,
    )
    return image


def unstack(image):
    return tuple(np.moveaxis(image, -1, 0))


def stack(*channels):
    planes = [ch[..., 0] if ch.ndim == 3 and ch.shape[-1] == 1 else ch for ch in channels]
    return np.stack(planes, axis=-1)


def self_ensemble_augment(image):
    flipped = np.flip(image, axis=1)
    return [np.rot90(view, turns) for view in (image, flipped) for turns in range(4)]


def self_ensemble_combine(images):
    total = None
    count = 0
    for index, image in enumerate(images):
        if index == 8:
            raise ValueError("Expected eight ensemble images.")
        restored = np.rot90(image, -(index % 4))
        if index >= 4:
            restored = np.flip(restored, axis=1)
        if total is None:
            dtype = restored.dtype
            total = restored.astype(np.result_type(dtype, np.float32), copy=True)
        else:
            total += restored
        count += 1
    if count != 8:
        raise ValueError("Expected eight ensemble images.")
    total /= count
    return total.astype(dtype, copy=False)


def convert_to_fp16(model_fp32):
    from onnxconverter_common import float16

    return float16.convert_float_to_float16(model_fp32)
