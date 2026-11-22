# SPDX-FileCopyrightText: 2026 perennialtech contributors
# SPDX-License-Identifier: MIT

"""Realistic anime degradation pipeline (multi-kernel resize, ringing, 4x4 DCT codec).

Extends ArtCNN v1.6.2's 4x4 DCT compression idea from Chroma models to Luma
super-resolution training, alongside randomized downscaling kernels, mild blur,
sub-pixel grid shifts, and pre-existing sharpening halos.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np
from scipy.fft import dctn, idctn


def _radial_qtable_4x4(strength: float) -> np.ndarray:
    """Monotonically increasing radial 4x4 quantization table (mirrors ArtCNN v1.6.2)."""
    u, v = np.meshgrid(np.arange(4, dtype=np.float32), np.arange(4, dtype=np.float32), indexing="ij")
    radius = np.sqrt(u * u + v * v)
    return 1.0 + strength * (radius ** 1.35)


def apply_dct4x4_compression(img: np.ndarray, strength: float = 0.015) -> np.ndarray:
    """Simulate AVC/HEVC/AV1-like 4x4 DCT block quantization on float32 [0, 1] luma."""
    H, W = img.shape
    pad_h = (4 - H % 4) % 4
    pad_w = (4 - W % 4) % 4
    padded = np.pad(img, ((0, pad_h), (0, pad_w)), mode="edge")
    Hp, Wp = padded.shape
    blocks = padded.reshape(Hp // 4, 4, Wp // 4, 4).transpose(0, 2, 1, 3)  # (Bh, Bw, 4, 4)
    qtable = _radial_qtable_4x4(strength)
    step = strength * 0.65 * qtable  # normalized quantization step in [0, 1] domain
    coeff = dctn(blocks, axes=(-2, -1), norm="ortho")
    q_coeff = np.round(coeff / step) * step
    recon = idctn(q_coeff, axes=(-2, -1), norm="ortho")
    out = recon.transpose(0, 2, 1, 3).reshape(Hp, Wp)[:H, :W]
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def apply_unsharp_ringing(img: np.ndarray, amount: float = 0.45, sigma: float = 0.9) -> np.ndarray:
    """Simulate baked-in sharpening halos/ringing from prior broadcast/mastering filters."""
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)
    sharpened = img + amount * (img - blurred)
    return np.clip(sharpened, 0.0, 1.0).astype(np.float32)


@dataclass
class DegradationConfig:
    mode: str = "realistic"  # "clean_area" (baseline) or "realistic"
    clean_prob: float = 0.35
    dct_prob: float = 0.50
    ringing_prob: float = 0.30
    blur_prob: float = 0.30
    noise_std_max: float = 0.008


def degrade_hr_to_lr(
    hr: np.ndarray,
    rng: np.random.Generator,
    cfg: DegradationConfig,
) -> np.ndarray:
    """Create a 0.5x LR input from an HR float32 [0, 1] luma image."""
    H, W = hr.shape
    lr_size = (W // 2, H // 2)

    if cfg.mode == "clean_area":
        return np.clip(cv2.resize(hr, lr_size, interpolation=cv2.INTER_AREA), 0.0, 1.0).astype(np.float32)

    work = hr.copy()
    is_clean_branch = rng.random() < cfg.clean_prob

    # 1. Optional mild pre-blur (simulates optical/anti-aliasing softness)
    if not is_clean_branch and rng.random() < cfg.blur_prob:
        sigma = float(rng.uniform(0.35, 0.85))
        work = cv2.GaussianBlur(work, (0, 0), sigmaX=sigma)

    # 2. Randomized resampling kernel
    interp = int(
        rng.choice([cv2.INTER_AREA, cv2.INTER_LINEAR_EXACT, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4])
    )
    lr = cv2.resize(work, lr_size, interpolation=interp)
    lr = np.clip(lr, 0.0, 1.0).astype(np.float32)

    if not is_clean_branch:
        # 3. Optional pre-existing sharpening halos
        if rng.random() < cfg.ringing_prob:
            lr = apply_unsharp_ringing(lr, amount=float(rng.uniform(0.25, 0.55)), sigma=0.8)

        # 4. Optional 4x4 DCT codec quantization
        if rng.random() < cfg.dct_prob:
            lr = apply_dct4x4_compression(lr, strength=float(rng.uniform(0.008, 0.024)))

        # 5. Mild sensor/dither noise
        if cfg.noise_std_max > 0:
            std = float(rng.uniform(0.0, cfg.noise_std_max))
            lr = np.clip(lr + rng.normal(0.0, std, size=lr.shape), 0.0, 1.0).astype(np.float32)

    return lr


def make_degraded_pair(
    hr: np.ndarray,
    rng: np.random.Generator,
    cfg: DegradationConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (lr[1, H/2, W/2], hr[1, H, W])."""
    lr = degrade_hr_to_lr(hr, rng, cfg)
    return lr[None, :, :], hr[None, :, :]
