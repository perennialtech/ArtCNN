# SPDX-FileCopyrightText: 2026 perennialtech contributors
# SPDX-License-Identifier: MIT

"""Procedural anime-like HR scene generator and balanced patch sampler.

Covers the 6 visual regimes critical for anime super-resolution evaluation:
  1. line_art: crisp dark-on-light curved/diagonal outlines & hair intersections
  2. flat_fills: large uniform cel-shaded regions (tests freedom from speckling)
  3. gradients: smooth sky/lighting ramps (tests freedom from banding)
  4. small_text: crisp high-contrast glyph-like strokes
  5. repeating_grid: fences, meshes, windowpanes (tests moire/aliasing)
  6. fine_texture: subtle film grain / painted background texture
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np


CATEGORIES = (
    "line_art",
    "flat_fills",
    "gradients",
    "small_text",
    "repeating_grid",
    "fine_texture",
)


def make_anime_scene(category: str, size: int = 96, seed: int = 0) -> np.ndarray:
    """Generate a single-channel float32 [0, 1] HR anime-style test patch."""
    rng = np.random.default_rng(seed)
    img = np.full((size, size), 0.85, dtype=np.float32)

    if category == "line_art":
        # Flat cel background + crisp curved/diagonal outlines + hair-like crossings
        bg = float(rng.uniform(0.72, 0.94))
        img.fill(bg)
        # Cel-shaded polygon fill
        pts = rng.integers(8, size - 8, size=(4, 2))
        cv2.fillConvexPoly(img, pts, float(rng.uniform(0.45, 0.72)))
        # Crisp dark outlines and intersecting diagonal hair strands
        for _ in range(7):
            p1 = tuple(int(v) for v in rng.integers(2, size - 2, size=2))
            p2 = tuple(int(v) for v in rng.integers(2, size - 2, size=2))
            thickness = int(rng.choice([1, 2]))
            cv2.line(img, p1, p2, float(rng.uniform(0.05, 0.18)), thickness, cv2.LINE_AA)
        center = (int(rng.integers(size // 4, 3 * size // 4)), int(rng.integers(size // 4, 3 * size // 4)))
        axes = (int(rng.integers(max(4, size // 8), max(8, size // 3))), int(rng.integers(max(4, size // 8), max(8, size // 3))))
        cv2.ellipse(img, center, axes, float(rng.uniform(0, 180)), 0, 360, 0.08, 1, cv2.LINE_AA)

    elif category == "flat_fills":
        # Large uniform cel-shaded panels with sharp clean boundaries
        c1, c2, c3 = rng.uniform(0.25, 0.92, size=3)
        img.fill(float(c1))
        split_x = int(rng.integers(size // 4, 3 * size // 4))
        split_y = int(rng.integers(size // 4, 3 * size // 4))
        img[:, split_x:] = float(c2)
        img[split_y:, :split_x] = float(c3)
        cv2.line(img, (split_x, 0), (split_x, size - 1), 0.1, 1, cv2.LINE_AA)
        cv2.line(img, (0, split_y), (split_x, split_y), 0.1, 1, cv2.LINE_AA)

    elif category == "gradients":
        # Smooth sky/backlight gradient with a clean silhouette edge
        y = np.linspace(0.0, 1.0, size, dtype=np.float32)[:, None]
        x = np.linspace(0.0, 1.0, size, dtype=np.float32)[None, :]
        angle = float(rng.uniform(0, 2 * np.pi))
        ramp = 0.5 + 0.35 * (np.cos(angle) * (x - 0.5) + np.sin(angle) * (y - 0.5))
        img = np.clip(ramp, 0.15, 0.92).astype(np.float32)
        cx, cy = int(rng.integers(size // 4, 3 * size // 4)), int(rng.integers(size // 4, 3 * size // 4))
        cv2.circle(img, (cx, cy), int(rng.integers(max(4, size // 8), max(8, size // 4))), 0.18, -1, cv2.LINE_AA)

    elif category == "small_text":
        # Subtitle/credit-like crisp characters on dark/light bar
        bg = float(rng.choice([0.12, 0.88]))
        fg = 0.94 if bg < 0.5 else 0.08
        # OpenCV 5 renders text on 8-bit images. Blend the glyph coverage into luma.
        text_mask = np.zeros(img.shape, dtype=np.uint8)
        words = ["EP.04", "ARTCNN", "1080p", "END"]
        for row_idx, word in enumerate(words):
            y_pos = 18 + row_idx * 22
            if y_pos < size - 4:
                cv2.putText(
                    text_mask,
                    word,
                    (6, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    255,
                    1,
                    cv2.LINE_AA,
                )
        img = bg + (fg - bg) * (text_mask.astype(np.float32) / 255.0)

    elif category == "repeating_grid":
        # Fence / windowpane / screen-tone repeating lines
        img.fill(0.86)
        step = int(rng.choice([6, 8, 10]))
        for k in range(0, size, step):
            cv2.line(img, (k, 0), (k, size - 1), 0.16, 1, cv2.LINE_AA)
            cv2.line(img, (0, k), (size - 1, k), 0.16, 1, cv2.LINE_AA)
        # Slight diagonal fence overlay
        for k in range(-size, size, step * 2):
            cv2.line(img, (k, 0), (k + size, size), 0.25, 1, cv2.LINE_AA)

    elif category == "fine_texture":
        # Subtle painted background + controlled grain + crisp foreground outline
        base = np.linspace(0.35, 0.75, size, dtype=np.float32)[:, None]
        img = np.repeat(base, size, axis=1)
        grain = rng.normal(0.0, 0.025, size=(size, size)).astype(np.float32)
        img = np.clip(img + grain, 0.0, 1.0)
        cv2.rectangle(img, (18, 18), (size - 18, size - 18), 0.1, 2, cv2.LINE_AA)

    else:
        raise ValueError(f"Unknown category: {category}")

    return np.clip(img, 0.0, 1.0).astype(np.float32)


def generate_dataset(
    n_per_category: int = 8,
    size: int = 64,
    seed_offset: int = 0,
) -> Tuple[List[np.ndarray], List[str]]:
    """Generate a balanced HR dataset across all 6 anime visual categories."""
    images: List[np.ndarray] = []
    labels: List[str] = []
    idx = seed_offset
    for cat in CATEGORIES:
        for _ in range(n_per_category):
            images.append(make_anime_scene(cat, size=size, seed=idx))
            labels.append(cat)
            idx += 1
    return images, labels


@dataclass
class PatchStats:
    edge_density: float
    flat_ratio: float
    png_bytes: int
    bucket: str


def classify_patch(img: np.ndarray) -> PatchStats:
    """Classify an HR patch by edge density and flat-region coverage
    (replaces PNG-filesize-only pruning that discards flat fills/gradients)."""
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    edge_density = float(np.mean(mag > 0.15))
    flat_ratio = float(np.mean(mag < 0.03))
    ok, buf = cv2.imencode(".png", np.clip(img * 255.0, 0, 255).astype(np.uint8))
    png_bytes = int(len(buf)) if ok else 0
    if flat_ratio > 0.65 and edge_density < 0.10:
        bucket = "flat_or_gradient"
    elif edge_density > 0.25:
        bucket = "dense_detail"
    else:
        bucket = "balanced_lines"
    return PatchStats(edge_density=edge_density, flat_ratio=flat_ratio, png_bytes=png_bytes, bucket=bucket)


class BalancedPatchSampler:
    """Stratified sampler ensuring flat fills, gradients, and dense edges are all kept."""

    def __init__(self, images: List[np.ndarray], seed: int = 0):
        self.images = images
        self.rng = np.random.default_rng(seed)
        self.buckets: Dict[str, List[int]] = {
            "flat_or_gradient": [],
            "balanced_lines": [],
            "dense_detail": [],
        }
        for i, img in enumerate(images):
            st = classify_patch(img)
            self.buckets[st.bucket].append(i)
        self.active_buckets = [k for k, v in self.buckets.items() if v]

    def sample_batch(self, batch_size: int) -> List[np.ndarray]:
        batch: List[np.ndarray] = []
        for i in range(batch_size):
            bname = self.active_buckets[i % len(self.active_buckets)]
            idx = int(self.rng.choice(self.buckets[bname]))
            img = self.images[idx]
            # Random 90-deg rotation + horizontal flip (matches ArtCNN_R_Flax)
            k = int(self.rng.integers(0, 4))
            if self.rng.random() < 0.5:
                img = np.fliplr(img)
            img = np.rot90(img, k).copy()
            batch.append(img)
        return batch
