// SPDX-FileCopyrightText: 2026 perennialtech contributors
// SPDX-License-Identifier: MIT

/**
 * Overlap-tiling planner for bounded-VRAM still-image upscaling.
 */

export function planTiles(width, height, tileSize = 256, overlap = 16) {
  if (width <= tileSize && height <= tileSize) {
    return [
      {
        srcX: 0,
        srcY: 0,
        srcW: width,
        srcH: height,
        validX: 0,
        validY: 0,
        validW: width,
        validH: height,
        dstX: 0,
        dstY: 0,
      },
    ];
  }

  const tiles = [];
  for (let y = 0; y < height; y += tileSize) {
    const validH = Math.min(tileSize, height - y);
    const srcY = Math.max(0, y - overlap);
    const srcEndY = Math.min(height, y + validH + overlap);
    const srcH = srcEndY - srcY;
    const validY = y - srcY;

    for (let x = 0; x < width; x += tileSize) {
      const validW = Math.min(tileSize, width - x);
      const srcX = Math.max(0, x - overlap);
      const srcEndX = Math.min(width, x + validW + overlap);
      const srcW = srcEndX - srcX;
      const validX = x - srcX;

      tiles.push({
        srcX,
        srcY,
        srcW,
        srcH,
        validX,
        validY,
        validW,
        validH,
        dstX: x,
        dstY: y,
      });
    }
  }
  return tiles;
}
