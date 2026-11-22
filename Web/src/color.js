// SPDX-FileCopyrightText: 2026 perennialtech contributors
// SPDX-License-Identifier: MIT

/**
 * BT.709 RGB <-> YCbCr and 2x bilinear helpers matching Inferencer/utils.py.
 */

export function clamp01(v) {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

/**
 * Extract normalized BT.709 Y, Cb, Cr planes (Float32Array in [0, 1]) from RGBA bytes.
 */
export function rgbaToYCbCrBT709(rgba, width, height) {
  const n = width * height;
  const y = new Float32Array(n);
  const cb = new Float32Array(n);
  const cr = new Float32Array(n);
  for (let i = 0, p = 0; i < n; i++, p += 4) {
    const r = rgba[p] / 255;
    const g = rgba[p + 1] / 255;
    const b = rgba[p + 2] / 255;
    y[i] = clamp01(0.2126 * r + 0.7152 * g + 0.0722 * b);
    cb[i] = clamp01(-0.114572 * r - 0.385428 * g + 0.5 * b + 0.5);
    cr[i] = clamp01(0.5 * r - 0.454153 * g - 0.045847 * b + 0.5);
  }
  return { y, cb, cr };
}

/**
 * Combine BT.709 Y, Cb, Cr planes (Float32Array in [0, 1]) into RGBA Uint8ClampedArray.
 */
export function ycbcrToRgbaBT709(y, cb, cr, width, height) {
  const n = width * height;
  const rgba = new Uint8ClampedArray(n * 4);
  for (let i = 0, p = 0; i < n; i++, p += 4) {
    const yv = clamp01(y[i]);
    const cbv = clamp01(cb[i]) - 0.5;
    const crv = clamp01(cr[i]) - 0.5;
    const r = clamp01(yv + 1.5748 * crv);
    const g = clamp01(yv - 0.1873 * cbv - 0.4681 * crv);
    const b = clamp01(yv + 1.8556 * cbv);
    rgba[p] = Math.round(r * 255);
    rgba[p + 1] = Math.round(g * 255);
    rgba[p + 2] = Math.round(b * 255);
    rgba[p + 3] = 255;
  }
  return rgba;
}

/**
 * 2x bilinear upscale of a single Float32Array plane matching OpenCV INTER_LINEAR / WebGPU linear sampling.
 */
export function bilinear2x(src, width, height) {
  const outW = width * 2;
  const outH = height * 2;
  const dst = new Float32Array(outW * outH);
  for (let oy = 0; oy < outH; oy++) {
    const sy = Math.max(0, Math.min(height - 1, (oy + 0.5) * 0.5 - 0.5));
    const y0 = Math.floor(sy);
    const y1 = Math.min(height - 1, y0 + 1);
    const fy = sy - y0;
    const wy0 = 1 - fy;
    const row0 = y0 * width;
    const row1 = y1 * width;
    const outRow = oy * outW;
    for (let ox = 0; ox < outW; ox++) {
      const sx = Math.max(0, Math.min(width - 1, (ox + 0.5) * 0.5 - 0.5));
      const x0 = Math.floor(sx);
      const x1 = Math.min(width - 1, x0 + 1);
      const fx = sx - x0;
      const wx0 = 1 - fx;
      const v00 = src[row0 + x0];
      const v10 = src[row0 + x1];
      const v01 = src[row1 + x0];
      const v11 = src[row1 + x1];
      dst[outRow + ox] = (v00 * wx0 + v10 * fx) * wy0 + (v01 * wx0 + v11 * fx) * fy;
    }
  }
  return dst;
}
