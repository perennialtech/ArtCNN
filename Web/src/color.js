// SPDX-FileCopyrightText: 2026 perennialtech contributors
// SPDX-License-Identifier: MIT

/**
 * Clean-still RGB helpers, 2x bilinear interpolation, and IEEE-754 float16 decode.
 */

export function clamp01(v) {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

/**
 * Decode an array of IEEE-754 binary16 (uint16) values into Float32Array.
 */
export function float16ToFloat32(u16) {
  const out = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) {
    const h = u16[i];
    const sign = h & 0x8000 ? -1 : 1;
    const exp = (h >> 10) & 0x1f;
    const frac = h & 0x03ff;
    if (exp === 0) {
      out[i] = sign * Math.pow(2, -14) * (frac / 1024);
    } else if (exp === 31) {
      out[i] = frac === 0 ? sign * Infinity : NaN;
    } else {
      out[i] = sign * Math.pow(2, exp - 15) * (1 + frac / 1024);
    }
  }
  return out;
}

/**
 * Convert RGBA Uint8ClampedArray/Uint8Array (width x height) to normalized planar RGB Float32Array [3, H, W].
 */
export function rgbaToPlanarRgb(rgba, width, height) {
  const n = width * height;
  const r = new Float32Array(n);
  const g = new Float32Array(n);
  const b = new Float32Array(n);
  for (let i = 0, p = 0; i < n; i++, p += 4) {
    r[i] = rgba[p] / 255;
    g[i] = rgba[p + 1] / 255;
    b[i] = rgba[p + 2] / 255;
  }
  return { r, g, b };
}

/**
 * 2x bilinear upscale of a single Float32Array plane matching PyTorch align_corners=False / WebGPU linear sampling.
 */
export function bilinear2xPlane(src, width, height) {
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

/**
 * 2x bilinear upscale of packed RGBA bytes (width x height -> 2W x 2H).
 */
export function bilinear2xRgba(rgba, width, height) {
  const outW = width * 2;
  const outH = height * 2;
  const dst = new Uint8ClampedArray(outW * outH * 4);
  for (let oy = 0; oy < outH; oy++) {
    const sy = Math.max(0, Math.min(height - 1, (oy + 0.5) * 0.5 - 0.5));
    const y0 = Math.floor(sy);
    const y1 = Math.min(height - 1, y0 + 1);
    const fy = sy - y0;
    const wy0 = 1 - fy;
    const row0 = y0 * width * 4;
    const row1 = y1 * width * 4;
    const outRow = oy * outW * 4;
    for (let ox = 0; ox < outW; ox++) {
      const sx = Math.max(0, Math.min(width - 1, (ox + 0.5) * 0.5 - 0.5));
      const x0 = Math.floor(sx);
      const x1 = Math.min(width - 1, x0 + 1);
      const fx = sx - x0;
      const wx0 = 1 - fx;
      const p00 = row0 + x0 * 4;
      const p10 = row0 + x1 * 4;
      const p01 = row1 + x0 * 4;
      const p11 = row1 + x1 * 4;
      const dp = outRow + ox * 4;
      for (let c = 0; c < 4; c++) {
        const v = (rgba[p00 + c] * wx0 + rgba[p10 + c] * fx) * wy0 + (rgba[p01 + c] * wx0 + rgba[p11 + c] * fx) * fy;
        dst[dp + c] = Math.round(v);
      }
    }
  }
  return dst;
}
