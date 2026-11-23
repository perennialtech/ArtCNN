// SPDX-FileCopyrightText: 2026 perennialtech contributors
// SPDX-License-Identifier: MIT

/**
 * Zero-dependency Float32Array CPU fallback for clean-still RGB ArtCNN bundles.
 * Uses the exact same planar-vec4 packed weights and overlap tiling as WebGPU.
 */

import { bilinear2xRgba, clamp01, float16ToFloat32, rgbaToPlanarRgb } from "./color.js";
import { planTiles } from "./tiling.js";

function applyActivation(v, activation, leakySlope, preluSlope = null) {
  if (activation === "relu") return v > 0 ? v : 0;
  if (activation === "leaky_relu") return v > 0 ? v : v * leakySlope;
  if (activation === "prelu" && preluSlope !== null) return v > 0 ? v : v * preluSlope;
  return v;
}

function convIn3(rPlane, gPlane, bPlane, width, height, params, cOutV4, activation, leakySlope, preluSlopes = null) {
  const plane = width * height;
  const out = new Float32Array(cOutV4 * plane * 4);
  const wStart = cOutV4 * 4;
  for (let p = 0; p < cOutV4; p++) {
    const b0 = params[p * 4];
    const b1 = params[p * 4 + 1];
    const b2 = params[p * 4 + 2];
    const b3 = params[p * 4 + 3];
    const pW = wStart + p * 27 * 4;
    const outPlane = p * plane * 4;
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        let a0 = b0, a1 = b1, a2 = b2, a3 = b3;
        for (let ky = 0; ky < 3; ky++) {
          const sy = y + ky - 1;
          if (sy < 0 || sy >= height) continue;
          const row = sy * width;
          for (let kx = 0; kx < 3; kx++) {
            const sx = x + kx - 1;
            if (sx < 0 || sx >= width) continue;
            const pix = row + sx;
            const vr = rPlane[pix];
            const vg = gPlane[pix];
            const vb = bPlane[pix];
            const wOff = pW + (ky * 3 + kx) * 12;
            a0 += params[wOff] * vr + params[wOff + 4] * vg + params[wOff + 8] * vb;
            a1 += params[wOff + 1] * vr + params[wOff + 5] * vg + params[wOff + 9] * vb;
            a2 += params[wOff + 2] * vr + params[wOff + 6] * vg + params[wOff + 10] * vb;
            a3 += params[wOff + 3] * vr + params[wOff + 7] * vg + params[wOff + 11] * vb;
          }
        }
        const o = outPlane + (y * width + x) * 4;
        out[o] = applyActivation(a0, activation, leakySlope, preluSlopes ? preluSlopes[p * 4] : null);
        out[o + 1] = applyActivation(a1, activation, leakySlope, preluSlopes ? preluSlopes[p * 4 + 1] : null);
        out[o + 2] = applyActivation(a2, activation, leakySlope, preluSlopes ? preluSlopes[p * 4 + 2] : null);
        out[o + 3] = applyActivation(a3, activation, leakySlope, preluSlopes ? preluSlopes[p * 4 + 3] : null);
      }
    }
  }
  return out;
}

function mat4MulAdd(acc, params, mOff, v0, v1, v2, v3) {
  acc[0] += params[mOff] * v0 + params[mOff + 4] * v1 + params[mOff + 8] * v2 + params[mOff + 12] * v3;
  acc[1] += params[mOff + 1] * v0 + params[mOff + 5] * v1 + params[mOff + 9] * v2 + params[mOff + 13] * v3;
  acc[2] += params[mOff + 2] * v0 + params[mOff + 6] * v1 + params[mOff + 10] * v2 + params[mOff + 14] * v3;
  acc[3] += params[mOff + 3] * v0 + params[mOff + 7] * v1 + params[mOff + 11] * v2 + params[mOff + 15] * v3;
}

function convDense(inp, width, height, params, cInV4, cOutV4, activation, leakySlope, preluSlopes = null, resBuf = null) {
  const plane = width * height;
  const out = new Float32Array(cOutV4 * plane * 4);
  const wStart = cOutV4 * 4;
  const acc = new Float32Array(4);
  for (let p = 0; p < cOutV4; p++) {
    const b0 = params[p * 4], b1 = params[p * 4 + 1], b2 = params[p * 4 + 2], b3 = params[p * 4 + 3];
    const pBase = wStart + p * cInV4 * 144;
    const outPlane = p * plane * 4;
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        acc[0] = b0; acc[1] = b1; acc[2] = b2; acc[3] = b3;
        for (let z = 0; z < cInV4; z++) {
          const zPlane = z * plane * 4;
          const zBase = pBase + z * 144;
          for (let ky = 0; ky < 3; ky++) {
            const sy = y + ky - 1;
            if (sy < 0 || sy >= height) continue;
            const row = zPlane + sy * width * 4;
            for (let kx = 0; kx < 3; kx++) {
              const sx = x + kx - 1;
              if (sx < 0 || sx >= width) continue;
              const i = row + sx * 4;
              mat4MulAdd(acc, params, zBase + (ky * 3 + kx) * 16, inp[i], inp[i + 1], inp[i + 2], inp[i + 3]);
            }
          }
        }
        const o = outPlane + (y * width + x) * 4;
        if (resBuf) {
          acc[0] += resBuf[o];
          acc[1] += resBuf[o + 1];
          acc[2] += resBuf[o + 2];
          acc[3] += resBuf[o + 3];
        }
        out[o] = applyActivation(acc[0], activation, leakySlope, preluSlopes ? preluSlopes[p * 4] : null);
        out[o + 1] = applyActivation(acc[1], activation, leakySlope, preluSlopes ? preluSlopes[p * 4 + 1] : null);
        out[o + 2] = applyActivation(acc[2], activation, leakySlope, preluSlopes ? preluSlopes[p * 4 + 2] : null);
        out[o + 3] = applyActivation(acc[3], activation, leakySlope, preluSlopes ? preluSlopes[p * 4 + 3] : null);
      }
    }
  }
  return out;
}

function cropRgba(rgba, width, sx, sy, sw, sh) {
  const out = new Uint8ClampedArray(sw * sh * 4);
  for (let y = 0; y < sh; y++) {
    const srcOff = ((sy + y) * width + sx) * 4;
    out.set(rgba.subarray(srcOff, srcOff + sw * 4), y * sw * 4);
  }
  return out;
}

function copyTileOutput(tileOut, tileW, dst, dstW, validX, validY, validW, validH, dstX, dstY) {
  const tOutW = tileW * 2;
  const dOutW = dstW * 2;
  const vx = validX * 2;
  const vy = validY * 2;
  const vw = validW * 2;
  const vh = validH * 2;
  const dx = dstX * 2;
  const dy = dstY * 2;
  for (let y = 0; y < vh; y++) {
    const srcOff = ((vy + y) * tOutW + vx) * 4;
    const dstOff = ((dy + y) * dOutW + dx) * 4;
    dst.set(tileOut.subarray(srcOff, srcOff + vw * 4), dstOff);
  }
}

function sourceToImageData(source) {
  if (source instanceof ImageData) return source;
  const w = source.naturalWidth || source.width;
  const h = source.naturalHeight || source.height;
  const canvas = typeof OffscreenCanvas !== "undefined" ? new OffscreenCanvas(w, h) : document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(source, 0, 0, w, h);
  return ctx.getImageData(0, 0, w, h);
}

export class CpuUpscaler {
  constructor(model) {
    this.backend = "cpu";
    this.precision = model.precision === "f16" ? "f16" : "f32";
    this.model = model;
    const isF16 = this.precision === "f16";
    this._layerWeights = model.manifest.layers.map((layer) => {
      if (isF16) {
        const u16 = new Uint16Array(
          model.weights.buffer,
          model.weights.byteOffset + layer.byte_offset_f16,
          layer.byte_length_f16 / 2,
        );
        return float16ToFloat32(u16);
      }
      return new Float32Array(
        model.weights.buffer,
        model.weights.byteOffset + layer.byte_offset,
        layer.byte_length / 4,
      );
    });
  }

  _upscaleTileRgba2x(rgba, width, height) {
    const { manifest } = this.model;
    const layers = manifest.layers;
    const { r, g, b } = rgbaToPlanarRgb(rgba, width, height);
    let skip = null;
    let cur = null;
    let resAnchor = null;

    for (let i = 0; i < layers.length; i++) {
      const layer = layers[i];
      const params = this._layerWeights[i];
      if (layer.role === "entry") {
        skip = convIn3(
          r,
          g,
          b,
          width,
          height,
          params,
          layer.c_out / 4,
          layer.activation,
          manifest.leaky_slope,
          layer.prelu_slopes ?? null,
        );
        cur = skip;
      } else if (layer.role === "res_conv1") {
        resAnchor = cur;
        cur = convDense(
          cur,
          width,
          height,
          params,
          layer.c_in / 4,
          layer.c_out / 4,
          layer.activation,
          manifest.leaky_slope,
          layer.prelu_slopes ?? null,
        );
      } else if (layer.role === "res_conv2") {
        cur = convDense(
          cur,
          width,
          height,
          params,
          layer.c_in / 4,
          layer.c_out / 4,
          layer.activation,
          manifest.leaky_slope,
          layer.prelu_slopes ?? null,
          resAnchor,
        );
      } else if (layer.role === "features_d2s") {
        const added = new Float32Array(cur.length);
        for (let j = 0; j < cur.length; j++) added[j] = cur[j] + skip[j];
        // feats has 3 planar vec4s: plane 0 = R(2x2), plane 1 = G(2x2), plane 2 = B(2x2)
        const feats = convDense(added, width, height, params, layer.c_in / 4, 3, "none", 0);
        const outW = width * 2;
        const outH = height * 2;
        const planeIn = width * height;
        const baseRgba = bilinear2xRgba(rgba, width, height);
        const outRgba = new Uint8ClampedArray(outW * outH * 4);
        const useSkip = Boolean(manifest.input_skip);
        const gOff = planeIn * 4;
        const bOff = planeIn * 8;

        for (let y = 0; y < height; y++) {
          for (let x = 0; x < width; x++) {
            const p = (y * width + x) * 4;
            const ox = x * 2;
            const oy = y * 2;
            const coords = [
              (oy * outW + ox) * 4,
              (oy * outW + ox + 1) * 4,
              ((oy + 1) * outW + ox) * 4,
              ((oy + 1) * outW + ox + 1) * 4,
            ];
            for (let s = 0; s < 4; s++) {
              const dst = coords[s];
              const baseR = useSkip ? baseRgba[dst] / 255 : 0;
              const baseG = useSkip ? baseRgba[dst + 1] / 255 : 0;
              const baseB = useSkip ? baseRgba[dst + 2] / 255 : 0;
              outRgba[dst] = Math.round(clamp01(feats[p + s] + baseR) * 255);
              outRgba[dst + 1] = Math.round(clamp01(feats[gOff + p + s] + baseG) * 255);
              outRgba[dst + 2] = Math.round(clamp01(feats[bOff + p + s] + baseB) * 255);
              outRgba[dst + 3] = baseRgba[dst + 3];
            }
          }
        }
        return outRgba;
      } else {
        cur = convDense(
          cur,
          width,
          height,
          params,
          layer.c_in / 4,
          layer.c_out / 4,
          layer.activation,
          manifest.leaky_slope,
          layer.prelu_slopes ?? null,
        );
      }
    }
    throw new Error("Model manifest is missing features_d2s layer.");
  }

  _upscale2xTiled(rgba, width, height, options = {}, progressOffset = 0, progressSpan = 1) {
    const tileSize = options.tileSize ?? 256;
    const defaultOverlap = Math.max(16, (this.model.manifest.receptive_radius || 8) + 2);
    const overlap = options.overlap ?? defaultOverlap;
    const tiles = planTiles(width, height, tileSize, overlap);
    if (tiles.length === 1) {
      const out = this._upscaleTileRgba2x(rgba, width, height);
      options.onProgress?.(progressOffset + progressSpan);
      return out;
    }

    const out = new Uint8ClampedArray(width * 2 * height * 2 * 4);
    for (let i = 0; i < tiles.length; i++) {
      if (options.signal?.aborted) throw new DOMException("Aborted", "AbortError");
      const t = tiles[i];
      const crop = cropRgba(rgba, width, t.srcX, t.srcY, t.srcW, t.srcH);
      const tileOut = this._upscaleTileRgba2x(crop, t.srcW, t.srcH);
      copyTileOutput(tileOut, t.srcW, out, width, t.validX, t.validY, t.validW, t.validH, t.dstX, t.dstY);
      options.onProgress?.(progressOffset + ((i + 1) / tiles.length) * progressSpan);
    }
    return out;
  }

  upscaleRgba(rgba, width, height, options = {}) {
    const scale = options.scale ?? 2;
    if (scale === 2) {
      return this._upscale2xTiled(rgba, width, height, options, 0, 1);
    }
    if (scale === 4) {
      const mid = this._upscale2xTiled(rgba, width, height, options, 0, 0.25);
      return this._upscale2xTiled(mid, width * 2, height * 2, options, 0.25, 0.75);
    }
    throw new Error(`Unsupported scale: ${scale}. Use 2 or 4.`);
  }

  async upscaleToImageData(source, options = {}) {
    const scale = options.scale ?? 2;
    const imgData = sourceToImageData(source);
    const outRgba = this.upscaleRgba(imgData.data, imgData.width, imgData.height, options);
    return new ImageData(outRgba, imgData.width * scale, imgData.height * scale);
  }

  async upscaleToBlob(source, options = {}) {
    const imgData = await this.upscaleToImageData(source, options);
    const canvas = document.createElement("canvas");
    canvas.width = imgData.width;
    canvas.height = imgData.height;
    canvas.getContext("2d").putImageData(imgData, 0, 0);
    return new Promise((resolve, reject) => {
      canvas.toBlob(
        (blob) => (blob ? resolve(blob) : reject(new Error("Failed to encode image blob."))),
        options.type || "image/png",
        options.quality,
      );
    });
  }

  async renderToCanvas(source, canvas, options = {}) {
    const scale = options.scale ?? 2;
    const split = options.split ?? 0;
    const imgData = sourceToImageData(source);
    const outW = imgData.width * scale;
    const outH = imgData.height * scale;
    const outRgba = this.upscaleRgba(imgData.data, imgData.width, imgData.height, options);

    const srcCanvas = document.createElement("canvas");
    srcCanvas.width = imgData.width;
    srcCanvas.height = imgData.height;
    srcCanvas.getContext("2d").putImageData(imgData, 0, 0);

    const upCanvas = document.createElement("canvas");
    upCanvas.width = outW;
    upCanvas.height = outH;
    upCanvas.getContext("2d").putImageData(new ImageData(outRgba, outW, outH), 0, 0);

    if (canvas.width !== outW || canvas.height !== outH) {
      canvas.width = outW;
      canvas.height = outH;
    }
    this._lastSrc = srcCanvas;
    this._lastUp = upCanvas;
    this.presentOnly(canvas, split);
    return { width: outW, height: outH };
  }

  presentOnly(canvas, split = 0) {
    if (!this._lastSrc || !this._lastUp) return;
    const ctx = canvas.getContext("2d");
    const outW = canvas.width;
    const outH = canvas.height;
    const cut = Math.round(outW * Math.max(0, Math.min(1, split)));
    ctx.drawImage(this._lastUp, 0, 0);
    if (cut > 0) {
      ctx.save();
      ctx.beginPath();
      ctx.rect(0, 0, cut, outH);
      ctx.clip();
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = "high";
      ctx.drawImage(this._lastSrc, 0, 0, outW, outH);
      ctx.restore();
    }
  }

  destroy() {
    this._lastSrc = null;
    this._lastUp = null;
  }
}
