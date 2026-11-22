// SPDX-FileCopyrightText: 2026 perennialtech contributors
// SPDX-License-Identifier: MIT

/**
 * Zero-dependency Float32Array CPU fallback for ArtCNN browser bundles.
 * Uses the exact same planar-vec4 packed weights as the WebGPU runtime so models
 * work on browsers without WebGPU and can be unit-tested in Node.js.
 */

import { bilinear2x, clamp01, rgbaToYCbCrBT709, ycbcrToRgbaBT709 } from "./color.js";

function applyActivation(v, activation, leakySlope) {
  if (activation === "relu") return v > 0 ? v : 0;
  if (activation === "leaky_relu") return v > 0 ? v : v * leakySlope;
  return v;
}

function convIn1(inp, width, height, params, cOutV4, activation, leakySlope) {
  const plane = width * height;
  const out = new Float32Array(cOutV4 * plane * 4);
  const wStart = cOutV4 * 4;
  for (let p = 0; p < cOutV4; p++) {
    const b0 = params[p * 4];
    const b1 = params[p * 4 + 1];
    const b2 = params[p * 4 + 2];
    const b3 = params[p * 4 + 3];
    const pW = wStart + p * 36;
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
            const v = inp[row + sx];
            const wOff = pW + (ky * 3 + kx) * 4;
            a0 += params[wOff] * v;
            a1 += params[wOff + 1] * v;
            a2 += params[wOff + 2] * v;
            a3 += params[wOff + 3] * v;
          }
        }
        const o = outPlane + (y * width + x) * 4;
        out[o] = applyActivation(a0, activation, leakySlope);
        out[o + 1] = applyActivation(a1, activation, leakySlope);
        out[o + 2] = applyActivation(a2, activation, leakySlope);
        out[o + 3] = applyActivation(a3, activation, leakySlope);
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

function convDense(inp, width, height, params, cInV4, cOutV4, activation, leakySlope) {
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
        out[o] = applyActivation(acc[0], activation, leakySlope);
        out[o + 1] = applyActivation(acc[1], activation, leakySlope);
        out[o + 2] = applyActivation(acc[2], activation, leakySlope);
        out[o + 3] = applyActivation(acc[3], activation, leakySlope);
      }
    }
  }
  return out;
}

function convGrouped(inp, width, height, params, gV4, activation, leakySlope) {
  const plane = width * height;
  const out = new Float32Array(gV4 * plane * 4);
  const spatialBase = gV4 * 4;
  const mixBase = spatialBase + gV4 * 144;
  const grouped = new Float32Array(gV4 * 4);
  const tmp = new Float32Array(4);
  const acc = new Float32Array(4);

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      for (let z = 0; z < gV4; z++) {
        tmp[0] = 0; tmp[1] = 0; tmp[2] = 0; tmp[3] = 0;
        const zPlane = z * plane * 4;
        const zSpatial = spatialBase + z * 144;
        for (let ky = 0; ky < 3; ky++) {
          const sy = y + ky - 1;
          if (sy < 0 || sy >= height) continue;
          const row = zPlane + sy * width * 4;
          for (let kx = 0; kx < 3; kx++) {
            const sx = x + kx - 1;
            if (sx < 0 || sx >= width) continue;
            const i = row + sx * 4;
            mat4MulAdd(tmp, params, zSpatial + (ky * 3 + kx) * 16, inp[i], inp[i + 1], inp[i + 2], inp[i + 3]);
          }
        }
        grouped[z * 4] = tmp[0];
        grouped[z * 4 + 1] = tmp[1];
        grouped[z * 4 + 2] = tmp[2];
        grouped[z * 4 + 3] = tmp[3];
      }
      const pixOff = (y * width + x) * 4;
      for (let p = 0; p < gV4; p++) {
        acc[0] = params[p * 4];
        acc[1] = params[p * 4 + 1];
        acc[2] = params[p * 4 + 2];
        acc[3] = params[p * 4 + 3];
        const pMix = mixBase + p * gV4 * 16;
        for (let z = 0; z < gV4; z++) {
          const gOff = z * 4;
          mat4MulAdd(
            acc,
            params,
            pMix + z * 16,
            grouped[gOff],
            grouped[gOff + 1],
            grouped[gOff + 2],
            grouped[gOff + 3],
          );
        }
        const o = p * plane * 4 + pixOff;
        out[o] = applyActivation(acc[0], activation, leakySlope);
        out[o + 1] = applyActivation(acc[1], activation, leakySlope);
        out[o + 2] = applyActivation(acc[2], activation, leakySlope);
        out[o + 3] = applyActivation(acc[3], activation, leakySlope);
      }
    }
  }
  return out;
}

export class CpuUpscaler {
  constructor(model) {
    this.backend = "cpu";
    this.model = model;
    this._layerWeights = model.manifest.layers.map(
      (layer) => new Float32Array(model.weights.buffer, model.weights.byteOffset + layer.byte_offset, layer.byte_length / 4),
    );
  }

  /**
   * Run ArtCNN on a normalized 1-channel Float32Array luma plane (width x height -> 2W x 2H).
   */
  upscaleLuma(yPlane, width, height) {
    const { manifest } = this.model;
    const layers = manifest.layers;
    let skip = null;
    let cur = null;

    for (let i = 0; i < layers.length; i++) {
      const layer = layers[i];
      const params = this._layerWeights[i];
      if (layer.role === "entry") {
        skip = convIn1(yPlane, width, height, params, layer.c_out / 4, layer.activation, manifest.leaky_slope);
        cur = skip;
      } else if (layer.role === "features_d2s") {
        const added = new Float32Array(cur.length);
        for (let j = 0; j < cur.length; j++) added[j] = cur[j] + skip[j];
        const feats = convDense(added, width, height, params, layer.c_in / 4, 1, "none", 0);
        const outW = width * 2;
        const outH = height * 2;
        const outY = new Float32Array(outW * outH);
        const base = manifest.input_skip ? bilinear2x(yPlane, width, height) : null;
        for (let y = 0; y < height; y++) {
          for (let x = 0; x < width; x++) {
            const fOff = (y * width + x) * 4;
            const ox = x * 2;
            const oy = y * 2;
            const i00 = oy * outW + ox;
            const i10 = i00 + 1;
            const i01 = (oy + 1) * outW + ox;
            const i11 = i01 + 1;
            outY[i00] = clamp01(feats[fOff] + (base ? base[i00] : 0));
            outY[i10] = clamp01(feats[fOff + 1] + (base ? base[i10] : 0));
            outY[i01] = clamp01(feats[fOff + 2] + (base ? base[i01] : 0));
            outY[i11] = clamp01(feats[fOff + 3] + (base ? base[i11] : 0));
          }
        }
        return outY;
      } else if (layer.kind === "conv_grouped") {
        cur = convGrouped(cur, width, height, params, layer.c_in / 4, layer.activation, manifest.leaky_slope);
      } else {
        cur = convDense(cur, width, height, params, layer.c_in / 4, layer.c_out / 4, layer.activation, manifest.leaky_slope);
      }
    }
    throw new Error("Model manifest is missing features_d2s layer.");
  }

  /**
   * Upscale RGBA bytes (width x height) to 2x RGBA Uint8ClampedArray.
   */
  upscaleRgba(rgba, width, height) {
    const { y, cb, cr } = rgbaToYCbCrBT709(rgba, width, height);
    const outY = this.upscaleLuma(y, width, height);
    const outCb = bilinear2x(cb, width, height);
    const outCr = bilinear2x(cr, width, height);
    return ycbcrToRgbaBT709(outY, outCb, outCr, width * 2, height * 2);
  }

  /**
   * Upscale a browser image/canvas/video source to ImageData.
   */
  async upscaleToImageData(source) {
    let imageData;
    if (source instanceof ImageData) {
      imageData = source;
    } else {
      const w = source.videoWidth || source.naturalWidth || source.width;
      const h = source.videoHeight || source.naturalHeight || source.height;
      const tmp = document.createElement("canvas");
      tmp.width = w;
      tmp.height = h;
      const ctx = tmp.getContext("2d");
      ctx.drawImage(source, 0, 0, w, h);
      imageData = ctx.getImageData(0, 0, w, h);
    }
    const outRgba = this.upscaleRgba(imageData.data, imageData.width, imageData.height);
    return new ImageData(outRgba, imageData.width * 2, imageData.height * 2);
  }

  renderToCanvas(source, canvas, options = {}) {
    const split = options.split ?? 0;
    const w = source.videoWidth || source.naturalWidth || source.width;
    const h = source.videoHeight || source.naturalHeight || source.height;
    const outW = w * 2;
    const outH = h * 2;
    if (canvas.width !== outW || canvas.height !== outH) {
      canvas.width = outW;
      canvas.height = outH;
    }
    const tmp = document.createElement("canvas");
    tmp.width = w;
    tmp.height = h;
    const tmpCtx = tmp.getContext("2d");
    tmpCtx.drawImage(source, 0, 0, w, h);
    const srcData = tmpCtx.getImageData(0, 0, w, h);
    const outRgba = this.upscaleRgba(srcData.data, w, h);
    const ctx = canvas.getContext("2d");
    const upCanvas = document.createElement("canvas");
    upCanvas.width = outW;
    upCanvas.height = outH;
    upCanvas.getContext("2d").putImageData(new ImageData(outRgba, outW, outH), 0, 0);
    this._lastSrc = tmp;
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
