// SPDX-FileCopyrightText: 2026 perennialtech contributors
// SPDX-License-Identifier: MIT

/**
 * ArtCNN browser runtime entrypoint.
 */

import { CpuUpscaler } from "./cpu.js";
import { WebGPUUpscaler } from "./webgpu.js";

export { CpuUpscaler } from "./cpu.js";
export { WebGPUUpscaler } from "./webgpu.js";
export { OnnxLumaUpscaler } from "./onnx.js";
export { bilinear2x, float16ToFloat32, rgbaToYCbCrBT709, ycbcrToRgbaBT709 } from "./color.js";

let _supportPromise = null;

export function isWebGPUAvailable() {
  return typeof navigator !== "undefined" && Boolean(navigator.gpu);
}

/**
 * Probe WebGPU availability and optional `shader-f16` support on the active adapter.
 */
export async function detectWebGPUSupport(options = {}) {
  if (!isWebGPUAvailable()) {
    return { available: false, f16: false };
  }
  if (!options.forceRefresh && _supportPromise) {
    return _supportPromise;
  }
  _supportPromise = (async () => {
    try {
      const adapter = await navigator.gpu.requestAdapter({
        powerPreference: options.powerPreference || "high-performance",
      });
      if (!adapter) {
        return { available: false, f16: false };
      }
      return { available: true, f16: adapter.features.has("shader-f16") };
    } catch {
      return { available: false, f16: false };
    }
  })();
  return _supportPromise;
}

/**
 * Load an exported ArtCNN browser bundle (<model>.json + .f16.bin or .bin).
 *
 * When `options.precision` is "auto" (default), probes WebGPU `shader-f16` support
 * first and downloads only the half-size `.f16.bin` file when supported.
 */
export async function loadModel(manifestUrl, options = {}) {
  const fetcher = options.fetch || fetch;
  const manifestRes = await fetcher(manifestUrl);
  if (!manifestRes.ok) {
    throw new Error(`Failed to fetch model manifest (${manifestRes.status}): ${manifestUrl}`);
  }
  const manifest = await manifestRes.json();
  if (manifest.format !== "artcnn-webgpu") {
    throw new Error(`Unsupported model manifest format: ${manifest.format}`);
  }

  let requestedPrecision = options.precision || "auto";
  if (requestedPrecision === "auto") {
    const support = await detectWebGPUSupport(options);
    requestedPrecision = support.f16 && manifest.weights_f16_file ? "f16" : "f32";
  }

  const useF16 = requestedPrecision === "f16" && Boolean(manifest.weights_f16_file);
  const weightsFile = useF16 ? manifest.weights_f16_file : manifest.weights_file;
  const expectedBytes = useF16 ? manifest.weights_f16_bytes : manifest.weights_bytes;

  const weightsUrl = new URL(weightsFile, new URL(manifestUrl, globalThis.location?.href || "http://localhost/"));
  const weightsRes = await fetcher(weightsUrl);
  if (!weightsRes.ok) {
    throw new Error(`Failed to fetch model weights (${weightsRes.status}): ${weightsUrl}`);
  }
  const buf = await weightsRes.arrayBuffer();
  if (buf.byteLength !== expectedBytes) {
    throw new Error(`Weight size mismatch: expected ${expectedBytes} bytes, got ${buf.byteLength}`);
  }
  return {
    manifest,
    weights: new Uint8Array(buf),
    precision: useF16 ? "f16" : "f32",
  };
}

/**
 * Create a browser upscaler using WebGPU when available (or falling back to CPU).
 *
 * @param {object} options
 * @param {string} [options.modelUrl] URL to a <model>.json manifest.
 * @param {object} [options.model] Preloaded { manifest, weights, precision } object from loadModel().
 * @param {"auto"|"webgpu"|"cpu"} [options.backend="auto"] Preferred execution backend.
 * @param {"auto"|"f16"|"f32"} [options.precision="auto"] Preferred weight & compute precision.
 */
export async function createUpscaler(options) {
  const backend = options.backend || "auto";
  const precision = options.precision || (backend === "cpu" ? "f32" : "auto");
  const model = options.model || (await loadModel(options.modelUrl, { ...options, precision }));

  if (backend === "cpu") {
    return new CpuUpscaler(model);
  }
  if (backend === "webgpu") {
    return WebGPUUpscaler.create(model, options);
  }
  if (isWebGPUAvailable()) {
    try {
      return await WebGPUUpscaler.create(model, options);
    } catch {
      return new CpuUpscaler(model);
    }
  }
  return new CpuUpscaler(model);
}
