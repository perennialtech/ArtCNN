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
export { bilinear2x, rgbaToYCbCrBT709, ycbcrToRgbaBT709 } from "./color.js";

export function isWebGPUAvailable() {
  return typeof navigator !== "undefined" && Boolean(navigator.gpu);
}

/**
 * Load an exported ArtCNN browser bundle (<model>.json + <model>.bin).
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
  const weightsUrl = new URL(manifest.weights_file, new URL(manifestUrl, globalThis.location?.href || "http://localhost/"));
  const weightsRes = await fetcher(weightsUrl);
  if (!weightsRes.ok) {
    throw new Error(`Failed to fetch model weights (${weightsRes.status}): ${weightsUrl}`);
  }
  const buf = await weightsRes.arrayBuffer();
  if (buf.byteLength !== manifest.weights_bytes) {
    throw new Error(`Weight size mismatch: expected ${manifest.weights_bytes} bytes, got ${buf.byteLength}`);
  }
  return { manifest, weights: new Uint8Array(buf) };
}

/**
 * Create a browser upscaler using WebGPU when available (or falling back to CPU).
 *
 * @param {object} options
 * @param {string} [options.modelUrl] URL to a <model>.json manifest.
 * @param {object} [options.model] Preloaded { manifest, weights } object from loadModel().
 * @param {"auto"|"webgpu"|"cpu"} [options.backend="auto"] Preferred execution backend.
 */
export async function createUpscaler(options) {
  const model = options.model || (await loadModel(options.modelUrl, options));
  const backend = options.backend || "auto";

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
