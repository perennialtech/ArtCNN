// SPDX-FileCopyrightText: 2026 perennialtech contributors
// SPDX-License-Identifier: MIT

/**
 * ArtCNN browser runtime entrypoint for Pareto-efficient clean still upscaling.
 */

import { CpuUpscaler } from "./cpu.js";
import { WebGPUUpscaler } from "./webgpu.js";

export { CpuUpscaler } from "./cpu.js";
export { WebGPUUpscaler } from "./webgpu.js";
export { planTiles } from "./tiling.js";
export { bilinear2xPlane, bilinear2xRgba, float16ToFloat32, rgbaToPlanarRgb } from "./color.js";

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
 * Load an exported ArtCNN RGB browser bundle (<model>.json + .f16.bin or .bin).
 */
export async function loadModel(manifestUrl, options = {}) {
  const fetcher = options.fetch || fetch;
  const manifestRes = await fetcher(manifestUrl);
  if (!manifestRes.ok) {
    throw new Error(`Failed to fetch model manifest (${manifestRes.status}): ${manifestUrl}`);
  }
  const manifest = await manifestRes.json();
  if (manifest.format !== "artcnn-webgpu-rgb") {
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

/**
 * Create an off-main-thread Web Worker upscaler with zero-copy buffer transfers and progress events.
 */
export async function createWorkerUpscaler(options) {
  if (typeof Worker === "undefined") {
    return createUpscaler(options);
  }
  const worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
  let nextId = 1;
  const pending = new Map();

  worker.onmessage = (event) => {
    const msg = event.data;
    const entry = pending.get(msg.id);
    if (!entry) return;
    if (msg.type === "progress") {
      entry.onProgress?.(msg.progress);
      return;
    }
    pending.delete(msg.id);
    if (msg.type === "error") {
      entry.reject(new Error(msg.message));
    } else {
      entry.resolve(msg);
    }
  };

  worker.onerror = (err) => {
    for (const [, entry] of pending) {
      entry.reject(new Error(err.message || "Worker error"));
    }
    pending.clear();
  };

  const call = (payload, transfer = [], onProgress = null) =>
    new Promise((resolve, reject) => {
      const id = nextId++;
      pending.set(id, { resolve, reject, onProgress });
      worker.postMessage({ ...payload, id }, transfer);
    });

  const ready = await call({
    type: "init",
    modelUrl: options.modelUrl,
    backend: options.backend || "auto",
    precision: options.precision || "auto",
  });

  return {
    backend: ready.backend,
    precision: ready.precision,
    async upscaleRgba(rgba, width, height, runOptions = {}) {
      const copy = new Uint8ClampedArray(rgba);
      const res = await call(
        {
          type: "upscale",
          buffer: copy.buffer,
          width,
          height,
          scale: runOptions.scale ?? 2,
          tileSize: runOptions.tileSize ?? 256,
          overlap: runOptions.overlap,
        },
        [copy.buffer],
        runOptions.onProgress,
      );
      return new Uint8ClampedArray(res.buffer);
    },
    destroy() {
      worker.terminate();
      pending.clear();
    },
  };
}
