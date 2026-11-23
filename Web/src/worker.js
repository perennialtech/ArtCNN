// SPDX-FileCopyrightText: 2026 perennialtech contributors
// SPDX-License-Identifier: MIT

import { createUpscaler } from "./index.js";

let upscaler = null;

self.onmessage = async (event) => {
  const msg = event.data;
  try {
    if (msg.type === "init") {
      if (upscaler) {
        upscaler.destroy();
        upscaler = null;
      }
      upscaler = await createUpscaler({
        modelUrl: msg.modelUrl,
        backend: msg.backend || "auto",
        precision: msg.precision || "auto",
      });
      self.postMessage({
        id: msg.id,
        type: "ready",
        backend: upscaler.backend,
        precision: upscaler.precision,
      });
      return;
    }

    if (msg.type === "upscale") {
      if (!upscaler) throw new Error("Worker upscaler is not initialized.");
      const rgba = new Uint8ClampedArray(msg.buffer);
      const out = await upscaler.upscaleRgba(rgba, msg.width, msg.height, {
        scale: msg.scale ?? 2,
        tileSize: msg.tileSize ?? 256,
        overlap: msg.overlap,
        onProgress: (progress) => {
          self.postMessage({ id: msg.id, type: "progress", progress });
        },
      });
      self.postMessage(
        {
          id: msg.id,
          type: "result",
          buffer: out.buffer,
          width: msg.width * (msg.scale ?? 2),
          height: msg.height * (msg.scale ?? 2),
        },
        [out.buffer],
      );
      return;
    }

    if (msg.type === "destroy") {
      upscaler?.destroy();
      upscaler = null;
      self.postMessage({ id: msg.id, type: "destroyed" });
    }
  } catch (err) {
    self.postMessage({
      id: msg.id,
      type: "error",
      message: err?.message || String(err),
    });
  }
};
