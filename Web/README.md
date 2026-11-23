# ArtCNN for the Browser (WebGPU)

Zero-dependency ES module runtime and interactive workbench for **2× and 4× upscaling of clean still images** in modern browsers using WebGPU compute shaders (with automatic `shader-f16` half-precision downloads when supported), overlap tiling for bounded VRAM, off-main-thread Web Worker execution, and a `Float32Array` CPU fallback.

## Quick start

Serve the repository root locally and open `http://localhost:8000/`:

```sh
python3 -m http.server 8000
```

Drop, open, or paste (`Ctrl+V`) any clean still image (PNG, WebP, JPEG, or AVIF) to upscale it 2× or 4× with a live split comparison, 100%/200% pixel-peeping zoom, and lossless PNG export.

## JavaScript API

### Main-thread WebGPU / CPU upscaler

```js
import { createUpscaler } from "./Web/src/index.js";

const upscaler = await createUpscaler({
  modelUrl: "./Web/models/ArtCNN_C4F32_Clean.json",
  backend: "auto",   // "auto" | "webgpu" | "cpu"
  precision: "auto", // "auto" | "f16" | "f32"
});

// Upscale an Image, ImageBitmap, or ImageData with overlap tiling (bounded VRAM):
const outImageData = await upscaler.upscaleToImageData(sourceImage, {
  scale: 2,        // 2 or 4
  tileSize: 256,
  onProgress: (p) => console.log(`${Math.round(p * 100)}%`),
});

// Or encode directly to a lossless PNG blob:
const pngBlob = await upscaler.upscaleToBlob(sourceImage, { scale: 2, type: "image/png" });

// Release GPU buffers when done:
upscaler.destroy();
```

### Off-main-thread Web Worker upscaler

```js
import { createWorkerUpscaler } from "./Web/src/index.js";

const worker = await createWorkerUpscaler({
  modelUrl: "./Web/models/ArtCNN_R4F32_Clean.json",
});

const outRgba = await worker.upscaleRgba(imageData.data, imageData.width, imageData.height, {
  scale: 2,
  tileSize: 256,
  onProgress: (p) => updateProgressBar(p),
});

worker.destroy();
```

## Regenerating browser bundles

```sh
PYTHONPATH=training uv run --locked python training/export_web.py
```

## Running tests

```sh
node --test Web/tests/*.mjs
PYTHONPATH=training uv run --locked python -m unittest training/tests/test_export.py
```
