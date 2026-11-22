# ArtCNN for the Browser (WebGPU)

Zero-dependency ES module runtime and interactive demo for running ArtCNN 2× anime upscaling directly in modern browsers using WebGPU compute shaders, with an automatic `Float32Array` CPU fallback and an optional `onnxruntime-web` helper.

## Quick start

Open the [live GitHub Pages demo](https://perennialtech.github.io/ArtCNN/), or serve the repository root locally with any static HTTP server and open `http://localhost:8000/`:

```sh
python3 -m http.server 8000
```

Then visit `http://localhost:8000/` in a WebGPU-capable browser (Chrome, Edge, or recent Firefox/Safari). Drop any image or video file onto the viewer to upscale it in real time with a live bilinear vs. ArtCNN split comparison.

## Browser model catalog

The pre-exported bundles in [`Web/models/`](models/) pack each model's FP32 weights into a compact binary blob (`.bin`) plus a JSON manifest (`.json`):

| Model | Architecture | Weights | Intended use |
| --- | --- | ---: | --- |
| [`ArtCNN_C3F8_Fast`](models/ArtCNN_C3F8_Fast.json) | C3F8 | 11 KB | Lowest GPU cost; ideal for real-time video on integrated GPUs |
| [`ArtCNN_C4F16_Grouped`](models/ArtCNN_C4F16_Grouped.json) | C4F16 | 20 KB | Grouped 16-channel model for lower compute cost |
| [`ArtCNN_C4F16_Clean`](models/ArtCNN_C4F16_Clean.json) | C4F16 | 49 KB | Preserve fine detail in clean animation sources |
| [`ArtCNN_C4F16_Balanced`](models/ArtCNN_C4F16_Balanced.json) | C4F16 | 49 KB | Mixed clean, blurred, and compressed sources |
| [`ArtCNN_C4F16_Restore`](models/ArtCNN_C4F16_Restore.json) | C4F16 | 49 KB | Restore blurred or compressed sources |
| [`ArtCNN_C4F32_Restore`](models/ArtCNN_C4F32_Restore.json) | C4F32 | 187 KB | Wider 32-channel restoration model for dedicated GPUs |

## JavaScript API

### WebGPU (with automatic CPU fallback)

```js
import { createUpscaler } from "./Web/src/index.js";

const upscaler = await createUpscaler({
  modelUrl: "./Web/models/ArtCNN_C4F16_Restore.json",
  backend: "auto", // "auto" | "webgpu" | "cpu"
});

// Render an <img>, <video>, <canvas>, or ImageBitmap onto a 2x output canvas:
upscaler.renderToCanvas(sourceElement, outputCanvas, { split: 0 });

// Or read back upscaled RGBA pixels as ImageData:
const imageData = await upscaler.upscaleToImageData(sourceElement);

// Release GPU buffers when done:
upscaler.destroy();
```

### Using existing `.onnx` models with `onnxruntime-web`

ArtCNN's luma ONNX models in [`ONNX/`](../ONNX/) and [`ONNX/CPU/`](../ONNX/CPU/) expect a single BT.709 luma channel. `OnnxLumaUpscaler` handles the BT.709 RGB ↔ YCbCr conversion and 2× bilinear chroma upscaling for you:

```js
import * as ort from "onnxruntime-web";
import { OnnxLumaUpscaler } from "./Web/src/onnx.js";

const upscaler = await OnnxLumaUpscaler.create(ort, "./ONNX/CPU/ArtCNN_C4F16_Restore.onnx");
const outImageData = await upscaler.upscaleToImageData(inImageData);
```

## Regenerating browser bundles

Browser bundles are generated directly from the Pareto checkpoints in [`enhanced/pareto/`](../enhanced/pareto/) via [`enhanced/export_web.py`](../enhanced/export_web.py):

```sh
PYTHONPATH=enhanced uv run --locked python enhanced/export_web.py
```

## Running tests

```sh
node --test Web/tests/*.mjs
PYTHONPATH=enhanced uv run --locked python -m unittest enhanced/tests/test_webgpu_export.py
```
