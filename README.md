# ArtCNN

**Pareto-efficient 2× and 4× upscaling of clean still images in the browser.**

ArtCNN runs zero-dependency WebGPU compute shaders (with automatic `shader-f16` half-precision downloads, overlap tiling for bounded VRAM, off-main-thread Web Worker execution, and a `Float32Array` CPU fallback) directly on 3-channel RGB images. Models are trained strictly on clean multi-kernel downscales (`AREA`, `BICUBIC`, `BILINEAR`, `LANCZOS-3`) so clean lineart, cel fills, and digital illustrations upscale without restoration blur or codec assumptions.

## Browser Pareto frontier

Evaluated on 96 held-out clean 1080p RGB frames (`Hayao_0`) across clean 2× `area`, `bicubic`, and `lanczos4` downscales (`training/pareto_results.json`):

| Model | Family | FP16 size | MACs / input px | Mean PSNR | Mean SSIM | Mean LPIPS | Mean DISTS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Bilinear | Classical | — | — | 36.760 dB | 0.96647 | 0.07980 | 0.09972 |
| Bicubic | Classical | — | — | 38.042 dB | 0.97209 | 0.06235 | 0.07504 |
| Lanczos4 | Classical | — | — | 38.077 dB | 0.96988 | 0.06739 | 0.07202 |
| [C3F8 Clean](Web/models/ArtCNN_C3F8_Clean.json) | `C3F8` RGB | 6.8 KB | 3,384 | 38.708 dB | 0.97428 | 0.04286 | 0.05658 |
| [C4F16 Clean](Web/models/ArtCNN_C4F16_Clean.json) | `C4F16` RGB | 27 KB | 13,680 | 39.006 dB | 0.97604 | 0.03804 | 0.05084 |
| [C4F32 Clean](Web/models/ArtCNN_C4F32_Clean.json) | `C4F32` RGB | 99 KB | 50,400 | 39.384 dB | 0.97796 | 0.03262 | 0.04436 |
| [R6F48 Clean](Web/models/ArtCNN_R6F48_Clean.json) | `R6F48` RGB | 541 KB | 276,048 | 39.844 dB | 0.97993 | 0.02783 | 0.03717 |

Every tier is strictly non-dominated across compute, weight size, PSNR, SSIM, LPIPS, and DISTS.

## Browser demo & JavaScript API

Try the [live WebGPU demo](https://perennialtech.github.io/ArtCNN/) or serve the repository root locally:

```sh
python3 -m http.server 8000
# Open http://localhost:8000/
```

Drop, open, or paste (`Ctrl+V`) any clean still image to compare Bilinear vs. ArtCNN at 2× or 4× with 100%/200% zoom and lossless PNG export.

```js
import { createWorkerUpscaler } from "./Web/src/index.js";

const upscaler = await createWorkerUpscaler({
  modelUrl: "./Web/models/ArtCNN_C4F32_Clean.json",
});

const outRgba = await upscaler.upscaleRgba(imageData.data, imageData.width, imageData.height, {
  scale: 2,        // 2 or 4
  tileSize: 256,
  onProgress: (p) => console.log(`${Math.round(p * 100)}%`),
});

upscaler.destroy();
```

See [Web/README.md](Web/README.md) for the full browser API and [training/README.md](training/README.md) for training, Pareto evaluation, and bundle export.

## Tests

```sh
node --test Web/tests/*.mjs
PYTHONPATH=training python3 -m unittest training/tests/test_export.py
```

## Historical footnote

Earlier revisions of this repository (through commit [`a2560d0`](https://github.com/perennialtech/ArtCNN/tree/a2560d0), building on [Artoriuz/ArtCNN](https://github.com/Artoriuz/ArtCNN)) targeted real-time `mpv` GLSL video playback shaders, luma/chroma restoration models, and Python/ONNX CPU inference. Those assets and benchmarks remain preserved in Git history at [`a2560d0`](https://github.com/perennialtech/ArtCNN/tree/a2560d0).

ArtCNN is distributed under the [MIT license](LICENSE).
