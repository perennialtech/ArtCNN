# ArtCNN

ArtCNN upscales anime and animation. This fork provides optimized GLSL shaders for GPU playback, WebGPU bundles and a zero-dependency browser runtime, FP32 ONNX models for CPU image processing, and benchmarks on real animation frames. It builds on [Artoriuz/ArtCNN](https://github.com/Artoriuz/ArtCNN).

## Choose a model

The models below double luma resolution. Use the raw GLSL file in mpv, load the WebGPU bundle in a browser, or load the ONNX file with the Python inferencer.

| Source or goal | GPU playback (mpv) | Browser (WebGPU) | CPU image processing |
| --- | --- | --- | --- |
| Lowest inference cost | [C3F8 Fast](GLSL/Optimized/ArtCNN_C3F8_Fast.glsl) | [C3F8 Fast](Web/models/ArtCNN_C3F8_Fast.json) | [C3F8 Fast](ONNX/CPU/ArtCNN_C3F8_Fast.onnx) |
| Clean sources | [C4F16 Clean](GLSL/Optimized/ArtCNN_C4F16_Clean.glsl) | [C4F16 Clean](Web/models/ArtCNN_C4F16_Clean.json) | [C4F16 Clean](ONNX/CPU/ArtCNN_C4F16_Clean.onnx) |
| Blurred or compressed sources | [C4F16 Restore](GLSL/Optimized/ArtCNN_C4F16_Restore.glsl) | [C4F16 Restore](Web/models/ArtCNN_C4F16_Restore.json) | [C4F16 Restore](ONNX/CPU/ArtCNN_C4F16_Restore.onnx) |
| Higher detail on clean images | — | — | [R8F64 Rank32 Clean](ONNX/CPU/ArtCNN_R8F64_Rank32_Clean.onnx) |

The [optimized shader catalog](GLSL/Optimized/README.md) includes compact, grouped, restoration, and stock-weight variants. The [browser guide](Web/README.md) covers the WebGPU runtime, interactive demo, and `onnxruntime-web` helper. The [CPU model catalog](ONNX/CPU/README.md) covers all six CPU models.

## GPU playback with mpv

Start with [C4F16 FusedFMA](GLSL/Optimized/ArtCNN_C4F16_FusedFMA.glsl) for the neutral upstream preset with a faster shader. Add one ArtCNN shader to your mpv configuration or playback profile.

```ini
vo=gpu-next
glsl-shader="/path/to/ArtCNN/GLSL/Optimized/ArtCNN_C4F16_FusedFMA.glsl"
```

See the [shader catalog](GLSL/Optimized/README.md) for alternatives, including the Exact variants that matched stock 16-bit luma in the benchmark.

## Browser playback and upscaling (WebGPU)

Try the [live WebGPU demo](https://perennialtech.github.io/ArtCNN/) directly in your browser, or serve the repository root locally and open [`Web/demo/`](Web/demo/index.html) to upscale images or `<video>` files in real time with a live split comparison:

```sh
python3 -m http.server 8000
# Open http://localhost:8000/
```

To embed ArtCNN directly in a web app with zero dependencies (automatic WebGPU → CPU fallback):

```js
import { createUpscaler } from "./Web/src/index.js";

const upscaler = await createUpscaler({
  modelUrl: "./Web/models/ArtCNN_C4F16_Restore.json",
});
upscaler.renderToCanvas(videoOrImage, outputCanvas);
```

See the [browser guide](Web/README.md) for the full API, `onnxruntime-web` integration for `ONNX/` models, and bundle export commands.

## CPU image upscaling

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and run from the repository root. uv uses the repository's [Python version](.python-version) and [locked dependencies](uv.lock). This example uses Restore for compressed or blurred inputs and writes PNGs to the working directory.

```sh
uv run --locked --group openvino python Inferencer/inferencer.py frames/ --model ONNX/CPU/ArtCNN_C4F16_Restore.onnx \
  --cpu-backend openvino --cpu-threads 4
```

The [inferencer guide](Inferencer/README.md) covers ONNX Runtime, tiling for large images, and color and chroma tasks.

## Benchmarks

- **GPU playback.** At 1080p → 2160p on an RTX 3080, stock C4F32 reached 41.0 FPS and C4F32 FusedFMA reached 117.7 FPS. Mean luma PSNR differed by less than 0.01 dB in each of four degradation tests on 96 held-out frames. [GPU benchmark and reproduction](enhanced/PARETO.md)
- **CPU inference.** At 960×540 input on a Ryzen 5 5600X guest, C4F16 Restore took 33.7 ms versus 110.7 ms for stock C4F32 DN on four-thread OpenVINO. Restore improved mean PSNR in all four degradation tests on both evaluation sets. [CPU results](Inferencer/benchmarks/results.md) · [Reproduction](Inferencer/benchmarks/README.md)

## Upstream models and training

The [GLSL directory](GLSL/README.md) lists the original mpv shaders and experimental YCbCr shaders. The [ONNX directory](ONNX/README.md) describes the upstream luma, chroma, and JPEG models, along with VapourSynth integration.

[Training and export tools](enhanced/README.md) cover real-frame fine-tuning, portable checkpoints, and the earlier ablation experiments. See [dataset and metric scripts](Scripts/README.md) for the remaining command-line tools. The [upstream notebooks](Notebooks/) use framework-specific notebook runtimes. ArtCNN is distributed under the [MIT license](LICENSE).
