# ArtCNN CPU benchmark

The upstream control is Artoriuz/ArtCNN at `0837e5326e710f49e2ab0a8d995da0708047442e`. The fork's original `Inferencer/` and `ONNX/` trees match this revision. See the [measured results](results.md) for the CPU comparisons.

## Inputs and measurements

- `holdout.json` pins 96 original [ATD-12K](https://huggingface.co/datasets/ProgramYourFace/ATD-12k) animation frames across five source groups. Training and calibration use separate source groups. Four of the five evaluation groups provided different frames for development. The benchmark verifies every input's SHA-256 hash.
- `unseen-holdout.json` contains 48 additional frames from four groups absent from training, development, and the primary evaluation. Eight are also evaluated as whole 960×540 references with `--limit 8 --full-size 960 540`.
- Latency is measured after warm-up on whole luma images resized to 960×540, 1280×720 and, for selected comparisons, 1920×1080. The image inputs are kept in memory. The benchmark shuffles model order each round and records every sample.
- Reconstruction quality uses one deterministic 512×288 crop per original frame. Each crop is degraded to 256×144 in four ways. The conditions are area downsampling, blur followed by area downsampling, area downsampling and H.264 at QP 32, and bicubic downsampling and HEVC at QP 34. PSNR and SSIM use the crop interior. Full-crop PSNR, edge error and flat-area error are also recorded.
- Runtime tests use CPU inference throughout. Both ONNX Runtime and OpenVINO run the FP32 models. Timing and quality use the same model bytes. An unmodified upstream CLI run is the control for wall time, output PNGs, and peak process RSS.

## Reproduce

Install the [inference dependencies](../README.md#cpu-inference), plus `scipy` and `scikit-image` for quality scoring. Quality scoring also requires FFmpeg with `libx264` and `libx265`.

From the repository root, point `--upstream-engine-dir` at the pinned upstream checkout's `Inferencer/` directory to time its unmodified `Engine` for the `ort` entries.

```shell
python Inferencer/benchmarks/performance.py \
  --manifest Inferencer/benchmarks/holdout.json --fetch \
  --upstream-engine-dir /path/to/upstream/Inferencer \
  --sizes 960x540 1280x720 --samples 6 --rounds 4 \
  --models ONNX/ArtCNN_C4F16.onnx:ort,ort-4,ov-4,tile160x4 \
           ONNX/ArtCNN_C4F32.onnx:ort,ort-4,ov-4,tile160x4 \
  --output cpu-performance.jsonl

python Inferencer/benchmarks/quality.py \
  --manifest Inferencer/benchmarks/holdout.json --fetch \
  --models bicubic ONNX/ArtCNN_C4F16.onnx ONNX/ArtCNN_C4F32.onnx \
  --output cpu-quality.jsonl
```
