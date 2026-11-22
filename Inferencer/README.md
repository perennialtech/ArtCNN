# ArtCNN image inferencer

The CLI runs an ONNX model on a single image or a directory. It writes PNGs to the working directory and reuses each model session across frames and self-ensemble orientations. PNG compression can overlap inference on the next frame.

## CPU inference

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and run the commands from the repository root. Inference requires Python 3.12 or newer. uv defaults to Python 3.12 and creates the project environment from [uv.lock](../uv.lock). Run `uv sync --locked` to install the default environment ahead of time. Start with [C4F16 Clean](../ONNX/CPU/ArtCNN_C4F16_Clean.onnx) for clean luma inputs or [C4F16 Restore](../ONNX/CPU/ArtCNN_C4F16_Restore.onnx) for blurred or compressed inputs.

```sh
uv run --locked python Inferencer/inferencer.py image.png --model ONNX/CPU/ArtCNN_C4F16_Clean.onnx --cpu
```

The output in this example is `image_ArtCNN_C4F16_Clean_luma.png`. The default task is `luma`. Pass a directory to process a set of images with the same model. The default environment includes ONNX Runtime CPU.

### OpenVINO

Select the `openvino` group for its FP32 CPU kernels. Both runtimes load the same ONNX models. Set `--cpu-threads` to tune inference for the available cores.

```sh
uv run --locked --group openvino python Inferencer/inferencer.py frames/ --model ONNX/CPU/ArtCNN_C4F16_Restore.onnx \
  --cpu-backend openvino --cpu-threads 4
```

### Tiled ONNX Runtime

For large frames, select the `onnx` group and specify a tile edge in **input pixels**. The inferencer computes the required overlap from the model graph. Tiling uses ONNX Runtime on dynamic image models with spatially local operators. In this mode, `--cpu-threads` sets the number of tile workers, each with one inference thread.

```sh
uv run --locked --group onnx python Inferencer/inferencer.py frames/ --model ONNX/CPU/ArtCNN_C4F32_Restore.onnx \
  --cpu-tile 160 --cpu-threads 4
```

See the [CPU model catalog](../ONNX/CPU/README.md) and [benchmark results](benchmarks/results.md) for measured speed, reconstruction quality, and peak memory.

### DirectML on Windows

On x86-64 Windows, use the `directml` group in a dedicated environment. Its ONNX Runtime wheel provides both DirectML and CPU execution, and the CLI prefers DirectML when available.

```powershell
$env:UV_PROJECT_ENVIRONMENT = ".venv-directml"
uv run --locked --no-group inference --group directml python Inferencer/inferencer.py image.png --model ONNX/CPU/ArtCNN_C4F16_Clean.onnx
Remove-Item Env:UV_PROJECT_ENVIRONMENT
```

## Color and chroma examples

Double RGB resolution with the upstream RGB model.

```sh
uv run --locked python Inferencer/inferencer.py image.png \
  --model ONNX/Experiments/ArtCNN_R8F64_RGB.onnx --task rgb
```

Combine the fork's luma model with upstream chroma reconstruction.

```sh
uv run --locked python Inferencer/inferencer.py image.png --model ONNX/CPU/ArtCNN_C4F16_Clean.onnx \
  --chroma-model ONNX/ArtCNN_R8F64_Chroma.onnx --task luma-chroma
```

Clean artifacts from a 4:2:0 JPEG.

```sh
uv run --locked python Inferencer/inferencer.py image.jpg --model ONNX/ArtCNN_R8F64_JPEG420.onnx --task rgb
```

## Convert ONNX models to FP16

The converter writes an FP16 copy of each ONNX file in the working directory. Run it from a directory containing the models you want to convert.

```sh
repo=/path/to/ArtCNN
uv run --project "$repo" --locked --group onnx python "$repo/Inferencer/convert_to_fp16.py"
```

## Command reference

Run `uv run --locked python Inferencer/inferencer.py --help` for the current options and tasks. Add `--self-ensemble` to average predictions across eight orientations.

Run the CPU regression suite from the repository root with both optional backends enabled.

```sh
uv run --locked --group onnx --group openvino python -m unittest discover -s Inferencer/tests
```
