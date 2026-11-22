# ArtCNN image inferencer

The CLI runs an ONNX model on a single image or a directory. It writes PNGs to the working directory and reuses each model session across frames and self-ensemble orientations. PNG compression can overlap inference on the next frame.

## CPU inference

Run the commands from the repository root. Start with [C4F16 Clean](../ONNX/CPU/ArtCNN_C4F16_Clean.onnx) for clean luma inputs or [C4F16 Restore](../ONNX/CPU/ArtCNN_C4F16_Restore.onnx) for blurred or compressed inputs.

```sh
python -m pip install numpy pillow opencv-python-headless onnxruntime
python Inferencer/inferencer.py image.png --model ONNX/CPU/ArtCNN_C4F16_Clean.onnx --cpu
```

The output in this example is `image_ArtCNN_C4F16_Clean_luma.png`. The default task is `luma`. Pass a directory to process a set of images with the same model. Automatic backend selection prefers DirectML and falls back to ONNX Runtime CPU.

### OpenVINO

Install the optional OpenVINO package for its FP32 CPU kernels. Both runtimes load the same ONNX models. Set `--cpu-threads` to tune inference for the available cores.

```sh
python -m pip install openvino
python Inferencer/inferencer.py frames/ --model ONNX/CPU/ArtCNN_C4F16_Restore.onnx \
  --cpu-backend openvino --cpu-threads 4
```

### Tiled ONNX Runtime

For large frames, install the optional `onnx` package and specify a tile edge in **input pixels**. The inferencer computes the required overlap from the model graph. Tiling uses ONNX Runtime on dynamic image models with spatially local operators. In this mode, `--cpu-threads` sets the number of tile workers, each with one inference thread.

```sh
python -m pip install onnx
python Inferencer/inferencer.py frames/ --model ONNX/CPU/ArtCNN_C4F32_Restore.onnx \
  --cpu-tile 160 --cpu-threads 4
```

See the [CPU model catalog](../ONNX/CPU/README.md) and [benchmark results](benchmarks/results.md) for measured speed, reconstruction quality, and peak memory.

## Color and chroma examples

Double RGB resolution with the upstream RGB model.

```sh
python Inferencer/inferencer.py image.png \
  --model ONNX/Experiments/ArtCNN_R8F64_RGB.onnx --task rgb
```

Combine the fork's luma model with upstream chroma reconstruction.

```sh
python Inferencer/inferencer.py image.png --model ONNX/CPU/ArtCNN_C4F16_Clean.onnx \
  --chroma-model ONNX/ArtCNN_R8F64_Chroma.onnx --task luma-chroma
```

Clean artifacts from a 4:2:0 JPEG.

```sh
python Inferencer/inferencer.py image.jpg --model ONNX/ArtCNN_R8F64_JPEG420.onnx --task rgb
```

## Command reference

Run `python Inferencer/inferencer.py --help` for the current options and tasks. Add `--self-ensemble` to average predictions across eight orientations.

Run the CPU regression suite from the repository root with `python -m unittest discover -s Inferencer/tests`. The tiling and OpenVINO tests run when their optional packages are installed.
