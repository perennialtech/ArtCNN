# Inferencer
`inferencer.py` runs ArtCNN ONNX models on images or directories. It reuses each model session across the directory and all eight self-ensemble orientations. PNG compression can run alongside inference on the next frame.

## Instructions
```shell
usage: inferencer.py [-h] [-m MODEL] [-cm CHROMA_MODEL] [-t {luma,rgb,ycbcr,chroma,luma-chroma,luma-luma-luma}]
                     [--self-ensemble] [--cpu | --cpu-backend {onnxruntime,openvino}]
                     [--cpu-threads CPU_THREADS] [--cpu-tile PIXELS]
                     input

ONNX Inferencer

positional arguments:
  input                 Input image or directory

options:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        ArtCNN Model
  -cm CHROMA_MODEL, --chroma-model CHROMA_MODEL
                        ArtCNN Chroma Model
  -t {luma,rgb,ycbcr,chroma,luma-chroma,luma-luma-luma}, --task {luma,rgb,ycbcr,chroma,luma-chroma,luma-luma-luma}
                        Task to perform
  --self-ensemble       Enable self-ensemble inference
  --cpu                 Use ONNX Runtime's CPU provider
  --cpu-backend {onnxruntime,openvino}
                        Select a CPU runtime
  --cpu-threads CPU_THREADS
                        Set the CPU inference thread count
  --cpu-tile PIXELS     Run ONNX Runtime CPU inference in overlapping tiles
```

## Examples

> [!IMPORTANT]
> The examples given below assume the ONNX models are in the inferencer's directory.

Luma upscaling using a luma model:
```shell
python inferencer.py image.png --model ArtCNN_R8F64.onnx --task luma
```

RGB upscaling using a RGB model:
```shell
python inferencer.py image.png --model ArtCNN_R8F64_RGB.onnx --task rgb
```

YCbCr upscaling using a luma model and a chroma model together:
```shell
python inferencer.py image.png --model ArtCNN_R8F64.onnx --chroma-model ArtCNN_R8F64_Chroma.onnx --task luma-chroma
```

JPEG denoising using a JPEG model:
```shell
python inferencer.py image.jpg --model ArtCNN_R8F64_JPEG420.onnx --task rgb
```

## CPU inference

Install the CPU dependencies with `python -m pip install numpy pillow opencv-python-headless onnxruntime`.
The CLI selects DirectML when available and ONNX Runtime CPU otherwise. Pass `--cpu` to select CPU explicitly. Use `--cpu-threads` to tune inference for the available cores.

```shell
python inferencer.py frames/ -m ../ONNX/ArtCNN_C4F16.onnx --cpu-threads 4
```

Install the optional `openvino` package to use its FP32 CPU kernels. Both runtimes load the same ONNX file.

```shell
python inferencer.py frames/ -m ../ONNX/ArtCNN_C4F16.onnx --cpu-backend openvino --cpu-threads 4
```

For large frames, install the optional `onnx` package and pass a tile edge in **input pixels**. The inferencer reads the model graph to compute the overlap each tile needs. Tiling supports dynamic ONNX images and spatially local ArtCNN operators. With tiling enabled, `--cpu-threads` sets the number of tile workers and each worker runs a single-threaded ONNX Runtime inference.

```shell
python inferencer.py frames/ -m ../ONNX/ArtCNN_C4F32.onnx --cpu-tile 160 --cpu-threads 4
```

Run the CPU regression suite from the repository root with `python -m unittest discover -s Inferencer/tests`. The tiling and OpenVINO tests run when their optional packages are installed.

See the [CPU model directory](../ONNX/CPU/README.md) and [benchmark results](benchmarks/results.md) for measured examples.
