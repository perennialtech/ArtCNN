# ArtCNN ONNX models

For CPU image upscaling, start with the [CPU model catalog](CPU/README.md) and the [inferencer guide](../Inferencer/README.md). The original Artoriuz/ArtCNN ONNX models are also available here.

## Upstream architectures

| Model | Architecture | Blocks or layers | Filters | Parameters |
| --- | --- | ---: | ---: | ---: |
| [C4F16](ArtCNN_C4F16.onnx) | C | 4 | 16 | ~12k |
| [C4F32](ArtCNN_C4F32.onnx) | C | 4 | 32 | ~48k |
| [R8F64](ArtCNN_R8F64.onnx) | R | 8 | 64 | ~926k |
| [R16F96](ArtCNN_R16F96.onnx) | R | 16 | 96 | ~4m |

The C-series uses a long skip connection and favors low inference cost. The R-series adds residual blocks and more filters for offline reconstruction. [C4F32 diagram](../Images/c4f32.png) · [R8F64 diagram](../Images/r8f64.png) · [Upstream examples](../Images/artcnn_examples.png)

## Upstream variants

- A model without a suffix doubles luma resolution with the neutral preset.
- `_DN` and `_DS` double luma resolution. DN denoises and softens. DS denoises and sharpens.
- `_Chroma` and `_Chroma_DN` reconstruct chroma for 4:2:0 BT.709 YCbCr. Upscale chroma with bilinear interpolation first. The DN variant also denoises.
- `_JPEG444` and `_JPEG420` clean compression artifacts on JPEGs with the corresponding chroma subsampling.
- [Experimental ONNX models](Experiments/) include other scales and color tasks.

## VapourSynth

Use [vs-mlrt](https://github.com/AmusementClub/vs-mlrt) and the ArtCNN wrapper in [vs-jetpack](https://github.com/Jaded-Encoding-Thaumaturgy/vs-jetpack/) for the supported upstream models. GLSL shaders can also be run through [vs-placebo](https://github.com/sgt0/vs-placebo).
