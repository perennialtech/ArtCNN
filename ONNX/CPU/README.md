# CPU models

These FP32 ONNX models accept luma and produce 2× luma output. Use the [inferencer](../../Inferencer/README.md) with ONNX Runtime or OpenVINO.

| Model | Intended use |
| --- | --- |
| [C3F8 Fast](ArtCNN_C3F8_Fast.onnx) | Lowest measured latency among these models |
| [C4F16 Clean](ArtCNN_C4F16_Clean.onnx) | Clean sources at C-series speed |
| [C4F16 Restore](ArtCNN_C4F16_Restore.onnx) | Blurred or compressed sources at C-series speed |
| [C4F32 Restore](ArtCNN_C4F32_Restore.onnx) | Wider model for restoration |
| [R8F64 Rank16](ArtCNN_R8F64_Rank16.onnx) | Lower-cost R-series model with 16-channel projections |
| [R8F64 Rank32 Clean](ArtCNN_R8F64_Rank32_Clean.onnx) | R-series model with 32-channel projections tuned for clean sources |

Run from the repository root with the optional OpenVINO group. See [CPU inference setup](../../Inferencer/README.md#cpu-inference) for the other backends.

```sh
uv run --locked --group openvino python Inferencer/inferencer.py frames/ --model ONNX/CPU/ArtCNN_C4F16_Restore.onnx \
  --cpu-backend openvino --cpu-threads 4
```

The C-series models use FP32 ONNX exports of the [GPU training checkpoints](../../enhanced/pareto/README.md). The R-series models use SVD to initialize pointwise projections and spatial convolutions from upstream R8F64, then distill on separate ATD-12K training frames.

See the [measured results](../../Inferencer/benchmarks/results.md) for speed, PSNR, and memory. The [benchmark guide](../../Inferencer/benchmarks/README.md) pins the inputs and commands. Downloaded weights can be checked against [SHA256SUMS](SHA256SUMS).
