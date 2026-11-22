# ArtCNN CPU models

These FP32 ONNX models accept luma and produce 2× luma output. They run with the CPU backends in the [inferencer](../../Inferencer/README.md#cpu-inference).

| Model | Workload |
| --- | --- |
| `ArtCNN_C3F8_Fast.onnx` | Lowest latency among the released models |
| `ArtCNN_C4F16_Clean.onnx` | 16-channel model for clean sources |
| `ArtCNN_C4F16_Restore.onnx` | 16-channel model for blurred or compressed sources |
| `ArtCNN_C4F32_Restore.onnx` | 32-channel model for restoration |
| `ArtCNN_R8F64_Rank16.onnx` | Faster R-series model with 16-channel projections |
| `ArtCNN_R8F64_Rank32_Clean.onnx` | R-series model with 32-channel projections tuned for clean sources |

The C-series weights are FP32 ONNX exports of the checkpoints from [the ArtCNN GPU exploration](https://github.com/perennialtech/ArtCNN/pull/1). The R-series models initialize pointwise projections and spatial convolutions from upstream R8F64 using SVD, then distill on separate ATD-12K training frames.

```shell
python Inferencer/inferencer.py frames/ -m ONNX/CPU/ArtCNN_C4F16_Restore.onnx \
  --cpu-backend openvino --cpu-threads 4
```

See the [CPU benchmark](../../Inferencer/benchmarks/README.md) for the pinned inputs and commands and the [measured results](../../Inferencer/benchmarks/results.md) for the speed and reconstruction trade-offs.
