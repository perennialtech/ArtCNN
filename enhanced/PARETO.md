# Real-frame ArtCNN shaders

See the [shader catalog](../GLSL/Optimized/README.md) for playback setup and model selection. This page covers the measurements and how to reproduce them.

The stock-weight variants combine pixel shuffle with the last convolution and use explicit fused multiply-adds. FMA changes the order of rounding in the shader. The exact variants use only pass fusion and workgroup changes. The trained variants use real animation frames and include compact dense and grouped convolutions. A grouped block runs a 3×3 convolution within each four-channel input group and mixes the resulting channels with a 1×1 convolution in the same pass.

The trainer can load upstream C-series checkpoints or train a smaller student with a frozen upstream C- or R-series teacher. It samples patches from 1,483 ATD-12K training frames, uses clean area downscaling, Gaussian blur, and x264-compressed inputs, and selects checkpoints on separate development crops from ATD-12K, Morevna, and Sol Levante.

## Held-out results

The benchmark scores 25 model and scaler choices on 96 held-out frames with clean area downscaling, blur, H.264, and bicubic plus HEVC. Throughput is the median of three 1080p → 2160p repeats on an RTX 3080. Quality values are paired mean differences in luma PSNR.

Each stock C-series shader is dominated in each of the four measured conditions. For every stock point, at least one tested choice renders faster at equal or higher mean PSNR. The compact models occupy faster points between stock neural shaders and Spline36.

| Shader | RTX 3080 FPS | Comparison |
| --- | ---: | --- |
| `ArtCNN_C4F16_FusedFMA` | 325.2 | Stock C4F16 reached 256.6 FPS. The largest change in mean PSNR was 0.010 dB. |
| `ArtCNN_C4F32_FusedFMA` | 117.7 | Stock C4F32 reached 41.0 FPS. Mean PSNR differed by less than 0.01 dB in each condition. |
| `ArtCNN_C4F16_Grouped` | 437.9 | Against stock C4F16, blur improved by 0.87 dB and clean input lost 1.46 dB. |
| `ArtCNN_C3F8_Fast` | 508.0 | Spline36 reached 578.6 FPS. The compact model gained 1.78 dB on clean input and 0.60 dB on H.264. |
| `ArtCNN_C4F32_Restore` | 117.5 | The strongest stock H.264 result came from C4F32 DS at 40.9 FPS. Restore improved H.264 PSNR by 0.09 dB and blur PSNR by 0.59 dB. |

The exact C4F16 and C4F32 versions reached 270.2 and 50.7 FPS. The corresponding DS exact versions reached 271.1 and 50.6 FPS. All four matched stock 16-bit luma on every scored frame. All six FMA presets stayed within 0.013 dB mean PSNR of the corresponding stock preset in every condition. [Frontier chart](pareto/frontier.png) · [Full measurements](pareto/frontier.json)

At 720p → 1440p, the FMA C4F32 reached 263.7 FPS versus 92.1 for stock. The FMA C4F16 reached 533.5 FPS versus 520.6 for stock. The measured C4F16 improvement was larger at 4K. [720p repeat ranges](pareto/timing-720.json)

## Reproduce the data and run the benchmark

Use FFmpeg with Vulkan, libplacebo custom shaders, libx264, and libx265. The `training` group installs ONNX and the CUDA 13.4 PyTorch build on x86-64 Linux or Windows. See the [training requirements](README.md#running-tests-and-local-ablations) for supported Python versions and NVIDIA drivers. Run the commands in a scratch directory. `uv run --project` selects the checkout's locked environment and keeps scratch outputs in your working directory.

```bash
repo=/path/to/ArtCNN
for split in train dev holdout morevna sol-levante; do
    uv run --project "$repo" --locked python "$repo/enhanced/fetch_pareto_data.py" "$split" data
done
uv run --project "$repo" --locked python "$repo/enhanced/prepare_pareto_data.py" train-pairs data/train-manifest.json
uv run --project "$repo" --locked python "$repo/enhanced/prepare_pareto_data.py" dev-pairs \
    data/dev-manifest.json data/morevna-manifest.json data/sol-levante-manifest.json --limit 12
```

Generate an optimized copy of any of the six upstream neutral, DN, and DS C-series shaders with the same tool.

```bash
uv run --project "$repo" --locked python "$repo/enhanced/experiments/optimize_stock.py" \
    "$repo/GLSL/ArtCNN_C4F16.glsl" ArtCNN_C4F16_FusedFMA.glsl --fma
```

Regenerate a grouped shader from its checkpoint with the workgroup used in the benchmark.

```bash
uv run --project "$repo" --locked python "$repo/enhanced/export_pareto.py" \
    "$repo/enhanced/pareto/ArtCNN_C4F16_Grouped.npz" ArtCNN_C4F16_Grouped.glsl --workgroup 16 8 --fma
```

To train the grouped model from stock weights, use the real-frame pairs prepared above with CUDA PyTorch.

```bash
uv run --project "$repo" --locked --group training python "$repo/enhanced/train_pareto.py" --train train-pairs --dev dev-pairs --output grouped \
    --upstream "$repo/ONNX/ArtCNN_C4F16.onnx" --grouped --recipe balanced --steps 15000 \
    --evaluate-every 2500 --lr 0.00045 --seed 41004
```

Benchmark paired quality with the held-out manifest. The example also adds bicubic downsampling and HEVC compression as a codec and resampling condition absent from training.

```bash
bench="$repo/enhanced/experiments/bench_pareto.py"
shaders=(
  "upstream-f16=$repo/GLSL/ArtCNN_C4F16.glsl"
  "f16-fast=$repo/GLSL/Optimized/ArtCNN_C4F16_FusedFMA.glsl"
)
uv run --project "$repo" --locked --group benchmarks python "$bench" quality --stress --output quality --manifests data/holdout-manifest.json \
    --register "${shaders[@]}" --models spline36 upstream-f16 f16-fast
uv run --project "$repo" --locked --group benchmarks python "$bench" speed --repeats 3 --output timing --manifests data/sol-levante-manifest.json \
    --register "${shaders[@]}" --models spline36 upstream-f16 f16-fast
```

The timing command uses 1080p → 2160p input upload and Vulkan rendering with GPU-resident output. It starts measuring after warm-up. Use `speed-all` to also test 540p → 1080p and 720p → 1440p. Set `ARTCNN_VULKAN_DEVICE` to a substring of the Vulkan device name. It defaults to `NVIDIA`.

For quality runs on systems where libplacebo creates the Vulkan device, set `ARTCNN_INTERNAL_VULKAN=1` and set `ARTCNN_VULKAN_DEVICE` to the reported GPU name.

## Experiments that reached a limit

- The shader sweep tried nine workgroup shapes. The direct FMA kernel favored 8×8 on the dense stock C4F16 and 16×8 on the grouped C4F16.
- The Winograd trial reached 154 FPS in FP16 and 146 FPS in FP32 in 4K screening. Stock C4F16 reached 257 FPS in that screening round.
- Splitting FMA into partial sums changed the stock models' mean development PSNR by at most 0.02 dB versus direct FMA and lowered throughput at both widths.
- Three real-data training rounds and a 15-point checkpoint blend search were run. Additional R8F64 distillation improved the compact 8-channel candidate's development score. The corresponding 12-channel and grouped alternatives stayed near their earlier scores.

The data pipeline records source URLs and SHA-256 hashes. ATD-12K is by Siyao Li and collaborators. Morevna Episode 3.0.2 is by Morevna Project. Sol Levante is by Production I.G and Netflix. [ATD-12K](https://zenodo.org/records/12517088) · [Morevna](https://morevnaproject.org/anime/episode-3/) · [Sol Levante](https://opencontent.netflix.com/)
