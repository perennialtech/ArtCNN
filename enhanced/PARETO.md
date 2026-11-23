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

The exact C4F16 and C4F32 versions reached 270.2 and 50.7 FPS. The corresponding DS exact versions reached 271.1 and 50.6 FPS. All four matched stock 16-bit luma on every scored frame. All six FMA presets stayed within 0.013 dB mean PSNR of the corresponding stock preset in every condition. [PSNR frontier chart](pareto/frontier.png) · [LPIPS frontier chart](pareto/perceptual_lpips_frontier.png) · [DISTS frontier chart](pareto/perceptual_dists_frontier.png) · [Full measurements](pareto/frontier.json) · [Perceptual holdout](pareto/perceptual_holdout_v4.json)

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

## Cross-project Vulkan benchmark (RTX 3080)

All 30 shaders below were benchmarked apples-to-apples through FFmpeg's Vulkan `libplacebo` renderer on an RTX 3080 (3-repeat 1080p → 2160p median FPS on Sol Levante; 16-bit Vulkan `yuv420p16le` luma output on the 96-frame held-out `Hayao_0` set scored with AlexNet LPIPS and VGG DISTS). Full per-regime JSON lives in `enhanced/pareto/cross_project_benchmark.json`.

| Model | Family | 1080p→4K FPS | Area (LP / DI) | Blur (LP / DI) | H.264 (LP / DI) | HEVC (LP / DI) |
| --- | --- | ---: | --- | --- | --- | --- |
| `Spline36` | Classical | 578.6 | 0.0849 / 0.1043 | 0.1285 / 0.1410 | 0.1771 / 0.1670 | 0.1708 / 0.1625 |
| `ravu-lite-ar-r4` | [mpv-prescalers](https://github.com/bjin/mpv-prescalers) | 561.1 | 0.0310 / 0.0495 | 0.0842 / 0.1056 | 0.1301 / 0.1344 | 0.1267 / 0.1299 |
| `CuNNy-fast-DS` | [CuNNy](https://github.com/funnyplanter/CuNNy) | 530.4 | 0.0427 / 0.0633 | 0.0722 / 0.0929 | 0.1282 / 0.1305 | 0.1202 / 0.1233 |
| **`ArtCNN_C3F8_Fast`** | **Ours** | **513.0** | **0.0306 / 0.0459** | **0.0612 / 0.0795** | 0.1291 / 0.1364 | **0.1167 / 0.1210** |
| `Anime4K_CNN_x2_M` | [Anime4K v4.0.1](https://github.com/bloc97/Anime4K) | 483.9 | 0.0401 / 0.0538 | 0.0834 / 0.1000 | 0.1323 / 0.1325 | 0.1270 / 0.1279 |
| `CuNNy-4x16-DS-Q` (`dp4a`) | [CuNNy](https://github.com/funnyplanter/CuNNy) | 473.9 | 0.0443 / 0.0695 | 0.0718 / 0.0974 | 0.1264 / 0.1292 | 0.1197 / 0.1254 |
| **`ArtCNN_C3F12_Fast`** | **Ours** | **472.8** | **0.0291 / 0.0425** | **0.0610 / 0.0788** | **0.1252 / 0.1320** | **0.1157 / 0.1188** |
| `CuNNy-4x12-DS` | [CuNNy](https://github.com/funnyplanter/CuNNy) | 452.6 | 0.0418 / 0.0651 | 0.0722 / 0.0957 | 0.1276 / 0.1287 | 0.1209 / 0.1247 |
| `Anime4K_Denoise_L` | [Anime4K v4.0.1](https://github.com/bloc97/Anime4K) | 435.8 | 0.0526 / 0.0720 | 0.0939 / 0.1134 | 0.1367 / 0.1373 | 0.1331 / 0.1359 |
| `Anime4K_CNN_x2_L` | [Anime4K v4.0.1](https://github.com/bloc97/Anime4K) | 434.1 | 0.0416 / 0.0621 | 0.0852 / 0.1085 | 0.1332 / 0.1361 | 0.1284 / 0.1333 |
| **`ArtCNN_C4F16_Grouped`** | **Ours** | **417.4** | **0.0265 / 0.0389** | **0.0562 / 0.0757** | **0.1204 / 0.1276** | **0.1118 / 0.1156** |
| `nnedi3-nns32-win8x4` | [mpv-prescalers](https://github.com/bjin/mpv-prescalers) | 402.4 | 0.0581 / 0.0774 | 0.1111 / 0.1267 | 0.1479 / 0.1471 | 0.1484 / 0.1464 |
| `FSRCNNX_x2_8-0-4-1` | [FSRCNNX](https://github.com/igv/FSRCNN-TensorFlow/releases) | 390.9 | 0.0337 / 0.0549 | 0.0754 / 0.1033 | 0.1347 / 0.1412 | 0.1228 / 0.1312 |
| `CuNNy-4x16-DS` | [CuNNy](https://github.com/funnyplanter/CuNNy) | 322.4 | 0.0438 / 0.0687 | 0.0719 / 0.0970 | 0.1269 / 0.1289 | 0.1204 / 0.1255 |
| **`ArtCNN_C4F16_Restore`** | **Ours** | **321.7** | 0.0332 / 0.0455 | **0.0515 / 0.0677** | **0.1137 / 0.1170** | **0.1084 / 0.1092** |
| **`ArtCNN_C4F16_Clean`** | **Ours** | **320.7** | **0.0210 / 0.0306** | 0.0653 / 0.0867 | 0.1239 / 0.1304 | 0.1160 / 0.1198 |
| **`ArtCNN_C4F16_Balanced`** | **Ours** | **319.3** | 0.0245 / 0.0349 | 0.0541 / 0.0719 | 0.1174 / 0.1231 | 0.1101 / 0.1123 |
| `nnedi3-nns64-win8x4` | [mpv-prescalers](https://github.com/bjin/mpv-prescalers) | 266.6 | 0.0572 / 0.0767 | 0.1117 / 0.1273 | 0.1472 / 0.1461 | 0.1485 / 0.1462 |
| `Upstream ArtCNN_C4F16` | [Artoriuz/ArtCNN](https://github.com/Artoriuz/ArtCNN) | 256.6 | 0.0237 / 0.0350 | 0.0761 / 0.0956 | 0.1255 / 0.1295 | 0.1213 / 0.1235 |
| `Upstream ArtCNN_C4F16_DS` | [Artoriuz/ArtCNN](https://github.com/Artoriuz/ArtCNN) | 256.9 | 0.0559 / 0.0769 | 0.0792 / 0.1016 | 0.1309 / 0.1310 | 0.1241 / 0.1280 |
| **`ArtCNN_C6F16_Restore`** | **Ours** | **253.0** | 0.0357 / 0.0486 | 0.0534 / **0.0667** | 0.1150 / **0.1150** | 0.1088 / **0.1065** |
| `Anime4K_CNN_x2_VL` | [Anime4K v4.0.1](https://github.com/bloc97/Anime4K) | 244.7 | 0.0419 / 0.0606 | 0.0849 / 0.1080 | 0.1307 / 0.1326 | 0.1271 / 0.1315 |
| `Anime4K_Denoise_VL` | [Anime4K v4.0.1](https://github.com/bloc97/Anime4K) | 236.1 | 0.0515 / 0.0717 | 0.0891 / 0.1108 | 0.1328 / 0.1329 | 0.1292 / 0.1325 |
| `FSRCNNX_x2_16-0-4-1` | [FSRCNNX](https://github.com/igv/FSRCNN-TensorFlow/releases) | 177.7 | 0.0287 / 0.0487 | 0.0703 / 0.1015 | 0.1302 / 0.1364 | 0.1179 / 0.1278 |
| `CuNNy-4x32-DS-Q` (`dp4a`) | [CuNNy](https://github.com/funnyplanter/CuNNy) | 163.1 | 0.0416 / 0.0689 | 0.0671 / 0.0934 | 0.1236 / 0.1257 | 0.1170 / 0.1229 |
| **`ArtCNN_C4F32_Clean`** | **Ours** | **115.8** | **0.0187 / 0.0277** | 0.0632 / 0.0801 | 0.1208 / 0.1275 | 0.1149 / 0.1166 |
| **`ArtCNN_C4F32_Restore`** | **Ours** | **115.8** | 0.0311 / 0.0423 | **0.0489 / 0.0654** | **0.1122 / 0.1127** | **0.1064 / 0.1066** |
| `Upstream ArtCNN_C4F32` | [Artoriuz/ArtCNN](https://github.com/Artoriuz/ArtCNN) | 41.0 | 0.0224 / 0.0341 | 0.0784 / 0.0984 | 0.1254 / 0.1284 | 0.1222 / 0.1244 |
| `Upstream ArtCNN_C4F32_DS` | [Artoriuz/ArtCNN](https://github.com/Artoriuz/ArtCNN) | 40.9 | 0.0554 / 0.0828 | 0.0721 / 0.0983 | 0.1270 / 0.1264 | 0.1195 / 0.1242 |
| `CuNNy-4x32-DS` | [CuNNy](https://github.com/funnyplanter/CuNNy) | 8.4 | 0.0413 / 0.0682 | 0.0671 / 0.0929 | 0.1242 / 0.1256 | 0.1173 / 0.1229 |

- **Frontier dominance**: Below 520 FPS, zero Anime4K, FSRCNNX, NNEDI3, or upstream ArtCNN shaders survive on any LPIPS or DISTS Pareto frontier across the four regimes.
- **Future speed lead**: CuNNy's 8-bit `dp4a` shaders (`CuNNy-4x16-DS-Q` at 473.9 FPS vs. FP16 322.4 FPS, +47%) show that optional Vulkan `GL_EXT_integer_dot_product` export could accelerate our `C4F16` and `C4F32` tiers substantially with minimal quantization loss; tracked in [Vulkan `dp4a` INT8 export](https://github.com/perennialtech/ArtCNN/issues/17).

## Experiments that reached a limit

- **v5 6.5× dataset scaling (9,662 ATD-12K frames) + `ClipAnimeIQA` (`ViT-B-32` NR-IQA) + from-scratch `SPAN+Dil2+PReLU` ablation (`enhanced/pareto/perceptual_holdout_v5.json`)**:
  - Scaling the clean training corpus from 1,483 to 9,662 non-`Hayao_0` ATD-12K frames and adding differentiable `ClipAnimeIQA` (`--clip-iqa-weight`) improved both DISTS (across Blur, H.264, and HEVC) and `ClipAnimeIQA` on `ArtCNN_C4F16_Restore` (321.7 FPS; Blur/H.264/HEVC DISTS `0.0672 / 0.1169 / 0.1090` vs. v4 `0.0681 / 0.1174 / 0.1097`; `ClipAnimeIQA` `0.3287` vs. `0.3301`) and `ArtCNN_C6F16_Restore` (253.0 FPS; Blur/H.264/HEVC DISTS `0.0653 / 0.1144 / 0.1062` vs. v4 `0.0657 / 0.1146 / 0.1065`; `ClipAnimeIQA` `0.3266` vs. `0.3278`) at zero FPS overhead. Both are shipped in `GLSL/Optimized/`, `enhanced/pareto/`, `Web/models/`, and `ONNX/CPU/`.
  - All other catalog tiers (`C3F8_Fast`, `C3F12_Fast`, `C4F16_Grouped`, `C4F16_Clean`, `C4F16_Balanced`, `C4F32_Grouped`, `C4F32_Clean`, `C4F32_Restore`, `CPU_R8F64_Rank16`, `CPU_R8F64_Rank32_Clean`) stayed within `<= 0.0005` LPIPS/DISTS of v4 (or regressed slightly without soup guarding), showing that standard 2× synthetic + H.264/HEVC restoration on `ArtCNN-C` + `PReLU` has reached empirical saturation.
  - From-scratch 2-phase pretraining (15k steps + 3k perceptual FT) of `SPAN + dilation=2 + PReLU` (`C4F16_SPAN_Restore` at 305.6 FPS, `C6F16_SPAN_Restore` at 241.5 FPS, `C4F24_SPAN_Restore` at 180.2 FPS) achieved the lowest `ClipAnimeIQA` loss (`0.2995` on `C4F24_SPAN`), but remained dominated by warm-started `C4F16_Restore` (321.7 FPS) and `C6F16_Restore` (253.0 FPS) on full-reference 96-frame holdout PSNR/LPIPS/DISTS and throughput (~5% FPS overhead), so they were not promoted to the production catalog. Further real-world perceptual gains are tracked in [Issue #15](https://github.com/perennialtech/ArtCNN/issues/15) (non-integer descale, banding/deband, AV1/VP9).
- Switching `C2F12_Chroma` from a bilinear skip base to an exact 2× Catmull-Rom bicubic residual base with EMA (`GLSL/Optimized/ArtCNN_C2F12_Chroma.glsl`) raised 96-frame holdout quality past plain bicubic (`54.19` vs. `54.03 dB` Chroma PSNR; `51.16` vs. `51.00 dB` RGB PSNR; `0.00158` vs. `0.00162` RGB LPIPS; `0.00118` RGB DISTS), so it is now shipped in `GLSL/Optimized/`.
- Register-blocked (`2×1` / `2×2` pixels/thread) and vertically fused internal-layer GLSL shaders were benchmarked on RTX 3080 (`305` / `246` / `252 FPS` vs. `325 FPS` baseline C4F16) and dropped due to register-pressure occupancy loss; per-channel `PReLU` matched baseline FPS (`0%` cost) and strictly improved perceptual quality across all four regimes.

- The shader sweep tried nine workgroup shapes. The direct FMA kernel favored 8×8 on the dense stock C4F16 and 16×8 on the grouped C4F16.
- The Winograd trial reached 154 FPS in FP16 and 146 FPS in FP32 in 4K screening. Stock C4F16 reached 257 FPS in that screening round.
- Splitting FMA into partial sums changed the stock models' mean development PSNR by at most 0.02 dB versus direct FMA and lowered throughput at both widths.
- Three real-data training rounds and a 15-point checkpoint blend search were run. Additional R8F64 distillation improved the compact 8-channel candidate's development score. The corresponding 12-channel and grouped alternatives stayed near their earlier scores.

The data pipeline records source URLs and SHA-256 hashes. ATD-12K is by Siyao Li and collaborators. Morevna Episode 3.0.2 is by Morevna Project. Sol Levante is by Production I.G and Netflix. [ATD-12K](https://zenodo.org/records/12517088) · [Morevna](https://morevnaproject.org/anime/episode-3/) · [Sol Levante](https://opencontent.netflix.com/)
