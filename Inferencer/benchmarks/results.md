# Measured CPU results

Linux guest with eight vCPUs on Ryzen 5 5600X. ONNX Runtime 1.32.0, OpenVINO 2026.4.1, Python 3.12.3, and FFmpeg 6.1.1. The [benchmark instructions](README.md) pin the upstream commit and the frame inputs.

## Same-weight inference

Median warmed latency for 960×540 input and 1920×1080 output. The upstream column runs the unmodified upstream Engine with its CPU provider defaults. The OpenVINO column uses the same FP32 ONNX files with four threads.

| Stock model | Upstream ORT (ms) | OpenVINO (ms) | Speedup |
| --- | --- | --- | --- |
| C4F16 | 51.2 | 33.6 | 1.52× |
| C4F16_DN | 50.6 | 33.6 | 1.51× |
| C4F16_DS | 51.7 | 33.5 | 1.54× |
| C4F32 | 167.1 | 110.4 | 1.51× |
| C4F32_DN | 167.5 | 110.7 | 1.51× |
| C4F32_DS | 168.1 | 111.3 | 1.51× |
| R8F64 | 3323.4 | 2079.5 | 1.60× |

## Released models

The median timings use full 960×540 luma inputs. The four PSNR columns use 96 held-out original frames and equal evaluation inputs for every model. Each metric is the mean of per-frame PSNRs.

| Model | OpenVINO (ms) | Area (dB) | Blur (dB) | H.264 (dB) | HEVC (dB) |
| --- | --- | --- | --- | --- | --- |
| C3F8_Fast | 21.5 | 41.00 | 38.80 | 35.75 | 35.47 |
| C4F16_Clean | 33.6 | 43.10 | 38.90 | 36.21 | 35.68 |
| C4F16_Restore | 33.7 | 41.84 | 40.46 | 36.54 | 36.08 |
| C4F32_Restore | 111.0 | 42.20 | 40.62 | 36.79 | 36.22 |
| R8F64_Rank16 | 890.2 | 43.82 | 39.66 | 36.59 | 35.94 |
| R8F64_Rank32_Clean | 1416.8 | 44.60 | 39.84 | 36.48 | 35.81 |

On the 48 additional frames from source groups absent from training and development, C4F16 Restore improved every tested degradation over C4F32 DN. Its warmed latency was about one-third of C4F32 DN using the same runtime. The clean-focused rank-32 R8F64 improved all three degraded conditions and tracked the stock R8F64 clean score closely on the unseen groups.

| Model | Area (dB) | Blur (dB) | H.264 (dB) | HEVC (dB) |
| --- | --- | --- | --- | --- |
| C4F32_DN | 41.772 | 37.288 | 36.467 | 35.692 |
| C4F16_Restore | 42.901 | 40.489 | 36.875 | 36.256 |
| R8F64 | 45.513 | 38.540 | 36.114 | 35.500 |
| R8F64_Rank32_Clean | 45.508 | 39.876 | 36.451 | 35.743 |

## Directory inference and memory

Wall time includes Python startup, PNG decode, inference, and PNG encode. The C-series rows are medians of three repeats. The R8 rows each process the same two 1920×1080 frames once. Pixel error is the largest difference from the upstream CLI in 8-bit output values.

| Workload | Path | Wall time (s) | Peak RSS (MiB) | Max 8-bit error |
| --- | --- | --- | --- | --- |
| C4F16, 12 frames at 480×270 | Upstream | 0.79 | 136 | 0 |
| C4F16, 12 frames at 480×270 | Session reuse + PNG overlap | 0.71 | 106 | 0 |
| C4F16, 12 frames at 480×270 | ONNX Runtime, four threads | 0.51 | 106 | 0 |
| C4F16, 12 frames at 480×270 | OpenVINO, four threads | 0.64 | 165 | 1 |
| C4F16, three eight-way frames at 960×540 | Upstream | 2.40 | 351 | 0 |
| C4F16, three eight-way frames at 960×540 | Session reuse + PNG overlap | 1.76 | 233 | 0 |
| C4F16, three eight-way frames at 960×540 | ONNX Runtime, four threads | 1.63 | 233 | 0 |
| C4F16, three eight-way frames at 960×540 | OpenVINO, four threads | 1.36 | 273 | 1 |
| C4F32, two frames at 1920×1080 | Upstream | 2.55 | 963 | 0 |
| C4F32, two frames at 1920×1080 | Session reuse + PNG overlap | 2.12 | 937 | 0 |
| C4F32, two frames at 1920×1080 | ONNX Runtime, four threads | 1.88 | 937 | 0 |
| C4F32, two frames at 1920×1080 | OpenVINO, four threads | 1.68 | 1038 | 1 |
| C4F32, two frames at 1920×1080 | ONNX Runtime, four 160-pixel tile workers | 1.63 | 257 | 0 |
| R8F64, two frames at 1920×1080 | Upstream | 25.79 | 2268 | 0 |
| R8F64, two frames at 1920×1080 | Session reuse + PNG overlap | 25.18 | 2241 | 0 |
| R8F64, two frames at 1920×1080 | OpenVINO, four threads | 17.68 | 2314 | 1 |
| R8F64, two frames at 1920×1080 | ONNX Runtime, four 160-pixel tile workers | 30.31 | 464 | 0 |
| R8F64, two frames at 1920×1080 | ONNX Runtime, four 256-pixel tile workers | 27.63 | 728 | 0 |

Tiled ONNX Runtime and the reused ONNX Runtime path matched upstream PNGs in these runs. OpenVINO's largest difference was one output code value. The Windows 11 checks exercised both CPU backends on a real two-image directory.

## Other configurations

The ncnn wheel's fastest Winograd configuration was slower on this host. Decomposing the original grouped models performed worse than their equivalent dense graphs. Fixed-shape OpenVINO compilation improved some smaller cases and slowed C4F32 at 1920×1080. INT8 calibration covered signed, unsigned, reduced-range, and percentile variants. The strongest INT8 speed gains came with materially lower reconstruction quality, so the released models use FP32.
