# Local 3-Way Ablation Results (Procedural 6-Category Anime Benchmark)

Trained from scratch on CPU (pure NumPy, 160 steps, 48x48 HR patches) to isolate
zero-cost training improvements from low-cost architectural/shader upgrades.

| Metric | Baseline C4F16 | Zero-Cost C4F16 | Enhanced C4F16 |
| :--- | ---: | ---: | ---: |
| Clean Downscale PSNR (dB, higher better) | 20.59 | 18.80 | 21.30 |
| Clean Sobel Edge-MAE (lower better) | 0.0550 | 0.0859 | 0.0616 |
| Realistic Compressed+Halo PSNR (dB, higher better) | 19.88 | 18.34 | 21.24 |
| Realistic Compressed+Halo Edge-MAE (lower better) | 0.0637 | 0.0913 | 0.0644 |
| Flat-Fill Error Std (lower better) | 0.0482 | 0.0473 | 0.0102 |
| Temporal Shift Shimmer MAE (lower better) | 0.0222 | 0.0180 | 0.0017 |
| Conv MACs / input pixel | 12,240 | 12,240 | 12,240 |
| Exported GLSL Passes | 8 | 7 | 7 |
| Intermediate RGBA Textures | 7 | 6 | 6 |
