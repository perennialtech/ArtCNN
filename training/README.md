# Clean-still RGB training & Pareto export

This directory trains, evaluates, and exports the Pareto-optimal clean-still RGB models shipped in [`Web/models/`](../Web/models/).

## Architecture families

- **C-series (`ArtCNNRGBConfig(family="c")`)**: 3-channel RGB input → `conv0` → `N` dense `3×3` layers with per-channel `PReLU` → `mix_global + conv0` → `features (F→12)` → `PixelShuffle(2)` + `2×` bilinear RGB skip.
- **R-series (`ArtCNNRGBConfig(family="r")`)**: Residual blocks (`conv3×3 → PReLU → conv3×3 + residual`) between `conv0` and `mix_global`, giving higher clean-still fidelity at larger compute budgets.

During training, internal `3×3` layers use lossless 5-branch `FoldableECB` reparameterization (`--ecb`) and FP16 straight-through quantization-aware training (`--qat-frac 0.25`). Before export, `FoldableECB` collapses into a single standard `3×3` convolution with zero runtime overhead.

## Training data & clean downscales

Models are trained on clean high-resolution animation frames from ATD-12K (excluding `Hayao_0` holdout) using clean multi-kernel `2×` downscales (`AREA`, `BICUBIC`, `BILINEAR`, and `LANCZOS-3`) with Charbonnier L1, Sobel edge, 2D FFT magnitude, SSIM, and DISTS perceptual losses—without blur, codec compression, or temporal shimmer terms.

## Commands

Train a model on GPU:

```sh
PYTHONPATH=training uv run --locked --group training python training/train_clean_stills.py \
  --train-manifest data/train-manifest.json --train-frames data/frames \
  --val-manifest data/holdout-manifest.json --val-frames data/frames \
  --family c --filters 32 --blocks 4 --steps 8000 --ecb \
  --output training/checkpoints/ArtCNN_C4F32_Clean.npz
```

Evaluate all checkpoints on held-out clean stills and mark the Pareto frontier:

```sh
PYTHONPATH=training uv run --locked --group training python training/eval_pareto.py \
  --ckpt-dir training/checkpoints \
  --holdout-manifest data/holdout-manifest.json --frames-dir data/frames \
  --output training/pareto_results.json
```

Export browser bundles (`Web/models/*.json`, `*.bin`, `*.f16.bin`, and `catalog.json`):

```sh
PYTHONPATH=training uv run --locked python training/export_web.py
```
