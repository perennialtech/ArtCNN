# ArtCNN Enhanced

The [real-frame shaders and benchmark](PARETO.md) cover the optimized upstream presets and the separately trained compact, grouped, and restoration models.

Standalone, verified engineering implementation of zero-cost and low-cost perceived-quality upgrades for [Artoriuz/ArtCNN](https://github.com/Artoriuz/ArtCNN) (`C4F16` / `C4F32`).

## What's implemented

1. **Zero-cost training upgrades (same runtime MACs)**
   - **Realistic luma degradation (`artcnn/degradation.py`):** Multi-kernel downscaling (`AREA`, `LINEAR`, `CUBIC`, `LANCZOS4`), mild pre-blur, unsharp ringing halos, and custom 4x4 radial-table DCT compression (extending ArtCNN v1.6.2's Chroma codec model to Luma).
   - **Flat-fill-aware balanced sampler (`artcnn/benchmark_data.py`):** Stratifies patches by edge density and flat-region coverage instead of PNG filesize alone, protecting flat cel fills and smooth gradients.
   - **Blended perceptual/structural losses (`artcnn/losses.py`):** Charbonnier L1 + Sobel edge-agreement + 2D FFT high-frequency magnitude + teacher distillation + single-frame temporal-shift consistency (anti-shimmer during camera pans).
   - **Foldable 5-branch EdgeConvBlock (`artcnn/reparam.py`):** Trains with `3x3 + 1x1 + identity + Sobel-X + Sobel-Y` branches and collapses them losslessly into a single standard `3x3` convolution before export.
2. **Lean runtime & shader upgrades (`artcnn/glsl_export.py`)**
   - **Fused final pass (`fuse_depth_to_space=True`):** Writes the 2x2 output block directly from `conv2d_6` via 4 `imageStore` calls, removing the separate `depth_to_space` pass and an entire intermediate RGBA texture write/read (`8 -> 7` passes, `7 -> 6` intermediate textures).
   - **`LeakyReLU` (`max(result, result * F(0.1))`)** and **direct bilinear input residual skip (`LUMA_tex`)**.
3. **Consecutive-linear 3x3 -> 5x5 filter collapse verifier (`artcnn/linear_collapse.py`)**
   - Proves float64 interior equivalence (`< 1e-12`) while documenting exact zero-padded border divergence.
4. **Drop-in GPU training (`train_pytorch.py`)**
   - Ready to train full `C4F32` / `C4F16` models on real anime datasets (`Anime_Train_HR` / `Digital_Art_Train_HR`) and export fused `.glsl` shaders directly.

## Running tests and local ablations

From the repository root, set up an environment for the CPU tools and run the test suite.

```bash
cd enhanced
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy scipy 'opencv-python-headless<5'
PYTHONPATH=. python tests/test_all.py
```

To rerun the CPU ablations and regenerate the results and shaders, run the following from `enhanced/`.

```bash
PYTHONPATH=. python run_ablations.py
```

See `RESULTS.md` for measured ablation numbers and `shaders/` for exported baseline vs. fused GLSL shaders.
