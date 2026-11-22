# ArtCNN shaders

Start with the [optimized shader catalog](Optimized/README.md) for playback setup, model selection, and the fork's tested 2× luma shaders.

The six shaders in this directory are the original Artoriuz/ArtCNN C4F16 and C4F32 presets. Each width has a neutral shader, a `_DN` shader tuned to denoise and soften, and a `_DS` shader tuned to denoise and sharpen. These files are also the upstream controls in the [GPU benchmark](../enhanced/PARETO.md).

The [experimental YCbCr shaders](Experiments/README.md) handle chroma subsampling as part of upscaling.
